import numpy as np
from numba import njit

# -------------------------
# Lorenz drift, divergence
# -------------------------

@njit(cache=True, fastmath=True)
def Lorenz_drift(x, y, z, r, s, b):
    # A = (s(y-x), rx - y - zx, xy - b z)
    Ax = s * (y - x)
    Ay = r * x - y - z * x
    Az = x * y - b * z
    return Ax, Ay, Az

@njit(cache=True, fastmath=True)
def Lorenz_divergence(r, s, b):
    # div A = dAx/dx + dAy/dy + dAz/dz = -s - 1 - b  (constant)
    return -(s + 1.0 + b)

# --------------------------------------------
# Noise prep: Cholesky of 2 D dt_sub (3x3 SPD)
# --------------------------------------------

@njit(cache=True, fastmath=True)
def cholesky_psd(M):
    # robust small helper: if tiny negative eigs, project to PSD
    # (for well-formed D you won't hit this)
    try:
        return np.linalg.cholesky(M)
    except:
        S = 0.5 * (M + M.T)
        w, V = np.linalg.eigh(S)
        for i in range(3):
            if w[i] < 0.0:
                w[i] = 0.0
        Spsd = (V * w) @ V.T
        eps = 1e-12
        return np.linalg.cholesky(Spsd + eps * np.eye(3))

# --------------------------------------------
# One substep (EM and Heun), additive noise
# --------------------------------------------

@njit(cache=True, fastmath=True)
def _substep_em(x, y, z, dt, Ldt, r, s, b):
    Ax, Ay, Az = Lorenz_drift(x, y, z, r, s, b)
    xi = np.random.randn(3)
    nx, ny, nz = Ldt @ xi
    x_new = x + dt * Ax + nx
    y_new = y + dt * Ay + ny
    z_new = z + dt * Az + nz
    return x_new, y_new, z_new

@njit(cache=True, fastmath=True)
def _substep_heun(x, y, z, dt, Ldt, r, s, b):
    # predictor
    Ax0, Ay0, Az0 = Lorenz_drift(x, y, z, r, s, b)
    xi = np.random.randn(3)
    nx, ny, nz = Ldt @ xi
    xp = x + dt * Ax0 + nx
    yp = y + dt * Ay0 + ny
    zp = z + dt * Az0 + nz
    # corrector
    Ax1, Ay1, Az1 = Lorenz_drift(xp, yp, zp, r, s, b)
    x_new = x + 0.5 * dt * (Ax0 + Ax1) + nx
    y_new = y + 0.5 * dt * (Ay0 + Ay1) + ny
    z_new = z + 0.5 * dt * (Az0 + Az1) + nz
    return x_new, y_new, z_new

# --------------------------------------------
# Simulator with oversampling
# --------------------------------------------

def simulate_Lorenz(
    T, dt, D,
    x0=1.0, y0=1.0, z0=1.0,
    r=10.0, s=3.0, b=1.0,
    scheme="Heun", oversampling=1, seed=None
):
    """
    Simulate the noisy Lorenz SDE for T output steps of size dt.
    Each output step is resolved with `oversampling` substeps of size dt/oversampling.

    Returns
    -------
    t : (T+1,) array
    X : (T+1, 3) trajectory
    Dinv : (3,3) inverse of diffusion (for downstream analysis)
    divA_const : float, divergence of drift (constant for Lorenz)
    """
    if seed is not None:
        np.random.seed(seed)

    # substep dt and noise Cholesky
    m = max(1, int(oversampling))
    dt_sub = dt / m
    Sigma_sub = 2.0 * D * dt_sub
    Ldt = cholesky_psd(Sigma_sub)

    # choose integrator
    heun = scheme.lower().startswith("h")

    # divergence (constant)
    divA_const = Lorenz_divergence(r, s, b)

    # allocate
    X = np.empty((T + 1, 3), dtype=np.float64)
    t = np.linspace(0.0, T * dt, T + 1)
    x, y, z = x0, y0, z0


    # "equilibration"
    for _ in range(10000):
        x, y, z = _substep_em(x, y, z, dt_sub, Ldt, r, s, b)

    # simulation for sampling
    X[0, 0] = x; X[0, 1] = y; X[0, 2] = z
    for n in range(T):
        for _ in range(m):
            if heun:
                x, y, z = _substep_heun(x, y, z, dt_sub, Ldt, r, s, b)
            else:
                x, y, z = _substep_em(x, y, z, dt_sub, Ldt, r, s, b)
        X[n + 1, 0] = x; X[n + 1, 1] = y; X[n + 1, 2] = z

    return t, X, divA_const

# --------------------------------------------
# Observables: sigma, Tr, G
# --------------------------------------------

@njit(cache=True, fastmath=True)
def _matvec(Dinv, v):
    # 3x3 matvec, numba-friendly
    return np.array([
        Dinv[0,0]*v[0] + Dinv[0,1]*v[1] + Dinv[0,2]*v[2],
        Dinv[1,0]*v[0] + Dinv[1,1]*v[1] + Dinv[1,2]*v[2],
        Dinv[2,0]*v[0] + Dinv[2,1]*v[1] + Dinv[2,2]*v[2],
    ])

@njit(cache=True, fastmath=True)
def observables_from_trajectory_Lorenz(X, dt, Dinv, r, s, b):
    """
    Compute:
      sigma = < (D^{-1} A) ∘ Δx >/dt      (Stratonovich midpoint)
      Tr    = < A^T D^{-1} A >/4 + <divA>/2
      G     = - <divA>
    where A is Lorenz drift, divA is constant = -(s+1+b).
    """
    Tsteps = X.shape[0] - 1
    s_sigma = 0.0
    s_quad  = 0.0
    divA = Lorenz_divergence(r, s, b)  # constant

    for k in range(Tsteps):
        xk, yk, zk   = X[k]
        xkp,ykp,zkp = X[k+1]

        # drift at ends
        A0 = np.empty(3); A1 = np.empty(3)
        Ax0, Ay0, Az0 = Lorenz_drift(xk,  yk,  zk,  r, s, b)
        Ax1, Ay1, Az1 = Lorenz_drift(xkp, ykp, zkp, r, s, b)
        A0[0]=Ax0; A0[1]=Ay0; A0[2]=Az0
        A1[0]=Ax1; A1[1]=Ay1; A1[2]=Az1

        # midpoint for Stratonovich
        Am = 0.5 * (A0 + A1)
        dx = np.array([xkp - xk, ykp - yk, zkp - zk])

        # sigma increment
        v = _matvec(Dinv, Am)
        s_sigma += (v[0]*dx[0] + v[1]*dx[1] + v[2]*dx[2]) / dt

        # quadratic form at (left) endpoint for Tr
        v0 = _matvec(Dinv, A0)
        s_quad += (A0[0]*v0[0] + A0[1]*v0[1] + A0[2]*v0[2])

    mean_quad = s_quad / Tsteps
    mean_divA = divA              # constant; averaging is trivial

    sigma = s_sigma / Tsteps
    Tr    = 0.25 * mean_quad + 0.5 * mean_divA
    G     = - mean_divA
    return sigma, Tr, G


@njit(cache=True, fastmath=True)
def sigma_partial_from_trajectory_Lorenz(X, dt, Dinv, r, s, b, mask):
    """
    Known-force, grid-free share of sigma restricted to a subset of
    components, sigma_S = < sum_{i in S} (D^{-1} A)^i (dx^i/dt) >
    (Stratonovich midpoint), using the FULL known drift A(x,y,z) and the
    FULL D^{-1} -- only the final sum over components is restricted, via
    `mask` (length-3 array of 0./1., 1. for components in S).

    Unlike the grid-based v^2-method's own block-restricted quadratic form
    sigma^S = sum_{i,j in S} (D^{-1})^{ij} <nu^i nu^j> (nu the estimated
    local mean velocity), this is a DIFFERENT, generally non-identical
    exact share: it is additive over ANY partition of components by
    construction (mask_S + mask_Sc = 1 termwise, so sigma_S+sigma_Sc =
    sigma exactly, regardless of whether D is block-diagonal), and it
    needs no spatial grid or occupation binning at all, so it carries none
    of the usual cell-size discretization bias. See the two-way check in
    03_lorenz.ipynb: mask=[1,1,1] reproduces
    observables_from_trajectory_Lorenz's own sigma exactly.

    IMPORTANT: this is NOT, in general, the sigma_S = <v_S . D_S^{-1} v_S>
    of Eq.~(sigmaS)/Eq.~(app:sigmaS) in the paper (the one the improved-bound
    formulas are built on) -- the two coincide only when D happens to be
    block-diagonal with respect to S and its complement (as in the (x,y)/z
    split used in 03_lorenz.ipynb, where D^{xz}=D^{yz}=0). When the noise
    couples an observed coordinate to a hidden one (e.g. S={x} alone, with
    D^{xy}!=0), this additive mask-restricted share and sigma_S differ, and
    only sigma_S is bounded from below by the partial CYNAR/v^2 estimates.
    For sigma_S itself, use `v2method.probability_flux_sigma_subsets` instead
    (grid-based, restricted-block v^2-method quantity, valid for any S).
    """
    Tsteps = X.shape[0] - 1
    s_sigma = 0.0
    for k in range(Tsteps):
        xk, yk, zk   = X[k]
        xkp,ykp,zkp = X[k+1]

        A0 = np.empty(3); A1 = np.empty(3)
        Ax0, Ay0, Az0 = Lorenz_drift(xk,  yk,  zk,  r, s, b)
        Ax1, Ay1, Az1 = Lorenz_drift(xkp, ykp, zkp, r, s, b)
        A0[0]=Ax0; A0[1]=Ay0; A0[2]=Az0
        A1[0]=Ax1; A1[1]=Ay1; A1[2]=Az1

        Am = 0.5 * (A0 + A1)
        dx = np.array([xkp - xk, ykp - yk, zkp - zk])

        v = _matvec(Dinv, Am)
        s_sigma += mask[0]*v[0]*dx[0]/dt + mask[1]*v[1]*dx[1]/dt + mask[2]*v[2]*dx[2]/dt

    return s_sigma / Tsteps
