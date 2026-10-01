import numpy as np
from numba import njit

# -------------------------
# Lorenz-96 drift and divergence
# x' = (x_{i+1}-x_{i-2})*x_{i-1} - x_i + F, indices mod N
# -------------------------

@njit(cache=True, fastmath=True)
def l96_drift(x, F):
    N = x.size
    A = np.empty_like(x)
    for i in range(N):
        im2 = (i - 2) % N
        im1 = (i - 1) % N
        ip1 = (i + 1) % N
        A[i] = (x[ip1] - x[im2]) * x[im1] - x[i] + F
    return A

@njit(cache=True, fastmath=True)
def l96_divergence(N):
    # sum_i dA_i/dx_i = -1 for each i -> total = -N
    return -float(N)

# -------------------------
# Robust Cholesky for 2*D*dt_sub
# -------------------------

@njit(cache=True, fastmath=True)
def cholesky_psd(M):
    # Try Cholesky, otherwise project tiny negatives
    try:
        return np.linalg.cholesky(M)
    except:
        S = 0.5*(M + M.T)
        w, V = np.linalg.eigh(S)
        for i in range(w.size):
            if w[i] < 0.0:
                w[i] = 0.0
        Spsd = (V * w) @ V.T
        eps = 1e-12
        return np.linalg.cholesky(Spsd + eps*np.eye(Spsd.shape[0]))

# -------------------------
# One substep (EM / Heun), additive noise
# -------------------------

@njit(cache=True, fastmath=True)
def _substep_em(x, dt, Ldt, F):
    A0 = l96_drift(x, F)
    xi = np.random.randn(x.size)
    n = Ldt @ xi
    return x + dt*A0 + n

@njit(cache=True, fastmath=True)
def _substep_heun(x, dt, Ldt, F):
    A0 = l96_drift(x, F)
    xi = np.random.randn(x.size)
    n = Ldt @ xi
    xp = x + dt*A0 + n
    A1 = l96_drift(xp, F)
    return x + 0.5*dt*(A0 + A1) + n

# -------------------------
# Simulator with oversampling
# -------------------------

def simulate_Lorenz96(
    T_steps, dt, D, x0=None, F=8.0,
    scheme="Heun", oversampling=1, seed=None
):
    """
    SDE: dx = A(x) dt + sqrt(2D) dW
    A_i(x) = (x_{i+1}-x_{i-2}) x_{i-1} - x_i + F

    Returns: t (T+1,), X (T+1,N), divA_const (float)
    """
    if seed is not None:
        np.random.seed(seed)
    N = D.shape[0]
    if x0 is None:
        x = F * np.ones(N)  # standard L96 init near fixed point
        x[0] += 0.01
    else:
        x = np.array(x0, dtype=np.float64)
    assert x.shape[0] == N

    m = max(1, int(oversampling))
    dt_sub = dt / m
    Sigma_sub = 2.0 * D * dt_sub
    Ldt = cholesky_psd(Sigma_sub)
    heun = scheme.lower().startswith("h")

    divA_const = l96_divergence(N)

    X = np.empty((T_steps+1, N), dtype=np.float64)
    t = np.linspace(0.0, T_steps*dt, T_steps+1)

    # "equilibration"
    for _ in range(10000):
        x = _substep_em(x, dt_sub, Ldt, F)

    # simulation for sampling
    X[0] = x.copy()
    for n in range(T_steps):
        for _ in range(m):
            if heun:
                x = _substep_heun(x, dt_sub, Ldt, F)
            else:
                x = _substep_em(x, dt_sub, Ldt, F)
        X[n+1] = x

    return t, X, divA_const

# -------------------------
# Observables: sigma, Tr, G (general N)
# -------------------------

@njit(cache=True, fastmath=True)
def _matvec(Dinv, v, out):
    N = v.size
    for i in range(N):
        s = 0.0
        for j in range(N):
            s += Dinv[i, j] * v[j]
        out[i] = s

@njit(cache=True, fastmath=True)
def observables_from_trajectory_96(X, dt, Dinv, divA_const, F):
    """
    sigma = < (D^{-1} A) ∘ Δx >/dt   (Stratonovich midpoint)
    Tr    = < A^T D^{-1} A >/4 + <divA>/2
    G     = - <divA>
    """
    Tsteps, N = X.shape[0]-1, X.shape[1]
    s_sigma = 0.0
    s_quad  = 0.0
    tmp = np.empty(N)

    for k in range(Tsteps):
        xk  = X[k]
        xkp = X[k+1]
        A0 = l96_drift(xk,  F)
        A1 = l96_drift(xkp, F)

        # Stratonovich midpoint
        Am = 0.5*(A0 + A1)
        dx = xkp - xk

        _matvec(Dinv, Am, tmp)
        inc = 0.0
        for i in range(N):
            inc += tmp[i] * dx[i]
        s_sigma += inc / dt

        # Quadratic form at left endpoint
        _matvec(Dinv, A0, tmp)
        q = 0.0
        for i in range(N):
            q += A0[i] * tmp[i]
        s_quad += q

    mean_quad = s_quad / Tsteps
    mean_divA = divA_const  # constant in L96
    sigma = s_sigma / Tsteps
    Tr    = 0.25*mean_quad + 0.5*mean_divA
    G     = -mean_divA
    return sigma, Tr, G
