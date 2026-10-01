import numpy as np
from numba import njit

# indices into the pars array
S_FEED, F_MAX, K_GS, K_SP, D_GATE, DELTAG, N_EL, MU1, MU2, KBT, KBT_EFF = range(11)

@njit(cache=True, fastmath=True)
def p0_stable(z, pars):
    # a = (k_gs*D_gate/(N*kBT)) * z
    a = (pars[K_GS]*pars[D_GATE] / (pars[N_EL]*pars[KBT])) * z
    # logA = (DeltaG + k_gs*D^2/(2N)) / (kBT)
    logA = (pars[DELTAG] + pars[K_GS]*pars[D_GATE]*pars[D_GATE]/(2.0*pars[N_EL]))/pars[KBT]
    # logistic in a - logA for numerical stability
    u = a - logA
    # stable sigmoid
    if u >= 0.0:
        eu = np.exp(-u)
        return 1.0/(1.0 + eu)
    else:
        eu = np.exp(u)
        return eu/(1.0 + eu)

@njit(cache=True, fastmath=True)
def forces_hair(x, y, pars):
    """Return Fx, Fy (physical forces) WITHOUT mobilities."""
    z = x - y
    P0 = p0_stable(z, pars)
    kgs = pars[K_GS]; ksp = pars[K_SP]; Dg = pars[D_GATE]
    # Fx = -dV/dx
    Fx = -kgs*z - ksp*x + kgs*Dg*P0
    # Fy = -dV/dy - F_act
    Fy =  kgs*z - kgs*Dg*P0 - pars[F_MAX]*(1.0 - pars[S_FEED]*P0)
    return Fx, Fy

@njit(cache=True, fastmath=True)
def _step_em(x, y, dt, pars, rng_state):
    # Mobilities
    mu1 = pars[MU1]; mu2 = pars[MU2]
    # Thermal noise strengths: sqrt(2 kBT mu dt)
    s1 = np.sqrt(2.0*pars[KBT]*mu1*dt)
    s2 = np.sqrt(2.0*pars[KBT_EFF]*mu2*dt)
    # Drift = mu * Force
    Fx, Fy = forces_hair(x, y, pars)
    x_new = x + dt*(mu1*Fx) + s1*np.random.randn()
    y_new = y + dt*(mu2*Fy) + s2*np.random.randn()
    return x_new, y_new

@njit(cache=True, fastmath=True)
def _step_heun(x, y, dt, pars, rng_state):
    mu1 = pars[MU1]; mu2 = pars[MU2]
    s1 = np.sqrt(2.0*pars[KBT]*mu1*dt)
    s2 = np.sqrt(2.0*pars[KBT_EFF]*mu2*dt)
    # noise is shared between predictor/corrector for strong order-1/2
    xi1 = np.random.randn(); xi2 = np.random.randn()
    Fx0, Fy0 = forces_hair(x, y, pars)
    x_pred = x + dt*(mu1*Fx0) + s1*xi1
    y_pred = y + dt*(mu2*Fy0) + s2*xi2
    Fx1, Fy1 = forces_hair(x_pred, y_pred, pars)
    x_new = x + 0.5*dt*mu1*(Fx0 + Fx1) + s1*xi1
    y_new = y + 0.5*dt*mu2*(Fy0 + Fy1) + s2*xi2
    return x_new, y_new

@njit(cache=True, fastmath=True)
def _is_stable_small_dt(pars, dt):
    # crude sanity: elastic frequencies << 1/dt helps with stability
    # (not a rigorous test; you can tune as needed)
    kgs = pars[K_GS]; ksp = pars[K_SP]
    mu1 = pars[MU1]; mu2 = pars[MU2]
    lam1 = mu1*(kgs + ksp)
    lam2 = mu2*(kgs)
    return (dt*lam1 < 0.2) and (dt*lam2 < 0.2)

def simulate_hair_bundle(T, dt, x0, y0, pars, scheme="EM", seed=None, T_conv=-1):
    """
    Simulate:
      dx = mu1*F_x dt + sqrt(2 kBT mu1) dW1
      dy = mu2*F_y dt + sqrt(2 kBT_eff mu2) dW2
    Returns t, X (shape [T+1,2]).
    """
    if seed is not None:
        np.random.seed(seed)
    if scheme.lower().startswith("h"): HEUN = True
    else: HEUN = False

    T = int(T)
    if T_conv<0: T_conv = T//100
    out = np.empty((T+1, 2), dtype=np.float64)

    # quick stability hint
    if not _is_stable_small_dt(pars, dt):
        pass  # optionally warn/log

    # convergence to the steady state, quicker with EM
    x, y = x0, y0
    for k in range(T_conv):
        x, y = _step_em(x, y, dt, pars, 0)
    x1 = out[0,0] = x; y1 = out[0,1] = y

    if HEUN:  # Heun
        for k in range(T):
            x, y = _step_heun(x, y, dt, pars, 0)
            out[k+1,0] = x; out[k+1,1] = y
    else:  # Euler–Maruyama
        for k in range(T):
            x, y = _step_em(x, y, dt, pars, 0)
            out[k+1,0] = x; out[k+1,1] = y

    t = np.linspace(0.0, T*dt, T+1)
    return t, out



@njit(cache=True, fastmath=True)
def drift_A(x, y, pars):
    """A(x) = (mu1*Fx, mu2*Fy)."""
    Fx, Fy = forces_hair(x, y, pars)
    return pars[MU1]*Fx, pars[MU2]*Fy

@njit(cache=True, fastmath=True)
def divA(x, y, pars):
    """Divergence of A: ∂A1/∂x + ∂A2/∂y (closed form)."""
    kgs = pars[K_GS]; ksp = pars[K_SP]; Dg = pars[D_GATE]
    mu1 = pars[MU1];  mu2 = pars[MU2]
    Fmax = pars[F_MAX]; S = pars[S_FEED]
    # helper
    z  = x - y
    P0 = p0_stable(z, pars)
    sigp = P0*(1.0 - P0)
    c = (kgs*Dg) / (pars[N_EL]*pars[KBT])

    dFxdx = -kgs - ksp + kgs*Dg*c*sigp
    dFydy = -kgs + c*sigp*(kgs*Dg - Fmax*S)
    return mu1*dFxdx + mu2*dFydy

@njit(cache=True, fastmath=True)
def Dinv_diag(pars):
    """Return the diagonal entries of D^{-1}."""
    d1_inv = 1.0 / (pars[KBT]     * pars[MU1])
    d2_inv = 1.0 / (pars[KBT_EFF] * pars[MU2])
    return d1_inv, d2_inv

@njit(cache=True, fastmath=True)
def observables_from_trajectory(X, dt, pars):
    """
    Inputs
      X: (T+1, 2) array of states along time (constant dt).
    Returns
      sigma_rate, Tr_rate, G_rate
    """
    Tsteps = X.shape[0] - 1
    d1_inv, d2_inv = Dinv_diag(pars)

    # accumulators
    s_sigma = 0.0   # for Stratonovich sigma estimator
    s_A_Dinv_A = 0.0
    s_divA = 0.0

    # first sample for Tr/G:
    for k in range(Tsteps):
        xk, yk = X[k, 0],   X[k, 1]
        xkp,ykp= X[k+1,0], X[k+1,1]

        # A at endpoints
        A1k, A2k   = drift_A(xk,  yk,  pars)
        A1kp,A2kp  = drift_A(xkp, ykp, pars)
        # midpoint for Stratonovich
        A1m = 0.5*(A1k + A1kp)
        A2m = 0.5*(A2k + A2kp)
        dx1 = xkp - xk
        dx2 = ykp - yk

        # sigma via midpoint rule: ((D^{-1} A_mid) · Δx_k) / dt
        s_sigma += (d1_inv*A1m*dx1 + d2_inv*A2m*dx2) / dt

        # A^T D^{-1} A at, say, left endpoint (Riemann)
        s_A_Dinv_A += (d1_inv*A1k*A1k + d2_inv*A2k*A2k)

        # divergence at left endpoint
        s_divA += divA(xk, yk, pars)

    # time averages
    Ttot = Tsteps * dt
    mean_A_Dinv_A = s_A_Dinv_A / Tsteps
    mean_divA     = s_divA / Tsteps
    sigma_rate    = s_sigma / Tsteps                     # ≈ <A^T D^{-1} A>
    Tr_rate       = 0.25*mean_A_Dinv_A + 0.5*mean_divA
    G_rate        = - mean_divA

    return sigma_rate, Tr_rate, G_rate
