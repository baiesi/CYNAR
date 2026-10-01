import numpy as np
from numba import njit

# --- Telegraph & drift (same as before) ---
@njit(cache=True, fastmath=True)
def telegraph_step(theta, dt, w0, w1):
    if theta == 0:
        if np.random.rand() < w0*dt:
            return 1
        return 0
    else:
        if np.random.rand() < w1*dt:
            return 0
        return 1

@njit(cache=True, fastmath=True)
def drift_A(x, theta, mu, kappa, dlam):
    return -mu * kappa * (x - dlam * theta)

# --- One substep: EM and Heun, returning pieces needed for estimators ---
@njit(cache=True, fastmath=True)
def substep_em_obs(x, theta, dt, mu, kappa, dlam, kBT, w0, w1):
    # Left-endpoint drift
    A0 = drift_A(x, theta, mu, kappa, dlam)
    # Noise increment
    xi = np.random.randn()
    n = np.sqrt(2.0*kBT*mu*dt) * xi
    # State update
    x_new = x + dt*A0 + n
    theta_new = telegraph_step(theta, dt, w0, w1)
    # For EM (Itô), the consistent Stratonovich estimator uses *midpoint* of A;
    # approximate with 0.5*(A0 + A1) to keep symmetry:
    A1 = drift_A(x_new, theta_new, mu, kappa, dlam)
    A_mid = 0.5*(A0 + A1)
    return x_new, theta_new, A0, A_mid, (x_new - x)

@njit(cache=True, fastmath=True)
def substep_heun_obs(x, theta, dt, mu, kappa, dlam, kBT, w0, w1):
    # Heun predictor-corrector with shared noise (Stratonovich-consistent)
    A0 = drift_A(x, theta, mu, kappa, dlam)
    xi = np.random.randn()
    n = np.sqrt(2.0*kBT*mu*dt) * xi
    # Predictor
    x_p = x + dt*A0 + n
    theta_p = telegraph_step(theta, dt, w0, w1)
    A1 = drift_A(x_p, theta_p, mu, kappa, dlam)
    # Corrector
    x_new = x + 0.5*dt*(A0 + A1) + n
    # Keep the predictor telegraph (piecewise-constant rates)
    theta_new = theta_p
    # Midpoint drift for Stratonovich is 0.5*(A0 + A1)
    A_mid = 0.5*(A0 + A1)
    return x_new, theta_new, A0, A_mid, (x_new - x)

def simulate_SST(
    T_steps, dt, *,
    mu=1.0, kappa=1.0, dlam=1.0, kBT=1.0,
    w0=1.0, w1=1.0,
    x0=0.0, theta0=0,
    scheme="Heun", oversampling=1, seed=None,
    return_observables=True
):
    """
    Simulate the stochastic switching trap and (optionally) compute sigma_sim, Tr_sim, G_sim
    *on the integration grid* to avoid coarse-grid bias.
    """
    if seed is not None:
        np.random.seed(seed)

    m = max(1, int(oversampling))
    dt_sub = dt / m
    use_heun = scheme.lower().startswith("h")

    X  = np.empty(T_steps+1, dtype=np.float64)
    TH = np.empty(T_steps+1, dtype=np.int32)
    t  = np.linspace(0.0, T_steps*dt, T_steps+1)

    x = x0
    th = theta0

    # relaxation
    for _ in range(1000):
         x, th, _, _ ,_ = substep_em_obs(x, th, dt_sub, mu, kappa, dlam, kBT, w0, w1)

    # starting point
    X[0] = x; TH[0] = th

    # Online accumulators (per substep)
    s_sigma = 0.0     # sum of (D^{-1} A_mid) * (dx_sub/dt_sub)
    s_quad  = 0.0     # sum of (A0^2 / D)
    invD = 1.0 / (kBT * mu)
    divA = -mu * kappa  # constant

    for n in range(T_steps):
        for _ in range(m):
            if use_heun:
                x_new, th_new, A0, A_mid, dx = substep_heun_obs(x, th, dt_sub, mu, kappa, dlam, kBT, w0, w1)
            else:
                x_new, th_new, A0, A_mid, dx = substep_em_obs(x, th, dt_sub, mu, kappa, dlam, kBT, w0, w1)

            # Stratonovich sigma increment at substep resolution
            s_sigma += invD * A_mid * (dx / dt_sub)
            # Quadratic form for Tr at left endpoint
            s_quad  += invD * (A0 * A0)

            x, th = x_new, th_new

        X[n+1]  = x
        TH[n+1] = th

    if not return_observables:
        return t, X, TH

    # Convert sums to time-averages
    total_substeps = T_steps * m
    sigma_sim = s_sigma / total_substeps
    mean_quad = s_quad  / total_substeps
    Tr_sim    = 0.25*mean_quad + 0.5*divA
    G_sim     = -divA

    return t, X, TH, sigma_sim, Tr_sim, G_sim

# Closed-form (same as before)
def theory_SST_sigma_Tr_G(mu, kappa, dlam, kBT, w0, w1):
    q = w0 / (w0 + w1)
    w = w0 + w1
    eps2 = (kappa*dlam)**2 * q * (1.0 - q)
    sigma_true = (mu / kBT) * eps2 / (1.0 + (kappa*mu)/w)
    G_true     = mu * kappa
    Tr_true    = 0.25*(sigma_true - G_true)
    return sigma_true, Tr_true, G_true
