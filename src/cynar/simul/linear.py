# =============================
# linear.py
# =============================
from typing import Tuple, Optional, List, Dict, Any, Callable
import numpy as np
from numba import njit

# ---------- stability + noise factor helpers (NumPy, run once on CPU) ----------

def check_stability(A: np.ndarray, D: np.ndarray, tol: float = 1e-12):
    """
    Returns:
      hurwitz_ok:   all Re(eig(A)) < 0  (strict)
      D_psd_ok:     D is symmetric PSD within tol
      eigA:         eigenvalues of A
      eigD:         eigenvalues of symmetrized D
    """
    A = np.asarray(A, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    eigA = np.linalg.eigvals(A)
    hurwitz_ok = np.all(np.real(eigA) < 0.0)

    Dsym = 0.5 * (D + D.T)
    eigD = np.linalg.eigvalsh(Dsym)
    D_psd_ok = np.all(eigD >= -tol)  # allow tiny negative due to roundoff
    return hurwitz_ok, D_psd_ok, eigA, eigD

def make_noise_factor(D: np.ndarray, dt_int: float, tol: float = 1e-12):
    """
    Build L such that for z~N(0,I), L z ~ N(0, 2 D dt_int).
    Tries Cholesky; if not PD, falls back to eigen clamp for PSD.
    """
    D = np.asarray(D, dtype=np.float64)
    Sigma = 2.0 * dt_int * 0.5 * (D + D.T)  # ensure symmetry

    # Fast path: Cholesky for (strictly) PD Sigma
    try:
        L = np.linalg.cholesky(Sigma)
        return L.astype(np.float64, copy=False)
    except np.linalg.LinAlgError:
        pass

    # PSD path: eigen-decompose, clamp tiny negatives, build L = Q sqrt(Lam) Q^T
    w, Q = np.linalg.eigh(Sigma)
    w_clamped = np.clip(w, 0.0, None)
    # Construct L via symmetric sqrt: L = Q * sqrt(Λ) * Q^T
    sqrt_w = np.sqrt(w_clamped)
    # Use Q * diag(sqrt_w) without forming dense diag twice
    L = (Q * sqrt_w) @ Q.T
    # For numerical symmetry
    L = 0.5 * (L + L.T)
    return L.astype(np.float64, copy=False)


# ---------- core simulator (Numba JIT) ----------

@njit(cache=True, fastmath=True)
def _simulate_linear_diffusion_kernel(T, A, L, dt_int, oversampling, method_flag, seed, x0):
    """
    method_flag: 0 -> Euler–Maruyama, 1 -> Heun (predictor-corrector, additive noise)
    """
    np.random.seed(seed)
    N = A.shape[0]
    X = np.zeros((T, N), dtype=np.float64)
    x = x0.copy()

    for t in range(1, T):
        for s in range(oversampling):
            # shared noise for additive-noise Heun (same dW in predictor/corrector)
            z = np.random.randn(N)
            y = L.dot(z)

            if method_flag == 0:
                # Euler–Maruyama
                drift = A.dot(x)
                x += drift * dt_int + y
            else:
                # Heun (predictor-corrector using same y)
                drift = A.dot(x)
                x_pred = x + drift * dt_int + y
                drift_pred = A.dot(x_pred)
                x += 0.5 * (drift + drift_pred) * dt_int + y

        # record at coarse grid
        for i in range(N):
            X[t, i] = x[i]

    return X


# ---------- user-facing wrapper ----------

def simulate_linear_diffusion(
    T: int,
    A: np.ndarray,
    D: np.ndarray,
    dt: float,
    *,
    oversampling: int = 1,
    method: str = "euler",      # "euler" or "heun"
    seed: int = 0,
    x0: np.ndarray | None = None,
    check: bool = True,
    raise_on_unstable: bool = True,
):
    """
    Simulate N-D linear diffusion dx = A x dt + sqrt(2D) dW.

    Args:
      T             : number of recorded samples
      A (N,N)       : drift matrix
      D (N,N)       : diffusion (covariance) matrix (symmetric PSD)
      dt            : recording step (coarse); internal step is dt/oversampling
      oversampling  : # of internal substeps per recorded sample
      method        : "euler" or "heun" (Heun uses same noise in both stages; additive noise)
      seed          : RNG seed
      x0 (N,)       : initial condition (defaults to zeros)
      check         : if True, verify stability & PSD
      raise_on_unstable: if True, raise if checks fail; else print a warning and continue.

    Returns:
      X (T, N): trajectory
    """
    A = np.asarray(A, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    N = A.shape[0]
    assert A.shape == (N, N) and D.shape == (N, N), "A and D must be square NxN."

    if x0 is None:
        x0 = np.zeros(N, dtype=np.float64)
    else:
        x0 = np.asarray(x0, dtype=np.float64).reshape(N)

    # Stability & PSD checks
    if check:
        hurwitz_ok, D_psd_ok, eigA, eigD = check_stability(A, D)
        msgs = []
        if not hurwitz_ok:
            msgs.append(
                f"A is not Hurwitz (max Re(eig(A)) = {np.max(np.real(eigA)):.3e}). "
                "Stationary covariance may not exist."
            )
        if not D_psd_ok:
            msgs.append(
                f"D is not PSD within tolerance (min eig(D_sym) = {np.min(eigD):.3e})."
            )
        if msgs:
            msg = " | ".join(msgs)
            if raise_on_unstable:
                raise ValueError(msg)
            else:
                print("Warning:", msg)

    # Build per-substep noise factor L so that L z ~ N(0, 2 D dt_int)
    dt_int = dt / float(max(1, oversampling))
    L = make_noise_factor(D, dt_int)

    # Method flag for Numba
    m = method.lower().strip()
    if m in ("euler", "em", "euler-maruyama"):
        method_flag = 0
    elif m in ("heun", "pc", "predictor-corrector"):
        method_flag = 1
    else:
        raise ValueError("method must be 'euler' or 'heun'.")

    # Run JIT kernel
    X = _simulate_linear_diffusion_kernel(
        int(T),
        A.astype(np.float64, copy=False),
        L.astype(np.float64, copy=False),
        float(dt_int),
        int(max(1, oversampling)),
        int(method_flag),
        int(seed),
        x0.astype(np.float64, copy=False),
    )
    return X


# ---------- closed-form theory for the 2D rotational-drift example ("linear_vortex") ----------

def theory_linear_2N(A: np.ndarray, D: np.ndarray):
    """
    Closed-form sigma, traffic, inflow for the 2D linear drift
    A = [[-alpha, -Phi], [alpha*Phi, -1]], diagonal D = diag(T1, T2).
    """
    A11, A22, A12, A21 = A[0, 0], A[1, 1], A[0, 1], A[1, 0]
    T1, T2 = D[0, 0], D[1, 1]

    def entr_th_2025(A11, A22, A12, A21, T1, T2):
        return - (A21*T1 - A12*T2)**2 / (T1*T2*(A11 + A22))

    def traff_th_2025(A11, A22, A12, A21, T1, T2):
        return (entr_th_2025(A11, A22, A12, A21, T1, T2) + (A11 + A22)) / 4.0

    Tr_true = traff_th_2025(A11, A22, A12, A21, T1, T2)
    sigma_true = entr_th_2025(A11, A22, A12, A21, T1, T2)
    G_true = sigma_true - 4*Tr_true
    return Tr_true, sigma_true, G_true


def theory_linear_general(A: np.ndarray, D: np.ndarray):
    """
    Closed-form sigma, traffic, inflow for a general stable N-D linear drift
    dx = A x dt + sqrt(2D) dW (constant A, D; D need not be diagonal, A need
    not be 2x2). Generalizes theory_linear_2N, which is restricted to a 2x2
    A with diagonal D, to arbitrary dimension.

    The stationary law is Gaussian with covariance Sigma solving the
    Lyapunov equation A Sigma + Sigma A^T + 2D = 0, giving the exact
    steady-state score s(x) = -Sigma^{-1} x and local mean velocity
    nu(x) = (A + D Sigma^{-1}) x, from which:
        G_true     = Tr[Sigma^{-1} D]                                   (= <s^T D s>)
        sigma_true = Tr[(A + D Sigma^{-1})^T D^{-1} (A + D Sigma^{-1}) Sigma]  (= <nu^T D^{-1} nu>)
        Tr_true    = (sigma_true - G_true) / 4
    """
    from scipy.linalg import solve_continuous_lyapunov
    A = np.asarray(A, dtype=float)
    D = np.asarray(D, dtype=float)
    Sigma = solve_continuous_lyapunov(A, -2.0 * D)
    Sigma_inv = np.linalg.inv(Sigma)
    D_inv = np.linalg.inv(D)

    G_true = float(np.trace(Sigma_inv @ D))
    nu_mat = A + D @ Sigma_inv
    sigma_true = float(np.trace(nu_mat.T @ D_inv @ nu_mat @ Sigma))
    Tr_true = (sigma_true - G_true) / 4.0
    return Tr_true, sigma_true, G_true
