# =============================
# correl.py
# =============================
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from scipy.optimize import least_squares
from .cfg import (
    TrafficCfg,
    ExpPolyCfg,
    WindowCfg,
)

# ---- correlation utilities ----

def _next_pow2(x: int) -> int:
    return 1 << int(np.ceil(np.log2(max(1, x))))

def correlations_fft(X: np.ndarray, max_lag: int, step: int = 1, demean: bool = True) -> np.ndarray:
    """Cross-correlations C[i,j,k] = E[(X_t[i]-mu_i)(X_{t+k}[j]-mu_j)] for k>=0.
    Uses FFT; memory-safe padding with nfft=next_pow2(2*T-1).
    """
    T, N = X.shape
    Xc = X - X.mean(axis=0, keepdims=True) if demean else X
    nfft = _next_pow2(2*T - 1)
    F = np.fft.rfft(Xc, n=nfft, axis=0)              # (nfft//2+1, N)
    S = F[:, :, None] * np.conj(F[:, None, :])       # (nfft//2+1, N, N)
    c = np.fft.irfft(S, n=nfft, axis=0).real         # (nfft, N, N)
    c = c[:T, :, :]
    counts = np.arange(T, 0, -1, dtype=float)[:, None, None]
    c = c / counts
    idx = np.arange(0, min(T-1, max_lag) + 1, step)
    C = np.transpose(c[idx], (1, 2, 0))              # (N,N,L)
    return C

def symmetrize_C(C: np.ndarray) -> np.ndarray:
    return 0.5 * (C + np.transpose(C, (1, 0, 2)))

# ---- window selection ----

def auto_window(cs_ij: np.ndarray, i1: int, H: float) -> Tuple[int, int]:
    """
    Choose [i1, i2] so that |C(τ)| stays within (1±H) of |C(i1)| before decaying.
    H in (0,1): larger H -> wider window tolerance.
    """
    i2 = i1
    C1 = abs(cs_ij[i1])
    if C1 == 0:
        return i1, min(i1 + 5, len(cs_ij) - 1)
    while (
        i2 < len(cs_ij) - 1
        and abs(cs_ij[i2]) / (C1 + 1e-300) > 1 - H
        and abs(cs_ij[i2]) / (C1 + 1e-300) < 1 + H
    ):
        i2 += 1
    return i1, max(i2, i1 + 3)

def exponential_binning(t, C, i_min, expansion_factor):
    """
    Exponential binning of (t, C) after copying the first i_min points verbatim.
    Non-positive t's after i_min are also copied verbatim; exponential bins start
    at the first strictly positive t. Returns (tt, CC).
    """
    t = np.asarray(t); C = np.asarray(C)
    L = len(t)
    if L != len(C): raise ValueError("t and C must have equal length.")
    if not (0 <= i_min <= L): raise ValueError(f"i_min must be in [0, {L}].")
    if i_min == L: return t.copy(), C.copy()
    if expansion_factor <= 1.0: raise ValueError("expansion_factor must be > 1.0.")

    # Prefix copied as-is
    tt, CC = list(t[:i_min]), list(C[:i_min])

    # Tail to process
    t_tail = t[i_min:]; C_tail = C[i_min:]
    if t_tail.size == 0: return np.array(tt), np.array(CC)

    # Pass through any non-positive t's, then begin exp-binning at first t>0
    pos_mask = t_tail > 0
    if not np.any(pos_mask):  # nothing positive to bin
        tt.extend(t_tail); CC.extend(C_tail)
        return np.array(tt), np.array(CC)

    first_pos = np.argmax(pos_mask)           # index of first True
    if first_pos > 0:                         # copy the non-positive chunk
        tt.extend(t_tail[:first_pos]); CC.extend(C_tail[:first_pos])

    t_pos = t_tail[first_pos:]; C_pos = C_tail[first_pos:]
    t0, tmax = t_pos[0], t_pos[-1]
    log_alpha = np.log(expansion_factor)

    # Build edges up to >= tmax*alpha so the last point is captured cleanly
    m = int(np.ceil(np.log((tmax / t0) * expansion_factor) / log_alpha))
    if m < 1:  # only one positive point (or all equal) -> one averaged bin
        tt.append(float(t_pos.mean())); CC.append(float(C_pos.mean()))
        return np.array(tt), np.array(CC)

    edges = t0 * (expansion_factor ** np.arange(m + 1))  # length m+1 -> m bins

    # Assign points to bins [edges[k], edges[k+1])
    bins = np.digitize(t_pos, edges, right=False)  # in 1..m (0 or >m ignored)

    # Vectorized bin means via bincount
    cnt = np.bincount(bins, minlength=m + 1)
    sum_t = np.bincount(bins, weights=t_pos, minlength=m + 1)
    sum_C = np.bincount(bins, weights=C_pos, minlength=m + 1)

    valid = (np.arange(m + 1) >= 1) & (np.arange(m + 1) <= m) & (cnt > 0)
    tt_avg = sum_t[valid] / cnt[valid]
    CC_avg = sum_C[valid] / cnt[valid]

    tt.extend(tt_avg.tolist()); CC.extend(CC_avg.tolist())
    return np.array(tt), np.array(CC)

# -----------------------------------------------
# ---- exp–poly model and fitting ----

def _exp_poly_model(t, params, Ne, Ng, clip_exp):
    r = np.asarray(params[:Ne])
    s = np.asarray(params[Ne:2*Ne])
    B = params[2*Ne:]
    f = np.zeros_like(t, dtype=float)
    off = 0
    for n in range(Ne):
        b = np.zeros(Ng + 1)
        b[:] = B[off:off + Ng + 1]; off += (Ng + 1)
        P = np.zeros_like(t, dtype=float)
        for k in range(Ng + 1):
            P += b[k] * (t ** k)
        z = -r[n] * t - s[n] * t * t
        z = np.clip(z, -clip_exp, clip_exp)
        f += np.exp(z) * P
    return f

def _exp_poly_derivs0(params, Ne, Ng):
    r = np.asarray(params[:Ne])
    s = np.asarray(params[Ne:2*Ne])
    B = params[2*Ne:]
    f1 = 0.0; f2 = 0.0
    off = 0
    for n in range(Ne):
        b = np.zeros(Ng + 1)
        b[:] = B[off:off + Ng + 1]; off += (Ng + 1)
        b0 = b[0]; b1 = b[1] if Ng >= 1 else 0.0; b2 = b[2] if Ng >= 2 else 0.0
        f1 += (-r[n]) * b0 + b1
        f2 += (r[n]**2 - 2.0 * s[n]) * b0 - 2.0 * r[n] * b1 + 2.0 * b2
    return float(f1), float(f2)

def fit_exp_poly_times(t: np.ndarray, y: np.ndarray, cfg: ExpPolyCfg):
    """
    Irregular-times version of the exp–poly fit.
    Fits model on the exact time stamps t (no dt, no index window).
    Returns: ((cp, cpp), fitC, t) where cp=C'(0), cpp=C''(0), fitC=model(t).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
        raise ValueError("t and y must be 1D arrays of the same length.")
    if len(t) < 2:
        return (0.0, 0.0), y.copy(), t.copy()

    if not np.all(np.diff(t) >= 0):
        idx = np.argsort(t)
        t = t[idx]; y = y[idx]

    Ne, Ng, clipE = cfg.Ne, cfg.Ng, cfg.clip_exp

    # Simple initialization
    r0 = np.linspace(1.0, 4.0, Ne)
    s0 = 1e-3 * np.linspace(1.0, 2.0, Ne)
    A0 = []
    for _ in range(Ne):
        A0 += [y[0] / max(Ne, 1), 0.0] + ([0.0] if Ng >= 2 else [])
    p0 = np.concatenate([r0, s0, np.array(A0, dtype=float)])

    bounds_low = np.concatenate([np.zeros(Ne), np.zeros(Ne), -np.full(Ne * (Ng + 1), np.inf)])
    bounds_high = np.concatenate([np.full(Ne, np.inf), np.full(Ne, np.inf), np.full(Ne * (Ng + 1), np.inf)])

    def resid(p):
        return _exp_poly_model(t, p, Ne, Ng, clipE) - y

    res = least_squares(
        resid, p0,
        max_nfev=cfg.max_iter,
        bounds=(bounds_low, bounds_high),
        method="trf",
    )

    fitC = _exp_poly_model(t, res.x, Ne, Ng, np.inf)
    return _exp_poly_derivs0(res.x, Ne, Ng), fitC, t

def fit_exp_poly(cs_ij: np.ndarray, dt: float, i1: int, i2: int, cfg: ExpPolyCfg):
    """
    Backward-compatible wrapper: builds (t, y) from dt/i1/i2 and delegates to fit_exp_poly_times.
    """
    t = dt * np.arange(i1, i2 + 1)
    y = cs_ij[i1:i2 + 1]
    return fit_exp_poly_times(t, y, cfg)
