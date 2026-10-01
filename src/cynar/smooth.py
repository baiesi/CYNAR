# smooth.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from .cfg import InflowCfg, SmoothingCfg

# ---------- 1) Scaling: "bw" mode (per-dimension bandwidth vector) ----------

def make_scaler_bw(bw_vec):
    """
    Return a scaler dict with .apply(A) = (A - mu) / bw, where mu is the mean of A.
    bw_vec is per-dimension (length N). Z-space distances are in 'bw' units.
    """
    bw_vec = np.asarray(bw_vec, dtype=float).ravel()
    def _build(X):
        X = np.asarray(X)
        mu = X.mean(axis=0)
        scale = bw_vec.copy()
        scale[scale == 0] = 1.0
        def apply(A):
            return (np.asarray(A) - mu) / scale
        return {"mode": "bw", "mu": mu, "scale": scale, "apply": apply}
    return _build

# ---------- 2) IQR per dimension ----------

def iqr_per_dim(X, lo=25, hi=75):
    X = np.asarray(X)
    qlo = np.percentile(X, lo, axis=0)
    qhi = np.percentile(X, hi, axis=0)
    I = (qhi - qlo).astype(float)
    I[I == 0] = 1.0
    return I

# ---------- 3) Neighbor selection by weight threshold ----------

def radius_from_wmin(w_min):
    """
    For Gaussian weights w = exp(-0.5 * r^2) in Z-space with scalar bw=1,
    include neighbors with w >= w_min  <=>  r <= sqrt(2 ln(1/w_min)).
    """
    w_min = float(w_min)
    if not (0 < w_min < 1):
        raise ValueError("w_min must be in (0,1).")
    return float(np.sqrt(2.0 * np.log(1.0 / w_min)))

def neighbors_above_wmin(x_cell, scaler, w_min, include_self=True):
    """
    Build KD-tree on Z = scaler.apply(x_cell), then return all neighbors
    within radius r = radius_from_wmin(w_min).
    Returns: (dlist, ilist, nbrs, meta)
    """
    Z = scaler["apply"](x_cell)
    r = radius_from_wmin(w_min)
    nbrs = NearestNeighbors(algorithm="kd_tree").fit(Z)
    dlist, ilist = nbrs.radius_neighbors(Z, radius=r, return_distance=True)
    if not include_self:
        for i in range(len(dlist)):
            keep = dlist[i] > 0.0
            dlist[i] = dlist[i][keep]
            ilist[i] = ilist[i][keep]
    meta = {"mode": "radius", "radius": r, "k": None, "include_self": include_self}
    return (dlist, ilist, nbrs, meta)

# ---------- 4) Gaussian kernel regression using precomputed neighbors ----------

def gaussian_kernel_regression_from_neighbors(Y, dlist, ilist, bw_scalar=1.0):
    """
    Nadaraya–Watson smoother with Gaussian weights on Z-space distances.
    Accepts Y:(T,) or (T,M). Returns (Te,) or (Te,M).
    In our setup we scale per-dimension by 'bw_vec', so we use bw_scalar=1.0.
    """
    Y = np.asarray(Y)
    was_1d = (Y.ndim == 1)
    if was_1d:
        Y = Y[:, None]

    T, M = Y.shape
    Te = len(dlist)
    fit = np.empty((Te, M), dtype=float)

    inv2bw2 = 0.5 / float(bw_scalar)**2
    for i, (di, ii) in enumerate(zip(dlist, ilist)):
        if len(ii) == 0:
            fit[i, :] = np.nan
            continue
        w = np.exp(-(di * di) * inv2bw2)
        s = w.sum()
        fit[i, :] = np.nan if s == 0.0 else (w[:, None] * Y[ii]).sum(axis=0) / s

    return fit[:, 0] if was_1d else fit
