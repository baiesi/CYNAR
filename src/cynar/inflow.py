# =============================
# inflow.py
# =============================
from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, Callable
import numpy as np
from dataclasses import dataclass, field, asdict
from .cfg import InflowCfg, SmoothingCfg

from ._grid import make_grid_simple, binning, covariance_matrix, sparse_neighbor_mask
from . import v2method

from .smooth import (
    make_scaler_bw,
    neighbors_above_wmin,
    gaussian_kernel_regression_from_neighbors,
    iqr_per_dim,
)

# make_grid_simple is re-exported here (from ._grid) for backward compatibility
# with code that imported it from cynar.inflow.

# =========================================================

def propose_cell_sizes(X: np.ndarray, n_target_bins_per_dim: float = 10.0,
                        R_list=(0.5, 0.75, 1.0, 1.5, 2.0)):
    """
    Propose a base cell size side0 and a set of candidate cell sizes
    R*side0 for R in R_list, for a first cell-size sweep on unfamiliar
    data (see notebooks.common.sweep_cellsize, which takes side0/R_list in
    this exact form). side0 is set so that each dimension's IQR spans
    about n_target_bins_per_dim cells -- a starting guess only, not a
    substitute for checking the resulting stability sweep.
    """
    side0 = iqr_per_dim(X) / float(n_target_bins_per_dim)
    R_list = np.asarray(R_list, dtype=float)
    candidates = [R * side0 for R in R_list]
    return side0, R_list, candidates

# =========================================================

def grid_inflow(
    X: np.ndarray,
    dt: float,
    side: np.ndarray,
    D: np.ndarray,
    *,
    cfg: InflowCfg,                      # <-- new unified config input
) -> Dict[str, Any]:

    assert X.ndim == 2 and X.shape[0] >= 3
    T, N = X.shape
    assert D.shape == (N, N)
    D_inv = np.linalg.pinv(D)

    # ---- Midpoint data and binning ----
    V  = (X[1:] - X[:-1]) / dt
    Xm = 0.5 * (X[1:] + X[:-1])

    edges, x_cell = make_grid_simple(Xm, side)
    sizes = [len(e) - 1 for e in edges]
    M = int(np.prod(sizes))

    counts, nu_cell = binning(Xm, V, edges, K=N)  # (M,), (M,N)

    # ---- Density and score on the raw grid ----
    cell_dx = [np.diff(e).mean() for e in edges]
    vol = np.prod(cell_dx)
    rho_cell = counts / ((T - 1) * vol)

    eps = 1e-300
    rho_grid = rho_cell.reshape(*sizes)
    log_rho = np.log(np.maximum(rho_grid, eps))
    grads_axis = np.gradient(log_rho, *cell_dx, edge_order=1)
    g_grid_comp = np.stack(grads_axis, axis=-1)
    np.nan_to_num(g_grid_comp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Primary defense: a cell whose finite-difference score touches a sparse/
    # empty neighbor (density floored at eps) has an unreliable, artifact-prone
    # gradient (see _grid.sparse_neighbor_mask docstring) -- exclude it
    # outright rather than rely on magnitude alone.
    occupied_grid = (counts > cfg.n_min).reshape(*sizes)
    sparse_adjacent_grid = sparse_neighbor_mask(occupied_grid)
    g_grid_comp[sparse_adjacent_grid] = 0.0

    # Secondary safety net: clip any remaining outsized component (e.g. from
    # a low-but-above-n_min count giving a still-noisy density estimate).
    g_grid_comp = np.where(np.abs(g_grid_comp) <= cfg.thr_g, g_grid_comp, 0.0)
    g_cell = g_grid_comp.reshape(-1, N)

    # ---- Base (unsmoothed) statistics under p = counts / sum counts ----
    out_params = {'side': side, 'cfg': asdict(cfg)}
    valid = (counts > cfg.n_min) & (rho_cell > 0) & np.all(np.isfinite(g_cell), axis=1)
    if not np.any(valid):
        if cfg.PRINT: print('No valid points')
        return {'params': out_params}

    p = counts.astype(float); p /= (p.sum() + 1e-300)
    p_use, g_use, nu_use  = p[valid], g_cell[valid, :], nu_cell[valid, :]

    Cov_g  = covariance_matrix(g_use,  weights=p_use)
    G        = float(np.sum(D     * Cov_g))
    # v^2-method (Gnesotto & Broedersz RPP2018) comparison estimator, on the same cells:
    sigma_v2 = v2method.probability_flux_sigma(nu_use, p_use, D_inv)

    out_grid = {
        'edges': edges, 'x_cell': x_cell, 'vol': vol,
        'rho': rho_cell, 'nu': nu_cell, 'g': g_cell,
        'G': G, 'sigma_v2': sigma_v2,
        'sigma_nu': sigma_v2,  # backward-compatible alias
    }
    if not cfg.smoothing.enabled:
        return {'params': out_params, 'grid': out_grid}

    # =========================
    #         Smoothing
    # =========================

    # 1) smoothing bandwidth, expressed as a multiple of the grid's own cell
    #    size (side), and Q defaults.
    #
    #    Previously this used a fraction of the raw trajectory's IQR
    #    (iqr_per_dim(Xm)), which has no relationship to the chosen cell size
    #    side: since side is picked per system/sweep-point independently of
    #    the data's raw spread, the ratio (Q*IQR)/side -- the smoothing
    #    radius in *cells* -- varied wildly across examples (checked: ~0.7-2
    #    cells for linear_vortex, but ~17-24 cells for the hair-bundle model,
    #    whose IQR is numerically much larger in its physical units). At that
    #    width the Gaussian smoothing averages the score field, which is
    #    roughly antisymmetric around the density mode, down to nearly zero
    #    -- not a smoothed estimate of G, an over-smoothed one. Tying the
    #    bandwidth to `side` instead keeps the smoothing radius (in cells)
    #    consistent across systems for a given Q, regardless of the data's
    #    raw physical scale.
    I = np.asarray(side, dtype=float)  # (N,)
    Q = cfg.Q if cfg.Q is not None else {'rho': 0.5, 'nu': 0.5, 'g': 0.5}
    Q_rho = float(Q['rho'])
    Q_nu  = float(Q['nu'])
    Q_g   = float(Q['g'])

    # 2) Restrict the smoothing neighbor search to *occupied* cells only.
    #
    #    x_cell spans the full rectangular bounding box of the data (the
    #    Cartesian product of per-axis grid edges), but for a distribution
    #    concentrated on a thin or curved region -- e.g. a chaotic
    #    attractor's fractal support -- most of that box is empty. A
    #    Euclidean-radius neighbor search over the *full* grid pulls in many
    #    such empty cells, and since their g/nu were never measured (0 by
    #    construction), averaging them in as literal data dilutes the
    #    smoothed estimate toward zero -- not a smoothing effect, a sampling
    #    artifact. Checked empirically: the Lorenz attractor's grid was only
    #    ~10% occupied (vs ~48% for the hair-bundle example) at a comparable
    #    per-axis bandwidth, and its neighbor counts came out ~3x larger for
    #    the same nominal smoothing radius -- almost entirely extra empty
    #    cells, not real structure. Restricting the search (and the smoothed
    #    quantities themselves) to occupied cells removes this dependence on
    #    how sparse or curved the support is.
    occ_idx = np.flatnonzero(counts > cfg.n_min)
    x_occ   = x_cell[occ_idx]
    rho_occ = rho_cell[occ_idx]
    nu_occ  = nu_cell[occ_idx]
    g_occ   = g_cell[occ_idx]

    # 3) scalers and neighbors per field (weight-threshold), over occupied cells
    sc_rho = make_scaler_bw(Q_rho * I)(x_occ)
    sc_nu  = make_scaler_bw(Q_nu  * I)(x_occ)
    sc_g   = make_scaler_bw(Q_g   * I)(x_occ)

    d_rho, i_rho, _, meta_rho = neighbors_above_wmin(x_occ, sc_rho, cfg.smoothing.w_min, include_self=True)
    d_nu,  i_nu,  _, meta_nu  = neighbors_above_wmin(x_occ, sc_nu,  cfg.smoothing.w_min, include_self=True)
    d_g,   i_g,   _, meta_g   = neighbors_above_wmin(x_occ, sc_g,   cfg.smoothing.w_min, include_self=True)

    # 4) smooth rho, nu, g (over occupied cells)
    rhoS_occ = gaussian_kernel_regression_from_neighbors(rho_occ, d_rho, i_rho, bw_scalar=1.0)
    nuS_occ  = gaussian_kernel_regression_from_neighbors(nu_occ,  d_nu,  i_nu,  bw_scalar=1.0)
    gS_occ   = gaussian_kernel_regression_from_neighbors(g_occ,   d_g,   i_g,   bw_scalar=1.0)

    # 5) build tilde fields using smoothed density
    sqrt_rhoS_occ = np.sqrt(np.clip(rhoS_occ, 0.0, np.inf))
    g_tilde_occ   = sqrt_rhoS_occ[:, None] * gS_occ
    nu_tilde_occ  = sqrt_rhoS_occ[:, None] * nu_occ

    # 6) smooth the tilde fields
    g_tildeS_occ  = gaussian_kernel_regression_from_neighbors(g_tilde_occ,  d_g,  i_g,  bw_scalar=1.0)
    nu_tildeS_occ = gaussian_kernel_regression_from_neighbors(nu_tilde_occ, d_nu, i_nu, bw_scalar=1.0)

    # 7) mask by smoothed density threshold
    rhoS_above_thr_occ = (rhoS_occ > cfg.n_min / ((T - 1) * vol))
    g_tildeS_occ[~rhoS_above_thr_occ, :]  = 0.0
    nu_tildeS_occ[~rhoS_above_thr_occ, :] = 0.0

    validS_occ = (
        np.all(np.isfinite(g_tildeS_occ), axis=1)
        & np.all(np.isfinite(nu_tildeS_occ), axis=1)
        & rhoS_above_thr_occ
    )

    # 8) integrals for G_tilde and sigma_v2_tilde over valid occupied cells
    G_tilde = (
        float(np.einsum('ij,ci,cj->', D, g_tildeS_occ[validS_occ], g_tildeS_occ[validS_occ]) * vol)
        if np.any(validS_occ) else np.nan
    )
    sigma_v2_tilde = v2method.probability_flux_sigma_smoothed(nu_tildeS_occ, validS_occ, D_inv, vol)

    # 9) scatter back onto full-grid-shaped arrays (M,[N]) for downstream
    #    plotting utilities (cynar.plot reshapes these against `edges`);
    #    unoccupied cells get 0, consistent with how they entered the
    #    G_tilde/sigma_v2_tilde sums (excluded, i.e. contributing nothing).
    M_full = len(rho_cell)
    rhoS_cell = np.zeros(M_full); rhoS_cell[occ_idx] = rhoS_occ
    nuS_cell  = np.zeros((M_full, N)); nuS_cell[occ_idx]  = nuS_occ
    gS_cell   = np.zeros((M_full, N)); gS_cell[occ_idx]   = gS_occ
    g_tilde   = np.zeros((M_full, N)); g_tilde[occ_idx]   = g_tilde_occ
    nu_tilde  = np.zeros((M_full, N)); nu_tilde[occ_idx]  = nu_tilde_occ
    g_tildeS  = np.zeros((M_full, N)); g_tildeS[occ_idx]  = g_tildeS_occ
    nu_tildeS = np.zeros((M_full, N)); nu_tildeS[occ_idx] = nu_tildeS_occ

    out_smooth = {
        'params': {'Q': {'rho': Q_rho, 'nu': Q_nu, 'g': Q_g}, 'w_min': float(cfg.smoothing.w_min)},
        # primary smoothed fields
        'rhoS': rhoS_cell,            # (M,)
        'nuS':  nuS_cell,             # (M,N)
        'gS':   gS_cell,              # (M,N)
        # tilde variables
        'g_tilde':   g_tilde,         # (M,N)
        'nu_tilde':  nu_tilde,        # (M,N)
        'g_tildeS':  g_tildeS,        # (M,N)
        'nu_tildeS': nu_tildeS,       # (M,N)
        'G_tilde':   G_tilde,
        'sigma_v2_tilde': sigma_v2_tilde,
        'sigma_tilde': sigma_v2_tilde,  # backward-compatible alias
        'info': {
            'rho': {'bw_vec': (Q_rho * I).tolist(), 'neighbor_params': meta_rho, 'counts': [len(ix) for ix in i_rho]},
            'nu':  {'bw_vec': (Q_nu  * I).tolist(), 'neighbor_params': meta_nu,  'counts': [len(ix) for ix in i_nu]},
            'g':   {'bw_vec': (Q_g   * I).tolist(), 'neighbor_params': meta_g,   'counts': [len(ix) for ix in i_g]},
        },
    }

    return {'params': out_params, 'grid': out_grid, 'smooth': out_smooth}
