# =============================
# v2method.py — the "v^2-method": steady-state probability-flux estimator of
# entropy production, as described in Gnesotto, Mura, Gladrow & Broedersz,
# Rep. Prog. Phys. 81, 066601 (2018), section 4.2 ("Probability flux analysis").
#
# The estimator bins trajectory data into the same spatial grid used by
# inflow.grid_inflow() for the inflow rate G, and evaluates the D^-1-weighted
# covariance of the local mean drift/velocity nu(x) in each cell:
#
#     sigma_v2 = sum_ij (D^-1)^ij  Cov_p(nu)^ij
#
# where p is the (weighted) occupation probability of each cell. Sharing the
# grid with inflow.grid_inflow() is what makes the cell-size comparison
# between CYNAR (traffic + inflow) and the v^2-method meaningful: both read
# off statistics from identical cells.
# =============================
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from dataclasses import asdict

from .cfg import InflowCfg
from ._grid import make_grid_simple, binning, covariance_matrix
from .smooth import (
    make_scaler_bw,
    neighbors_above_wmin,
    gaussian_kernel_regression_from_neighbors,
)


def probability_flux_sigma(nu_use: np.ndarray, p_use: np.ndarray, D_inv: np.ndarray) -> float:
    """
    The v^2-method estimator on already-binned, already-masked cell data.

    nu_use : (M_valid, N) local mean drift/velocity per valid cell
    p_use  : (M_valid,)   occupation probability per valid cell (sums to 1)
    D_inv  : (N, N)       inverse diffusion tensor
    """
    Cov_nu = covariance_matrix(nu_use, weights=p_use)
    return float(np.sum(D_inv * Cov_nu))


def probability_flux_sigma_subsets(
    X: np.ndarray,
    dt: float,
    side: np.ndarray,
    D: np.ndarray,
    *,
    cfg: InflowCfg,
    subsets: Dict[str, "list[int]"],
) -> Dict[str, float]:
    """
    sigma_Omega = <v_Omega . D_Omega^-1 v_Omega> for several observed subsets
    Omega (each a list of column indices into X), all from a SINGLE grid pass
    over the FULL trajectory X with its exact (or fitted) full-system D.

    v_Omega here is the restriction to the observed columns of the *same*
    local-mean-velocity field nu(x) used by the full-system v^2-method above,
    and D_Omega is the corresponding Omega-Omega block of D, inverted. This is
    the rigorous quantity of Eq.~(sigmaS)/Eq.~(app:sigmaS) (Sigma_Omega =
    <v_Omega . D_Omega^-1 v_Omega>, the observed-block restriction of the full
    velocity field), and unlike a Sekimoto/Stratonovich-style restriction of
    the full known-force sum to Omega's components (as `sigma_partial_from_trajectory_Lorenz`
    in cynar.simul.Lorenz does), it does NOT require D to be block-diagonal
    with respect to Omega and its complement to be exact: it works whenever
    the noise couples an observed coordinate to a hidden one (e.g. Omega={x}
    alone, with D^{xy}!=0 coupling it to hidden y). Use this whenever any
    tested Omega is not aligned with a block-diagonal D; the two coincide only
    when Omega IS such a block.

    Returns {key: sigma_Omega for key, idx in subsets.items()}.
    """
    assert X.ndim == 2 and X.shape[0] >= 3
    T, N = X.shape
    assert D.shape == (N, N)

    V  = (X[1:] - X[:-1]) / dt
    Xm = 0.5 * (X[1:] + X[:-1])
    edges, _ = make_grid_simple(Xm, side)
    counts, nu_cell = binning(Xm, V, edges, K=N)

    cell_dx = [np.diff(e).mean() for e in edges]
    vol = np.prod(cell_dx)
    rho_cell = counts / ((T - 1) * vol)

    valid = (counts > cfg.n_min) & (rho_cell > 0)
    if not np.any(valid):
        return {key: float("nan") for key in subsets}

    p = counts.astype(float); p /= (p.sum() + 1e-300)
    p_use, nu_use = p[valid], nu_cell[valid, :]

    out: Dict[str, float] = {}
    for key, idx in subsets.items():
        idx = list(idx)
        D_sub_inv = np.linalg.inv(D[np.ix_(idx, idx)])
        out[key] = probability_flux_sigma(nu_use[:, idx], p_use, D_sub_inv)
    return out


def probability_flux_sigma_smoothed(nu_tildeS: np.ndarray, valid: np.ndarray, D_inv: np.ndarray, vol: float) -> float:
    """Smoothed counterpart of probability_flux_sigma, matching inflow.grid_inflow's G_tilde convention."""
    if not np.any(valid):
        return float("nan")
    return float(np.einsum('ij,ci,cj->', D_inv, nu_tildeS[valid], nu_tildeS[valid]) * vol)


def grid_v2(
    X: np.ndarray,
    dt: float,
    side: np.ndarray,
    D: np.ndarray,
    *,
    cfg: InflowCfg,
) -> Dict[str, Any]:
    """
    Standalone v^2-method pipeline: bins X into the same kind of grid as
    inflow.grid_inflow() (same make_grid_simple/binning routines and the same
    cfg), and returns sigma_v2 (plus its smoothed variant) without computing
    the score/inflow rate. Use this when only the v^2-method comparison is
    needed; use inflow.grid_inflow()'s 'sigma_v2'/'sigma_v2_tilde' fields when
    both G and the v^2-method are wanted from a single binning pass.
    """
    assert X.ndim == 2 and X.shape[0] >= 3
    T, N = X.shape
    assert D.shape == (N, N)
    D_inv = np.linalg.pinv(D)

    V = (X[1:] - X[:-1]) / dt
    Xm = 0.5 * (X[1:] + X[:-1])

    edges, x_cell = make_grid_simple(Xm, side)
    sizes = [len(e) - 1 for e in edges]

    counts, nu_cell = binning(Xm, V, edges, K=N)

    cell_dx = [np.diff(e).mean() for e in edges]
    vol = np.prod(cell_dx)
    rho_cell = counts / ((T - 1) * vol)

    out_params = {'side': side, 'cfg': asdict(cfg)}
    valid = (counts > cfg.n_min) & (rho_cell > 0)
    if not np.any(valid):
        if cfg.PRINT: print('No valid points')
        return {'params': out_params}

    p = counts.astype(float); p /= (p.sum() + 1e-300)
    p_use, nu_use = p[valid], nu_cell[valid, :]

    sigma_v2 = probability_flux_sigma(nu_use, p_use, D_inv)

    out_grid = {
        'edges': edges, 'x_cell': x_cell, 'vol': vol,
        'rho': rho_cell, 'nu': nu_cell, 'sigma_v2': sigma_v2,
    }
    if not cfg.smoothing.enabled:
        return {'params': out_params, 'grid': out_grid}

    # Smoothing, mirroring inflow.grid_inflow()'s treatment of nu. Bandwidth is
    # a multiple of the grid's own cell size `side`, not the raw data's IQR --
    # see the matching comment in inflow.grid_inflow() for why (in short: an
    # IQR-based bandwidth has no relation to the chosen cell size, so the
    # smoothing radius in *cells* varies wildly and uncontrollably across
    # systems/sweep points).
    I = np.asarray(side, dtype=float)
    Q = cfg.Q if cfg.Q is not None else {'rho': 0.5, 'nu': 0.5}
    Q_rho = float(Q.get('rho', 0.001))
    Q_nu = float(Q.get('nu', 0.1))

    # Restrict the smoothing neighbor search to occupied cells only -- see
    # the matching comment in inflow.grid_inflow() (in short: the full grid
    # spans the data's bounding box, which for a sparse/curved support is
    # mostly empty; searching over it dilutes the smoothed nu with phantom
    # zero-valued cells).
    occ_idx = np.flatnonzero(counts > cfg.n_min)
    x_occ = x_cell[occ_idx]
    rho_occ = rho_cell[occ_idx]
    nu_occ = nu_cell[occ_idx]

    sc_rho = make_scaler_bw(Q_rho * I)(x_occ)
    sc_nu = make_scaler_bw(Q_nu * I)(x_occ)

    d_rho, i_rho, _, _ = neighbors_above_wmin(x_occ, sc_rho, cfg.smoothing.w_min, include_self=True)
    d_nu, i_nu, _, _ = neighbors_above_wmin(x_occ, sc_nu, cfg.smoothing.w_min, include_self=True)

    rhoS_occ = gaussian_kernel_regression_from_neighbors(rho_occ, d_rho, i_rho, bw_scalar=1.0)
    nuS_occ = gaussian_kernel_regression_from_neighbors(nu_occ, d_nu, i_nu, bw_scalar=1.0)

    sqrt_rhoS_occ = np.sqrt(np.clip(rhoS_occ, 0.0, np.inf))
    nu_tilde_occ = sqrt_rhoS_occ[:, None] * nu_occ
    nu_tildeS_occ = gaussian_kernel_regression_from_neighbors(nu_tilde_occ, d_nu, i_nu, bw_scalar=1.0)

    rhoS_above_thr_occ = (rhoS_occ > cfg.n_min / ((T - 1) * vol))
    nu_tildeS_occ[~rhoS_above_thr_occ, :] = 0.0
    validS_occ = np.all(np.isfinite(nu_tildeS_occ), axis=1) & rhoS_above_thr_occ

    sigma_v2_tilde = probability_flux_sigma_smoothed(nu_tildeS_occ, validS_occ, D_inv, vol)

    # scatter back onto full-grid-shaped arrays for downstream plotting utilities
    M_full = len(rho_cell)
    nuS_cell = np.zeros((M_full, N)); nuS_cell[occ_idx] = nuS_occ
    nu_tilde = np.zeros((M_full, N)); nu_tilde[occ_idx] = nu_tilde_occ
    nu_tildeS = np.zeros((M_full, N)); nu_tildeS[occ_idx] = nu_tildeS_occ

    out_smooth = {
        'nuS': nuS_cell,
        'nu_tilde': nu_tilde,
        'nu_tildeS': nu_tildeS,
        'sigma_v2_tilde': sigma_v2_tilde,
    }
    return {'params': out_params, 'grid': out_grid, 'smooth': out_smooth}
