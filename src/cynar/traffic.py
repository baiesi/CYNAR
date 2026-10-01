# =============================
# traffic.py
# =============================
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

from .correl import (
    correlations_fft,
    symmetrize_C,
    auto_window,
    exponential_binning,
    fit_exp_poly_times,
)
from .cfg import (
    TrafficCfg,
    ExpPolyCfg,
    WindowCfg,
)



def _make_grid_figure(N: int, cfg: TrafficCfg):
    fig, axs = plt.subplots(
        N, N,
        figsize=(1.1 * cfg.figsize_per_axis * N, cfg.figsize_per_axis * N),
        constrained_layout=True,
        sharex=cfg.share_axes,
        sharey=cfg.share_axes,
    )
    if N == 1:
        axs = np.array([[axs]])
    return fig, axs

def estimate_derivatives(X: np.ndarray, dt: float, cfg: TrafficCfg) -> Tuple[np.ndarray, np.ndarray]:
    C = correlations_fft(X, max_lag=cfg.max_lag, step=1, demean=True)
    N, _, L = C.shape
    Cp0 = np.zeros((N, N))
    Cpp0 = np.zeros((N, N))

    if cfg.plot:
        fig, axs = _make_grid_figure(N, cfg)

    tt = dt * np.arange(L)

    # Compatibility: accept legacy WindowCfg with .Q
    H_thresh = getattr(cfg.window, "H", getattr(cfg.window, "Q", 0.5))
    i1_win   = cfg.window.i1

    for i in range(N):
        for j in range(i, N):
            Cs_ij = 0.5 * (C[i, j, :] + C[j, i, :])

            # window on the *raw* (unbinned) symmetrized series
            i1, i2 = auto_window(Cs_ij, i1_win, H_thresh)
            t_w = tt[i1:i2 + 1]
            C_w = Cs_ij[i1:i2 + 1]

            if cfg.bin_alpha > 1.0:
                # bin *within* the window; since the window starts at i1, copy-through is 0
                tb, Cb = exponential_binning(t_w, C_w, i_min=0, expansion_factor=cfg.bin_alpha)
                t_fit_in, C_fit_in = tb, Cb
            else:
                t_fit_in, C_fit_in = t_w, C_w

            (cp, cpp), fitC, t_fit = fit_exp_poly_times(t_fit_in, C_fit_in, cfg.fit)
            Cp0[i, j] = cp
            Cpp0[i, j] = cpp

            if cfg.plot:
                ax = axs[i, j]
                ax.plot(tt, Cs_ij, color='orange', lw=cfg.line_width * 3.3, alpha=cfg.alpha_points)
                ax.scatter(t_fit_in, C_fit_in, color='#EE1111', s=13, alpha=cfg.alpha_points)
                ax.plot(t_fit, fitC, color='black', lw=cfg.line_width)

                # X range
                x_min, x_max = t_fit[0], t_fit[-1]
                dx = x_max - x_min
                ax.set_xlim(x_min, x_max + 0.25 * dx)

                # Y range
                y_min = min(np.min(C_w), np.min(fitC))
                y_max = max(np.max(C_w), np.max(fitC))
                dy = y_max - y_min
                ax.set_ylim(y_min - 0.1 * dy, y_max + 0.1 * dy)

                ax.set_xlabel(r"$t$")
                ax.set_ylabel(fr"$C_{{{i+1}{j+1}}}$")

                # Boxed text with -C'(0) and -C''(0)
                txt = (
                    fr"$-(C')^{{{i+1}{j+1}}} = {-cp:.3g}$" + "\n" +
                    fr"$-(C'')^{{{i+1}{j+1}}} = {-cpp:.3g}$"
                )
                ax.text(
                    0.97, 0.97, txt, transform=ax.transAxes,
                    fontsize=9, va='top', ha='right',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
                )

    if cfg.plot:
        for i in range(N):
            for j in range(i):
                axs[i, j].set_visible(False)
        for i in range(N):
            for j in range(N):
                if axs[i, j].get_visible():
                    axs[i, j].tick_params(labelbottom=True, labelleft=True)

    # Symmetrize (keep diagonals)
    Cp0 = Cp0 + Cp0.T - np.diag(np.diag(Cp0))
    Cpp0 = Cpp0 + Cpp0.T - np.diag(np.diag(Cpp0))
    return Cp0, Cpp0

def traffic_from_Cspp(D: np.ndarray, Cpp0: np.ndarray, ridge: float = 1e-9) -> float:
    Dinv = np.linalg.inv(D + ridge * np.eye(D.shape[0]))
    return -0.25 * float(np.einsum('ij,ij->', Dinv, Cpp0))

def compute_traffic(X: np.ndarray, dt: float, cfg: TrafficCfg) -> Dict[str, object]:
    Cp0, Cpp0 = estimate_derivatives(X, dt, cfg)
    D = -Cp0
    Tr = traffic_from_Cspp(D, Cpp0, ridge=cfg.ridge)
    return {"Cp0": Cp0, "Cpp0": Cpp0, "D": D, "Tr": Tr}
