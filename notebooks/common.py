"""
Shared helpers for the CYNAR paper-1 figure notebooks (notebooks/*.ipynb).

Not part of the installable `cynar` package -- these are demo/figure-generation
utilities for reproducing the comparisons between CYNAR and the
v^2-method. Adapted from the sweep/plotting logic proven out in
C6/CYNAR_demo.ipynb, generalized across models and relabeled to use the
explicit cynar.v2method naming (sigma_v2) instead of the old sigma_nu alias.
"""
from pathlib import Path
import pickle

import numpy as np
import matplotlib.pyplot as plt

from cynar.cfg import TrafficCfg, WindowCfg, ExpPolyCfg, InflowCfg, SmoothingCfg
from cynar.traffic import compute_traffic
from cynar.inflow import grid_inflow
from cynar.plot import add_panel_label
from cynar.correl import correlations_fft, auto_window, fit_exp_poly_times
from cynar._grid import make_grid_simple, binning, covariance_matrix

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

CYNAR_COLOR = "#1B9E77"
V2_COLOR = "#E95F02"
COLOR_INFLOW_S = "#77AAF0"
COLOR_INFLOW = "#4488D0"
COLOR_TRAFFIC = "#AF77AA"
COLOR_TRAFFIC_2 = "#229955"
TRUE_STYLE = dict(linestyle=":", linewidth=2, color="black")


# ---------------------------------------------------------------------------
# Generic statistics / slicing / persistence helpers
# ---------------------------------------------------------------------------

def mean_std(list_of_arrays):
    M = np.vstack(list_of_arrays)
    mean = np.nanmean(M, axis=0)
    std = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(M.shape[1])
    return mean, std


def make_slices(T, N_slice):
    L = T // N_slice
    out = []
    for k in range(N_slice):
        i0 = k * L
        i1 = (k + 1) * L if k < N_slice - 1 else T
        out.append((i0, i1))
    return out


def save_result(name, obj):
    with open(RESULTS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)


def load_result(name):
    with open(RESULTS_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Traffic-vs-H sweep: pick a stable fit window, get D_hat per slice
# ---------------------------------------------------------------------------

def sweep_traffic_vs_H(X, dt, H_list, *, N_slice=5, i1_window=2, ridge=1e-9,
                        Ne=1, Ng=1, max_iter=2000, clip_exp=60.0, max_lag=10_000):
    """
    Fit traffic at each H in H_list, on N_slice non-overlapping chunks of X.
    Returns TrH_mean/std vs H_list, and one D_hat per slice (taken at the
    middle H of H_list) for use in the downstream inflow/v2method sweep.
    Also returns D_over_H, a length-N_slice list of (len(H_list), N, N)
    arrays with the fitted D at *every* H (not just H_for_D), for figures
    that check the whole reconstructed diffusion tensor's stability vs H
    (e.g. its off-diagonal components) rather than just the traffic value.

    Defaults to the simplest exp-poly model (Ne=1, Ng=1: a single
    exp(-s t - r t^2)*(a0+a1 t) term), also cynar.cfg.ExpPolyCfg's package
    default. Checked empirically on linear_vortex: a richer model (Ne=2,
    Ng=2) is prone to multimodal nonlinear-fit behavior on this kind of
    two-timescale correlation function (a 2x2 linear system with two
    distinct eigenvalues) -- repeated fits on statistically equivalent data
    (same model, different seeds/lengths) can land in different local
    minima, giving Tr_hat anywhere from ~1.7 to ~3.3 for a true value of
    2.4. Ne=1,Ng=1 is far more stable (Tr_hat consistently within ~15% of
    truth) at a modest cost in flexibility -- prefer it unless a specific
    system's short-time correlation shape needs more terms.
    """
    T = len(X)
    slices = make_slices(T, N_slice)
    H_for_D = H_list[len(H_list) // 2]

    Tr_over_H, D_from_traffic, D_over_H = [], [], []
    for (i0, i1) in slices:
        X1 = X[i0:i1]
        Tr_list, D_list, D_for_slice = [], [], None
        for H in H_list:
            cfg = TrafficCfg(
                max_lag=max_lag, ridge=ridge,
                window=WindowCfg(i1=i1_window, H=float(H)),
                fit=ExpPolyCfg(Ne=Ne, Ng=Ng, max_iter=max_iter, clip_exp=clip_exp),
            )
            out = compute_traffic(X1, dt, cfg)
            Tr_list.append(float(out["Tr"]))
            D_list.append(np.array(out["D"], dtype=float))
            if np.isclose(H, H_for_D):
                D_for_slice = np.array(out["D"], dtype=float)
        Tr_over_H.append(np.array(Tr_list))
        D_from_traffic.append(D_for_slice)
        D_over_H.append(np.array(D_list))

    TrH_mean, TrH_std = mean_std(Tr_over_H)
    return dict(H_list=np.asarray(H_list, dtype=float), TrH_mean=TrH_mean, TrH_std=TrH_std,
                D_from_traffic=D_from_traffic, D_over_H=D_over_H, slices=slices, H_for_D=H_for_D)


def pick_stable_Tr(traffic_result):
    """Tr* at the H with smallest across-slice std (the most stable fit window)."""
    H_list = traffic_result["H_list"]
    TrH_mean, TrH_std = traffic_result["TrH_mean"], traffic_result["TrH_std"]
    idx = int(np.nanargmin(TrH_std))
    return dict(H_star=float(H_list[idx]), Tr_star_mean=float(TrH_mean[idx]), Tr_star_std=float(TrH_std[idx]))


def pick_stable_G(cellsize_result, key="Gb"):
    """
    G* at the cell size Delta x with smallest across-slice std for the given
    field (default "Gb", the raw inflow rate; also usable with "Gt" for the
    smoothed one), analogous to pick_stable_Tr for the traffic fit window.

    Checked against known ground truth on three systems: matches the actual
    lowest-bias Delta x exactly for linear_vortex and hair, but not for
    Lorenz, where the bias-vs-Delta x and variance-vs-Delta x trends run in
    the same direction (both shrink as Delta x grows) rather than crossing,
    so minimizing variance walks to the *most* biased point, not the least.
    Same failure mode as pick_stable_Tr: minimizing variance across chunks is
    not a guaranteed proxy for minimizing bias, only a useful default.
    """
    dx = cellsize_result["dx"]
    G_mean, G_std = cellsize_result[f"{key}_mean"], cellsize_result[f"{key}_std"]
    idx = int(np.nanargmin(G_std))
    return dict(dx_star=float(dx[idx]), G_star_mean=float(G_mean[idx]), G_star_std=float(G_std[idx]), idx=idx)


# ---------------------------------------------------------------------------
# Inflow / v^2-method sweep over cell size (merit B)
# ---------------------------------------------------------------------------

def sweep_cellsize(X, dt, D_from_traffic, side0, R_list, slices, *,
                    n_min=2, thr_g=200.0, Q=None, w_min=None):
    """
    For each slice and cell-size multiplier R, run grid_inflow (giving both
    the CYNAR inflow rate G and, from the *same* grid, the bundled v2-method
    comparison sigma_v2) -- this shared-grid property is what makes the
    cell-size comparison between CYNAR and the v2-method meaningful.
    """
    if Q is None:
        # Multiples of the grid's own cell size (side), not fractions of the
        # raw trajectory spread -- see cynar.cfg.InflowCfg's docstring for why.
        # Q=0.5 was tuned empirically across linear_vortex, hair, and Lorenz.
        Q = {"rho": 0.5, "nu": 0.5, "g": 0.5}
    if w_min is None:
        w_min = np.exp(-3.0)

    G_all, Gt_all, s2_all, s2t_all = [], [], [], []
    for s_idx, (i0, i1) in enumerate(slices):
        X1 = X[i0:i1]
        D_slice = D_from_traffic[s_idx]
        G_list, Gt_list, s2_list, s2t_list = [], [], [], []
        for R in R_list:
            side = R * np.asarray(side0, dtype=float)
            icfg = InflowCfg(n_min=n_min, thr_g=thr_g,
                              smoothing=SmoothingCfg(enabled=True, w_min=w_min),
                              Q=Q, PRINT=False)
            out = grid_inflow(X1, dt, side, D_slice, cfg=icfg)
            grid = out.get("grid", {})
            G_list.append(float(grid.get("G", np.nan)))
            s2_list.append(float(grid.get("sigma_v2", np.nan)))
            smooth = out.get("smooth")
            if smooth is not None:
                Gt_list.append(float(smooth.get("G_tilde", np.nan)))
                s2t_list.append(float(smooth.get("sigma_v2_tilde", np.nan)))
            else:
                Gt_list.append(np.nan)
                s2t_list.append(np.nan)
        G_all.append(np.array(G_list)); Gt_all.append(np.array(Gt_list))
        s2_all.append(np.array(s2_list)); s2t_all.append(np.array(s2t_list))

    Gb_mean, Gb_std = mean_std(G_all)
    Gt_mean, Gt_std = mean_std(Gt_all)
    s2_mean, s2_std = mean_std(s2_all)
    s2t_mean, s2t_std = mean_std(s2t_all)
    dx = np.asarray(R_list, dtype=float) * float(np.asarray(side0)[0])
    return dict(R_list=np.asarray(R_list, dtype=float), dx=dx,
                Gb_mean=Gb_mean, Gb_std=Gb_std, Gt_mean=Gt_mean, Gt_std=Gt_std,
                sigma_v2_mean=s2_mean, sigma_v2_std=s2_std,
                sigma_v2_tilde_mean=s2t_mean, sigma_v2_tilde_std=s2t_std)


def combine_sigma_cynar(Tr_star_mean, Tr_star_std, G_mean, G_std):
    """sigma_CYNAR = 4*Tr* + G (Tr* fixed at the stable H; G varies with cell size)."""
    sigma_mean = 4.0 * Tr_star_mean + G_mean
    sigma_std = np.sqrt(16.0 * Tr_star_std ** 2 + G_std ** 2)
    return sigma_mean, sigma_std


# ---------------------------------------------------------------------------
# Partial observation: "honest" estimate on a marginal subset Omega, and the
# matching bar-panel plot, shared across subsets so each one (whether or not
# it is aligned with a block of D) is run through the identical pipeline.
# ---------------------------------------------------------------------------

def honest_partial_estimate(X, D, dt, subset, H_list, side0, R_partial, *,
                             i1_window=2, N_slice=5):
    """
    The "honest" partial-observation pipeline: traffic, inflow, and the
    v^2-method all run on the marginal trajectory X[:, subset] and the
    corresponding block of D alone -- exactly as a real partial observer, with
    no access to the hidden coordinates, would. Works for any subset (single
    coordinate or several), whether or not D happens to be block-diagonal
    with respect to it and its complement.
    """
    subset = list(subset)
    D_sub = D[np.ix_(subset, subset)]
    X_sub = X[:, subset]

    traffic_sub = sweep_traffic_vs_H(X_sub, dt, H_list, N_slice=N_slice, i1_window=i1_window)
    tr_star_sub = pick_stable_Tr(traffic_sub)

    D_sub_slices = [D_sub for _ in traffic_sub["slices"]]
    cellsize_sub = sweep_cellsize(X_sub, dt, D_sub_slices, np.asarray(side0)[subset],
                                   np.array([R_partial]), traffic_sub["slices"])

    sigma_cynar_mean, sigma_cynar_std = combine_sigma_cynar(
        tr_star_sub["Tr_star_mean"], tr_star_sub["Tr_star_std"],
        cellsize_sub["Gb_mean"][0], cellsize_sub["Gb_std"][0],
    )
    return dict(
        subset=subset, D_sub=D_sub, traffic=traffic_sub, tr_star=tr_star_sub, cellsize=cellsize_sub,
        sigma_cynar_mean=sigma_cynar_mean, sigma_cynar_std=sigma_cynar_std,
        sigma_v2_mean=float(cellsize_sub["sigma_v2_mean"][0]), sigma_v2_std=float(cellsize_sub["sigma_v2_std"][0]),
    )


def plot_partial_observation_bar_panel(ax, sigma_true, sigma_ref, sigma_cynar_mean, sigma_cynar_std,
                                        sigma_v2_mean, sigma_v2_std, *, subset_label,
                                        panel_label=None, panel_label_kw=None, capsize=10):
    """
    One panel of a partial-observation bar chart: sigma_true, the reference
    sigma_Omega (`subset_label` sets its superscript), and the honest CYNAR
    and v^2-method estimates from `honest_partial_estimate`.
    """
    labels = [r"$\sigma_{\rm true}$", rf"$\sigma^{{{subset_label}}}$", "CYNAR\n", r"$v^2$-m."]
    vals = [sigma_true, sigma_ref, sigma_cynar_mean, sigma_v2_mean]
    errs = [0.0, 0.0, sigma_cynar_std, sigma_v2_std]
    colors = ["#5B66EC", "#999FEF", CYNAR_COLOR, V2_COLOR]
    ax.bar(labels, vals, yerr=errs, capsize=capsize, color=colors, alpha=0.7)
    ax.tick_params(axis="x", labelsize=11)
    ax.axhline(sigma_true, color="black", linestyle=":", lw=1.5, zorder=0)
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return ax


# ---------------------------------------------------------------------------
# Merit B figure: cell-size stability, CYNAR vs v^2-method
# ---------------------------------------------------------------------------

def plot_traffic_vs_H_panel(ax, traffic_result, tr_star, true_values=None,
                             *, panel_label=None, panel_label_kw=None, legend=True,
                             traffic_result2=None, tr_star2=None,
                             label1=r"$N_e=1$", label2=r"$N_e=2$"):
    """
    Single panel: traffic estimate vs fit-window tolerance H, with the chosen
    H* marked -- the diagnostic behind Tr* in the other two panels below.
    Fills the given ax in place and returns it, so it composes into any
    multi-panel figure (see plot_cellsize_stability for the standard 3-panel
    assembly, or notebooks/01_linear_vortex.ipynb for a custom 3x2 layout).

    Pass traffic_result2/tr_star2 (e.g. a second sweep_traffic_vs_H/pick_stable_Tr
    pair with a richer exp-poly model) to overlay a second T-vs-H series on the
    same axes -- filled markers for the primary series, open markers for the
    second, each with its own H* line -- to compare two model choices directly,
    as in notebooks/02_hair.ipynb's Ne=1-vs-Ne=2 comparison. label1/label2 name
    the two series in the legend.
    """
    true_values = true_values or {}
    H_list, TrH_mean, TrH_std = traffic_result["H_list"], traffic_result["TrH_mean"], traffic_result["TrH_std"]
    Tr_true = true_values.get("Tr_true")
    two_series = traffic_result2 is not None

    if Tr_true is not None:
        ax.axhline(Tr_true, **TRUE_STYLE, label=r"$\mathcal{T}$ (true)")
        
    curve1_label = rf"$\mathcal{{T}}$ (est., {label1})" if two_series else r"$\mathcal{T}$ (est.)"
    Hstar1_label = rf"$H^\star={tr_star['H_star']:.2f}$" + (f" ({label1})" if two_series else "")
    Hstar1_label = rf"$H^\star$" + (f" ({label1})" if two_series else "")
    ax.errorbar(H_list, TrH_mean, yerr=TrH_std, fmt="o-", capsize=3, markersize=7,
                color=COLOR_TRAFFIC, label=curve1_label)
    ax.axvline(tr_star["H_star"], linestyle=":", color=COLOR_TRAFFIC, linewidth=1.5,
               label=Hstar1_label)

    if two_series:
        H_list2 = traffic_result2["H_list"]
        TrH_mean2, TrH_std2 = traffic_result2["TrH_mean"], traffic_result2["TrH_std"]
        ax.errorbar(H_list2, TrH_mean2, yerr=TrH_std2, fmt="D-", capsize=3, markersize=6,
                    markerfacecolor="w", color=COLOR_TRAFFIC_2,
                    label=rf"$\mathcal{{T}}$ (est., {label2})")
        ax.axvline(tr_star2["H_star"], linestyle="--", color=COLOR_TRAFFIC_2, linewidth=1.5,
                   label=rf"$H^\star$ ({label2})")
                   #label=rf"$H^\star={tr_star2['H_star']:.2f}$ ({label2})")

    ax.set_xlabel(r"$H$"); ax.set_ylabel(r"$\mathcal{T}$")
    if legend:
        if two_series:
            ax.legend(fontsize=8, loc="lower right")
        else:
            ax.legend(fontsize=9)
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return ax


def plot_inflow_vs_cellsize_panel(ax, cellsize_result, true_values=None,
                                   *, panel_label=None, panel_label_kw=None, legend=True):
    """Single panel: raw and smoothed inflow rate G vs grid cell size Delta x."""
    true_values = true_values or {}
    dx = cellsize_result["dx"]
    Gb_mean, Gb_std = cellsize_result["Gb_mean"], cellsize_result["Gb_std"]
    Gt_mean, Gt_std = cellsize_result["Gt_mean"], cellsize_result["Gt_std"]
    G_true = true_values.get("G_true")

    ax.errorbar(dx, Gb_mean, yerr=Gb_std, fmt="o-", capsize=3, markersize=7,
                color=COLOR_INFLOW, label=r"$\mathcal{G}$ (grid)")
    ax.errorbar(dx, Gt_mean, yerr=Gt_std, fmt="o--", capsize=3, markersize=7,
                color=COLOR_INFLOW, label=r"$\mathcal{G}$ (smoothed)", markerfacecolor="w")
    if G_true is not None:
        ax.axhline(G_true, **TRUE_STYLE, label=r"$\mathcal{G}$ (true)",zorder=100)
    ax.set_xlabel(r"$\Delta x$"); ax.set_ylabel(r"$\mathcal{G}$")
    if legend:
        ax.legend(fontsize=9)
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return ax


def plot_sigma_vs_cellsize_panel(ax, tr_star, cellsize_result, true_values=None,
                                  *, show_smoothed=True, offset_frac=0.02,
                                  panel_label=None, panel_label_kw=None, legend=True):
    """
    Single panel: entropy-production rate vs grid cell size Delta x -- the
    direct visual test of merit B. When show_smoothed=True (the default),
    shows all four combinations: CYNAR and the v2-method, each raw and
    smoothed (same grid, same smoothing machinery, so smoothing's effect on
    each method is directly comparable). Solid/filled = raw, dashed/hollow =
    smoothed; green = CYNAR, orange = v2-method. The four series are nudged
    apart horizontally by offset_frac * (range of dx) so error bars at the
    same Delta x don't overlap; reduce offset_frac to tighten the spread.
    """
    true_values = true_values or {}
    dx = cellsize_result["dx"]
    Gb_mean, Gb_std = cellsize_result["Gb_mean"], cellsize_result["Gb_std"]
    Gt_mean, Gt_std = cellsize_result["Gt_mean"], cellsize_result["Gt_std"]
    s2_mean, s2_std = cellsize_result["sigma_v2_mean"], cellsize_result["sigma_v2_std"]
    s2t_mean, s2t_std = cellsize_result["sigma_v2_tilde_mean"], cellsize_result["sigma_v2_tilde_std"]
    sigma_g_mean, sigma_g_std = combine_sigma_cynar(tr_star["Tr_star_mean"], tr_star["Tr_star_std"], Gb_mean, Gb_std)
    sigma_gt_mean, sigma_gt_std = combine_sigma_cynar(tr_star["Tr_star_mean"], tr_star["Tr_star_std"], Gt_mean, Gt_std)
    sigma_true = true_values.get("sigma_true")

    ms, ec = 7, 3
    if sigma_true is not None:
        ax.axhline(sigma_true, **TRUE_STYLE, label=r"$\sigma$ (true)",zorder=100)
    n_series = 4 if show_smoothed else 2
    span = (dx.max() - dx.min()) if dx.size > 1 else 1.0
    offs = (np.arange(n_series) - (n_series - 1) / 2) * offset_frac * span
    ax.errorbar(dx + offs[0], sigma_g_mean, yerr=sigma_g_std, fmt="s-", capsize=ec, markersize=ms,
                color=CYNAR_COLOR, label=r"CYNAR: $4\mathcal{T}^\star+\mathcal{G}$")
    ax.errorbar(dx + offs[1], s2_mean, yerr=s2_std, fmt="^-", capsize=ec, markersize=ms,
                color=V2_COLOR, label=r"$v^2$-method")
    if show_smoothed:
        ax.errorbar(dx + offs[2], sigma_gt_mean, yerr=sigma_gt_std, fmt="s--", capsize=ec, markersize=ms,
                     markerfacecolor="w", color=CYNAR_COLOR, label=r"CYNAR, smoothed") #: $4\mathcal{T}^\star+\tilde{\mathcal{G}}$
        ax.errorbar(dx + offs[3], s2t_mean, yerr=s2t_std, fmt="^--", capsize=ec, markersize=ms,
                     markerfacecolor="w", color=V2_COLOR, label=r"$v^2$-m., smoothed")
    ax.set_xlabel(r"$\Delta x$"); ax.set_ylabel(r"$\sigma$")
    if legend:
        ax.legend(fontsize=9)
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return ax


def plot_cellsize_stability(traffic_result, tr_star, cellsize_result, true_values=None,
                             *, title=None, save_path=None, figsize=None, show_smoothed=True,
                             orient="row", legend=None,
                             panel_labels=("(a)", "(b)", "(c)"), panel_label_kw=None):
    """
    3-panel merit-B figure, assembled from plot_traffic_vs_H_panel,
    plot_inflow_vs_cellsize_panel, and plot_sigma_vs_cellsize_panel (call
    those directly to compose a different layout). panel_labels are the
    journal-style "(a)"/"(b)"/"(c)" corner tags (pass None to omit them,
    e.g. for standalone/exploratory use); panel_label_kw is forwarded to
    cynar.plot.add_panel_label for all three, e.g. panel_label_kw=dict(pad_x=0.15)
    to nudge the tags further from the corner.

    orient picks the panel layout: "row" (default) lays the three panels out
    side by side (1x3, figsize (12, 3.7) unless overridden); "col" stacks
    them vertically (3x1, figsize (4.2, 7.) unless overridden); "2+1" puts
    (a) and (b) side by side in a first row and (c) alone, at double their
    width, in a second row (figsize (8, 7.4) unless overridden).

    legend controls which of the three panels draw a legend, as a 3-tuple of
    bools for (a),(b),(c) (e.g. legend=(True, False, True)). Defaults to
    (True, True, True) for "row"/"col", but (False, False, True) for "2+1",
    since (c) repeats the same true-value/CYNAR/v2-method legend entries as
    (a)/(b) and its extra width is the natural place to keep just one.

    Always returns axs as a length-3, 1D array [ax_a, ax_b, ax_c] regardless
    of orient, so e.g. axs[0].set_xlim(...) works the same way in every case.
    """
    labels = panel_labels or (None, None, None)
    if legend is None:
        legend = (False, False, True) if orient == "2+1" else (True, True, True)
    legend_a, legend_b, legend_c = legend

    if orient == "row":
        fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=figsize or (12, 3.7), constrained_layout=True)
    elif orient == "col":
        fig, (ax_a, ax_b, ax_c) = plt.subplots(3, 1, figsize=figsize or (4.2, 7.), constrained_layout=True)
    elif orient == "2+1":
        fig, axd = plt.subplot_mosaic([["a", "b"], ["c", "c"]], figsize=figsize or (5.5, 5.1),
                                       constrained_layout=True)
        ax_a, ax_b, ax_c = axd["a"], axd["b"], axd["c"]
    else:
        raise ValueError(f"orient must be 'row', 'col', or '2+1', got {orient!r}")

    plot_traffic_vs_H_panel(ax_a, traffic_result, tr_star, true_values,
                             panel_label=labels[0], panel_label_kw=panel_label_kw, legend=legend_a)
    plot_inflow_vs_cellsize_panel(ax_b, cellsize_result, true_values,
                                   panel_label=labels[1], panel_label_kw=panel_label_kw, legend=legend_b)
    plot_sigma_vs_cellsize_panel(ax_c, tr_star, cellsize_result, true_values,
                                  show_smoothed=show_smoothed, panel_label=labels[2],
                                  panel_label_kw=panel_label_kw, legend=legend_c)

    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, np.array([ax_a, ax_b, ax_c], dtype=object)


# ---------------------------------------------------------------------------
# Merit A figure: unbiasedness near/at equilibrium, CYNAR vs v^2-method
# ---------------------------------------------------------------------------

def plot_equilibrium_bias_panel(ax, param_values, sigma_cynar_mean, sigma_cynar_std,
                                 sigma_v2_mean, sigma_v2_std, *, param_label=r"$\alpha$",
                                 equilibrium_value=0.0, sigma_true=None,
                                 panel_label=None, panel_label_kw=None, legend=True, dp=0.0):
    """
    Single panel: sigma_CYNAR and sigma_v2 vs a driving parameter. If
    equilibrium_value is given (the default, 0.0), it is marked with a
    vertical line and CYNAR is expected to scatter around zero there
    (including negative values -- it is not positivity-constrained), while
    the v2-method should stay positive; pass equilibrium_value=None to
    reuse this panel for a parameter sweep with no equilibrium point on it
    (e.g. a driving-strength sweep that never crosses detailed balance).
    """
    if dp<0:
        dp = (param_values[1]-param_values[0])*0.08
    ax.axhline(0.0, color="gray", linewidth=1)
    if equilibrium_value is not None:
        ax.axvline(equilibrium_value, color="gray", linewidth=1.5, linestyle="-")
    if sigma_true is not None:
        ax.plot(param_values, sigma_true, "k:", label=r"$\sigma$ (true)", lw=2)
    ax.errorbar(param_values+dp, sigma_cynar_mean, yerr=sigma_cynar_std, fmt="s-",capsize=3, 
                color=CYNAR_COLOR, label="CYNAR")
    ax.errorbar(param_values-dp, sigma_v2_mean, yerr=sigma_v2_std, fmt="^--",capsize=3, 
                color=V2_COLOR, label=r"$v^2$-method")
    ax.set_xlabel(param_label); ax.set_ylabel(r"$\sigma$")
    if legend:
        ax.legend()
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return ax


def plot_equilibrium_bias(param_values, sigma_cynar_mean, sigma_cynar_std,
                           sigma_v2_mean, sigma_v2_std, *, param_label=r"$\alpha$",
                           equilibrium_value=0.0, sigma_true=None,
                           title=None, save_path=None, figsize=(6, 4.5), panel_label=None,
                           panel_label_kw=None):
    """Single-axes merit-A figure, assembled from plot_equilibrium_bias_panel."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_equilibrium_bias_panel(ax, param_values, sigma_cynar_mean, sigma_cynar_std,
                                 sigma_v2_mean, sigma_v2_std, param_label=param_label,
                                 equilibrium_value=equilibrium_value, sigma_true=sigma_true,
                                 panel_label=panel_label, panel_label_kw=panel_label_kw)
    if title:
        ax.set_title(title)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# Merit C figure: high-dimensional scaling (traffic dominates inflow)
# ---------------------------------------------------------------------------

def plot_dimension_traffic_only_panel(ax, N_list, sigma_true_list, four_Tr_hat_mean, four_Tr_hat_std,
                                       *, panel_label=None, panel_label_kw=None, legend=True):
    """Single panel: CYNAR's traffic-only estimate 4*Tr_hat (no cells needed) vs the true sigma, over N."""
    ax.plot(N_list, sigma_true_list, "k:", lw=2,label=r"$\sigma$",zorder=100)
    ax.errorbar(np.asarray(N_list), four_Tr_hat_mean, yerr=four_Tr_hat_std, fmt="s-", color=COLOR_TRAFFIC,
                label=r"$4\mathcal{T}$")
    ax.set_xlabel("N"); 
    #ax.set_ylabel(r"$\sigma$")
    if legend:
        ax.legend()
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return ax


def plot_dimension_inflow_share_panel(ax, N_list, sigma_true_list, G_true_list,
                                       *, panel_label=None, panel_label_kw=None):
    """Single panel: the true inflow rate's share of sigma (the part that would need a cell grid) vs N."""
    ratio = np.asarray(G_true_list, dtype=float) / np.asarray(sigma_true_list, dtype=float)
    ax.plot(N_list, ratio, "o-", color=COLOR_INFLOW)
    ax.axhline(0, color="gray", lw=1)
    ax.set_ylabel(r"$\mathcal{G}_{\rm true}~/~\sigma_{\rm true}$")
    ax.set_xlabel("N")
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return ax


def plot_D_heatmap_panel(ax, D, *, vmin=None, vmax=None, cmap="Blues", annotate=False,
                          fmt="{:.0f}", panel_label=None, panel_label_kw=None, title=None):
    """Confusion-matrix-style heatmap of a (small) NxN diffusion matrix D."""
    im = ax.imshow(D, vmin=vmin, vmax=vmax, cmap=cmap)
    N = D.shape[0]
    ax.set_xticks(range(N)); ax.set_xticklabels(range(1, N + 1),fontsize=9)
    ax.set_yticks(range(N)); ax.set_yticklabels(range(1, N + 1),fontsize=9)
    ax.set_xlabel("$j$"); ax.set_ylabel("$i$")
    if annotate:
        for i in range(N):
            for j in range(N):
                ax.text(j, i, fmt.format(D[i, j]), ha="center", va="center",
                        fontsize=7, color="white")
    if title:
        ax.set_title(title)
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return im


def plot_partial_traffic_per_site_panel(ax, m_list, four_Tr_S_arr, per_site_true, *,
                                         panel_label=None, panel_label_kw=None, legend=True):
    """
    Per-site running average 4*Tr_Omega/m vs. the number of observed sites m -- unlike the
    cumulative share 4*Tr_Omega/sigma_true, this needs no prior knowledge of sigma_true, and
    should converge, as m grows, to the true per-site rate sigma_true/N (dashed line).
    """
    m_arr = np.asarray(m_list, dtype=float)
    avg = np.asarray(four_Tr_S_arr, dtype=float) / m_arr
    ax.plot(m_arr, avg, "o-", color=COLOR_TRAFFIC, label=r"$4\mathcal{T}^{\;\Omega}/~m$")
    ax.axhline(per_site_true, color="k", linestyle=":", lw=1.5, label=r"$\sigma_{\rm true}~/~N$")
    ax.set_xlabel("$m$")
    #ax.set_ylabel("per-site rate")
    ax.set_xlim(0,)
    if legend:
        ax.legend(fontsize=9)
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return ax


def plot_dimension_scaling(N_list, sigma_true_list, four_Tr_hat_mean, four_Tr_hat_std,
                            G_true_list=None, *, title=None, save_path=None, figsize=(11, 4.5),
                            panel_labels=("(a)", "(b)"), panel_label_kw=None):
    """
    2-panel merit-C figure, assembled from plot_dimension_traffic_only_panel
    and (if G_true_list is given) plot_dimension_inflow_share_panel -- shrinking
    with N, which is what licenses dropping the infeasible grid-based term at
    high N. Call the panel functions directly to compose a different layout.
    """
    N_list = np.asarray(N_list)
    labels = panel_labels or (None, None)
    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    plot_dimension_traffic_only_panel(axs[0], N_list, sigma_true_list, four_Tr_hat_mean, four_Tr_hat_std,
                                       panel_label=labels[0], panel_label_kw=panel_label_kw)
    if G_true_list is not None:
        plot_dimension_inflow_share_panel(axs[1], N_list, sigma_true_list, G_true_list,
                                           panel_label=labels[1], panel_label_kw=panel_label_kw)

    if title:
        fig.suptitle(title)
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, axs


# ---------------------------------------------------------------------------
# Illustrative Fig. 1: what the traffic fit actually does to C*(t)
# ---------------------------------------------------------------------------

def traffic_fit_illustration_data(X, dt, i, j, i1_window, H, *, Ne=1, Ng=1,
                                   max_iter=2000, clip_exp=60.0, max_lag=2000, ridge=1e-9):
    """
    Compute everything needed to illustrate the traffic fit on one correlation
    element C*_ij(t): the raw symmetrized curve, the [i1,i2] window picked by
    auto_window, and the exp-poly fit's derivatives cp=C'(0), cpp=C''(0), fit
    curve, exactly as done internally by cynar.traffic.estimate_derivatives
    (whose cfg.plot=True diagnostic is not reusable outside that function).
    """
    Cfull = correlations_fft(X, max_lag=max_lag, step=1, demean=True)
    Cs = 0.5 * (Cfull[i, j, :] + Cfull[j, i, :])
    L = Cs.shape[0]
    t = dt * np.arange(L)

    i1, i2 = auto_window(Cs, i1_window, H)
    t_w, C_w = t[i1:i2 + 1], Cs[i1:i2 + 1]
    (cp, cpp), fitC, t_fit = fit_exp_poly_times(
        t_w, C_w, ExpPolyCfg(Ne=Ne, Ng=Ng, max_iter=max_iter, clip_exp=clip_exp)
    )
    return dict(t=t, Cs=Cs, i1=i1, i2=i2, t_fit=t_fit, fitC=fitC, cp=float(cp), cpp=float(cpp), H=float(H))


def plot_traffic_fit_illustration(ax, data, *, t_max=None, tangent_frac=1.0,
                                  show_derivatives=True, show_excluded=True, show_beyond=True,
                                  annotate=False, annotateC=-1., 
                                  panel_label=None, panel_label_kw=None, title=None,
                                  order=1,
                                 ):
    """
    Draw one panel of the illustrative traffic-fit figure: raw C*(t) with
    points below i1 marked as excluded and points beyond i2 marked as unused,
    the H-tolerance band that actually selects the window (auto_window keeps
    extending i2 while |C(t)|/|C(t_i1)| stays within (1-H, 1+H) of C(t_i1),
    a horizontal criterion, not a time interval), the exp-poly fit over the
    window, and a tangent line / osculating parabola at t=0 visualizing
    C'(0)=-D and C''(0).

    Set show_derivatives=False (and typically show_excluded=False,
    show_beyond=False, annotate=False) for a wide-context panel where the
    window is too narrow relative to t_max for these to be legible -- pair it
    with a second, narrow-t_max call (defaults) as a "zoom" panel showing them
    clearly. The zoom panel's y-axis is autoscaled tightly to the data actually
    shown (not forced to include 0), since the window-scale differences between
    the fit/tangent/parabola are usually tiny compared to C(0).
    """
    
    COLOR_in = "#EE9922"
    COLOR_before = "#9999EE"
    COLOR_after = "#77BB90"
    MS = 6
    t, Cs = data["t"], data["Cs"]
    i1, i2 = data["i1"], data["i2"]
    cp, cpp, H = data["cp"], data["cpp"], data["H"]
    C0 = Cs[0]

    if t_max is None:
        t_max = t[min(len(t) - 1, int(round(2.2 * i2)))]
    mask = t <= t_max
    idx_max = int(np.sum(mask)) - 1

    # H-tolerance band: auto_window keeps i2 growing while |C(t)| stays within
    # (1-H,1+H) of |C(t_i1)| -- a horizontal criterion in C, not a time span.
    # Drawn as the true symmetric band around C(t_i1) itself: for a monotonic
    # decay this only ever matters on the lower edge (C(t) falls through it),
    # but for a correlation that is still rising past t_i1 -- e.g. an
    # off-diagonal element with D^{ij}=0, which starts with zero slope and can
    # grow away from C(t_i1) before eventually decaying -- the window can just
    # as well terminate on the upper edge instead, while C(t) is still rising.
    C1 = Cs[i1]
    band_lo, band_hi = sorted([C1 * (1 - H), C1 * (1 + H)])
    ax.axhspan(band_lo, band_hi, color=COLOR_in, alpha=0.15, zorder=0)#,label=r"$H$-tolerance band")
    band_span = [band_lo, band_hi]
    for tb in (t[i1], t[i2]):
        ax.axvline(tb, color=COLOR_in, lw=1, ls=":", alpha=0.5, zorder=0)
    ax.axhline(0, color="0.85", lw=1, zorder=0)

    # Every category is clipped to [0, t_max] up front (not just the marker
    # plotting, but also what feeds y_range below) -- t_max is meant to set
    # the actual displayed range for a "zoom" panel, so nothing drawn (fit
    # curve, tangent/parabola, or a few "beyond window" markers for context)
    # should be allowed to silently stretch it back out, as happened before
    # when only the raw-point markers respected t_max.
    i1_max = int(np.searchsorted(t[:i1], t_max, side="right"))
    ax.plot(t[:i1_max], Cs[:i1_max], "d", mfc="none", mec=COLOR_before, mew=1.3, ms=MS, zorder=3,
            label=r"short time")
    win_max = min(i2, idx_max)
    ax.plot(t[i1:win_max + 1], Cs[i1:win_max + 1], "o", mfc="none", mec=COLOR_in, ms=MS, zorder=3,
            label="fit window")
    ax.plot(t[i2 + 1:idx_max + 1], Cs[i2 + 1:idx_max + 1], "s", mfc="none", mec=COLOR_after,
            mew=1.2, ms=MS*0.8, zorder=3, label=r"outside tolerance")

    fit_mask = data["t_fit"] <= t_max
    ax.plot(data["t_fit"][fit_mask], data["fitC"][fit_mask], "-", color="black", lw=1.8, zorder=4,
            label="exp-poly fit")

    y_range = [Cs[mask].min(), Cs[mask].max()] + band_span
    if fit_mask.any():
        y_range += [data["fitC"][fit_mask].min(), data["fitC"][fit_mask].max()]
    if show_derivatives:
        t_loc = np.linspace(0, min(tangent_frac * t[i2], t_max), 60)
        tangent = C0 + cp * t_loc
        parabola = C0 + cp * t_loc + 0.5 * cpp * t_loc**2
        if order>=1:
            ax.plot(t_loc, tangent, "--", color="k", lw=1.5, zorder=6,) # label=r"$C_0+{C_0}'\,t$")
            y_range += [tangent.min(), tangent.max()]
        if order>=2:
            ax.plot(t_loc, parabola, "-.", color="k", lw=1.5, zorder=5,
                    label=r"osculating parabola")
            y_range += [parabola.min(), parabola.max()]

    y_lo, y_hi = min(y_range), max(y_range)
    pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.ticklabel_format(style="sci", axis="x", scilimits=(-2, 3))

    ax.set_xlim(0, t_max)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$C_*(t)$")
    if title:
        ax.set_title(title, fontsize=10)

    if annotate:
        txt = fr"$D={-cp:.3g}$" + "\n" + fr"$C''(0)={cpp:.3g}$"
        ax.text(0.97, 0.05, txt, transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.3"))
    if annotateC>0:
        # C1 itself (the reference C^ref) is marked only as a thin dotted
        # line, not a separate text label -- when H is small the three
        # y-values (band_lo, C1, band_hi) sit too close together for three
        # labels to avoid overlapping, so only the band edges actually used
        # by the tolerance criterion are labeled.
        ax.axhline(C1, color="0.4", lw=1, ls=":", zorder=0)
        ax.text(annotateC, band_hi, r"$(1+H)C^{\rm ref}$", fontsize=10, va="top", ha="center")
        ax.text(annotateC, band_lo, r"$(1-H)C^{\rm ref}$", fontsize=10, va="bottom", ha="center")

    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    return ax


# ---------------------------------------------------------------------------
# Measurement-noise robustness: corrupt an observed trajectory
# ---------------------------------------------------------------------------

def add_relative_noise(X, R, rng):
    """
    Corrupt each observed component independently with additive white Gaussian
    noise whose amplitude is R times the typical fluctuation of that component:
    y^i = x^i + std(x^i) * R * xi^i, with xi^i iid standard Gaussian drawn fresh
    per time step and per component (no temporal or cross-component
    correlation) and std taken over the whole clean trajectory X. R=0 returns X
    unchanged (up to float roundoff). `rng` is an explicit
    np.random.default_rng(seed), not a module-level global, so repeated calls
    are independently reproducible.
    """
    xi = rng.standard_normal(X.shape)
    return X + X.std(axis=0) * R * xi


# ---------------------------------------------------------------------------
# Partial observation: full-grid sigma-share of an observed subset
# ---------------------------------------------------------------------------

def full_grid_partial_sigma(X, dt, side, D_full, subset, *, n_min=2):
    """
    sigma-share carried by `subset` (a list/array of column indices of X),
    computed from the FULL system's own grid (all dimensions binned
    together, so the local mean velocity nu(x) it reads off is the genuine
    full-dimensional one) -- unlike the "honest" partial-observation
    estimate (built by feeding X[:, subset] alone through the ordinary
    compute_traffic/grid_inflow/v2method.grid_v2 pipeline, as a real partial
    observer would), this function is allowed to see every column of X.

    This is NOT a bias-free ground truth: it is the same raw
    nu-covariance/D^-1 estimator as the v2-method (same known cell-size
    discretization bias documented for merit (ii), e.g. Sec. sec:lorenz --
    at coarse cells this can overshoot the true sigma by a large factor).
    Its purpose here is only to give the honest partial estimate a
    same-discretization reference to be compared against, at the *same*
    cell size, so that comparison isolates the effect of hiding the
    unobserved coordinate(s) from the separate, already-documented,
    cell-size bias -- not to serve as an absolute-truth benchmark (use the
    system's own exact theoretical sigma_true for that, e.g. from
    observables_from_trajectory_Lorenz, keeping in mind it may itself sit
    far from this raw-grid reference at small/moderate cell sizes).

    Exact only when D_full is block-diagonal with `subset` and its
    complement in separate blocks: D_inv is then block-diagonal too, so the
    quadratic form sigma=<nu^T D^-1 nu> splits exactly into a subset share
    plus a complement share, regardless of any drift coupling between them
    (the Lorenz drift couples every component nonlinearly even when the
    noise/D does not, so this is still a genuine partial-observation
    problem, not a trivially decoupled one).
    """
    D_full = np.asarray(D_full, dtype=float)
    D_inv_full = np.linalg.pinv(D_full)
    V = (X[1:] - X[:-1]) / dt
    Xm = 0.5 * (X[1:] + X[:-1])
    edges, _ = make_grid_simple(Xm, np.asarray(side, dtype=float))
    N = X.shape[1]
    counts, nu_cell = binning(Xm, V, edges, K=N)
    valid = counts > n_min
    p = counts.astype(float); p /= (p.sum() + 1e-300)
    p_use, nu_use = p[valid], nu_cell[valid, :]
    Cov_nu = covariance_matrix(nu_use, weights=p_use)
    idx = np.ix_(subset, subset)
    return float(np.sum(D_inv_full[idx] * Cov_nu[idx]))
