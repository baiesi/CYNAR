# =============================
# plot.py
# =============================
from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgb


def add_panel_label(ax, label, *, loc="upper left", pad_x=0.08, pad_y=0.05, **text_kw):
    """
    Draw a PRE-style panel tag such as "(a)" in a corner of ax, in axes
    fraction coordinates. For use when composing single-panel plot functions
    into a multi-panel figure, where journal style calls for lettered panels
    rather than per-axes titles.

    pad_x/pad_y set the horizontal/vertical inset from the chosen corner
    independently, in axes-fraction units (default: 0.10 inward from the
    left/right edge, 0.05 down/up from the top/bottom edge). The defaults
    are deliberately larger horizontally than vertically -- a label sitting
    right at the corner (e.g. pad_x=0.03) tends to land on top of the first
    data point or an error-bar cap in the upper-left corner, which a pure
    pad=0.03 (equal on both axes) does not fully avoid. Every caller that
    forwards a panel_label_kw dict (the *_panel functions in
    notebooks/common.py, and cynar.plot.plot_field_rho/plot_rho_with_quiver)
    can override these per panel, e.g. panel_label_kw=dict(pad_x=0.15).
    """
    xy = {
        "upper left": (pad_x, 1 - pad_y, "left", "top"),
        "upper right": (1 - pad_x, 1 - pad_y, "right", "top"),
        "lower left": (pad_x, pad_y, "left", "bottom"),
        "lower right": (1 - pad_x, pad_y, "right", "bottom"),
    }[loc]
    x, y, ha, va = xy
    kw = dict(fontsize=11, fontweight="bold", ha=ha, va=va)
    kw.update(text_kw)
    ax.text(x, y, label, transform=ax.transAxes, **kw)


def plot_trajectory(X,i=0,j=1,tmin=0,tmax=10000):
    from matplotlib.collections import LineCollection
    x = X[tmin:tmax, i]
    y = X[tmin:tmax, j]
    t = np.linspace(0, 1, len(x))

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap='Blues', norm=plt.Normalize(0, 1))
    lc.set_array(t)
    lc.set_linewidth(1.5)

    fig, ax = plt.subplots()

    ax.axhline(0,color='gray')
    ax.axvline(0,color='gray')
    ax.add_collection(lc)

    ax.scatter(
        x[0], y[0],
        marker="s",
        facecolor="#CCCCEE",
        edgecolors="gold",
        linewidths=1.5,
        s=100,
        zorder=100
    )

    ax.scatter(
        x[-1], y[-1],
        marker="^",
        facecolor="#222266",
        edgecolors="gold",
        linewidths=1.5,
        s=150,
        zorder=200
    )
    ax.autoscale()
    plt.show()


def plot_D_vs_fit(
    D_true: np.ndarray,
    D_fit: np.ndarray,
    *,
    cmap=None,
    figsize=(10, 5),
    joint_color_scale=True,
    title_true=r"true $D$",
    title_fit=r"fitted $D$",
    fmt="{:.2f}",
):
    """
    Plot two square panels side-by-side comparing true and fitted D matrices,
    with annotations of values inside cells.

    Parameters
    ----------
    D_true, D_fit : (N,N) arrays
        True and fitted diffusion matrices.
    cmap : str or colormap, optional
    figsize : tuple, default (10, 5)
    joint_color_scale : bool, default True
        Whether to share vmin/vmax across both panels.
    fmt : str, number format for annotations.
    """

    D_true = np.asarray(D_true)
    D_fit = np.asarray(D_fit)
    assert D_true.shape == D_fit.shape and D_true.ndim == 2 and D_true.shape[0] == D_true.shape[1], \
        "D_true and D_fit must be square matrices of the same shape."

    N = D_true.shape[0]
    if joint_color_scale:
        vmin = np.minimum(D_true.min(), D_fit.min())
        vmax = np.maximum(D_true.max(), D_fit.max())
    else:
        vmin = vmax = None

    fig, axs = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    ims = []
    titles = [title_true, title_fit]

    # Create a normalization and colormap to reuse for text color
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)

    for ax, M, title in zip(axs, [D_true, D_fit], titles):
        im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax,
                       origin="upper", interpolation="nearest", aspect="equal")
        ims.append(im)
        ax.set_title(title)
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(np.arange(1, N+1))
        ax.set_yticklabels(np.arange(1, N+1))
        ax.set_box_aspect(1)

    # --- Annotate with adaptive text color ---
    def text_color(value):
        # Convert value to RGB and compute perceived brightness
        rgb = np.array(cmap(norm(value))[:3])
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return "black" if luminance > 0.5 else "white"

    # Annotate with true / fit pairs
    for i in range(N):
        for j in range(N):
            val_true = D_true[i, j]
            val_fit = D_fit[i, j]
            if D_true[i, j]==0: s_true = "0"
            else:               s_true = fmt.format(val_true)
            s_fit = fmt.format(val_fit)
            axs[0].text(j, i, s_true, ha="center", va="center",color=text_color(val_true), fontsize=9)
            axs[1].text(j, i, s_fit,ha="center", va="center",color=text_color(val_fit), fontsize=9)

    return fig, axs

def _centers_from_edges(edges):
    return [0.5*(e[:-1] + e[1:]) for e in edges]

def flat_to_grid_plot2d(arr_flat, edges, last_dim=None):
    """C-order flat -> (ny, nx[, last_dim]) for plotting."""
    nx = len(edges[0]) - 1  # x
    ny = len(edges[1]) - 1  # y
    if last_dim is None:
        # scalar field
        return arr_flat.reshape(nx, ny).T              # (ny, nx)
    else:
        # vector/tensor field with trailing component axis
        return arr_flat.reshape(nx, ny, last_dim).transpose(1, 0, 2)  # (ny, nx, last_dim)

# --- core: plot rho (or log rho) + quiver of a vector field (nu or g) ---
def plot_rho_with_quiver(
    edges,
    rho_flat,
    vec_flat,
    *,
    ax=None,
    log=False,
    stride=1,
    quiver_scale=None,
    quiver_width=0.002,
    cmap='coolwarm',
    title=None,
    panel_label=None,
    panel_label_kw=None,
    mask_empty=True,
    vmin=None,
    vmax=None,
    RHO=True,
    ALPHA=True,
    CONTOUR=False,
    colorbar=True,
    colorbar_loc="top",
):
    if len(edges) != 2:
        raise ValueError("plot_rho_with_quiver currently supports 2D grids only.")
    if ax is None:
        _, ax = plt.subplots(1, 1, constrained_layout=True)

    rho = flat_to_grid_plot2d(rho_flat, edges)
    V   = flat_to_grid_plot2d(vec_flat, edges, last_dim=2)

    if log:
        rho_show = np.full_like(rho, np.nan, dtype=float)
        pos = rho > 0
        rho_show[pos] = np.log(rho[pos])
        m = np.min(rho_show[pos]) if np.any(pos) else 0.0
        zero = rho == 0
        rho_show[zero] = m - np.log(10)
    else:
        rho_show = rho

    x_edges, y_edges = edges
    extent = (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1])
    cx, cy = _centers_from_edges(edges)
    CX, CY = np.meshgrid(cx, cy, indexing='xy')

    # --- use fixed vmin/vmax if given ---

    if RHO:
        if CONTOUR:
            im = ax.contourf(cx, cy, rho_show, cmap=cmap, levels =10, alpha=0.75)
        else:
            im = ax.imshow(rho_show,origin='lower',extent=extent,aspect='auto',
                            cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.75,)

    U,W = V[..., 0], V[..., 1]

    if mask_empty:
        good = np.isfinite(rho) & (rho > 0) & np.isfinite(U) & np.isfinite(W)
        U = np.where(good, U, 0.0)
        W = np.where(good, W, 0.0)

    if ALPHA: alpha_grid = (rho / np.nanmax(rho)) ** 0.5
    else: alpha_grid = np.ones_like(rho)

    ax.quiver(
        CX[::stride, ::stride],
        CY[::stride, ::stride],
        U[::stride, ::stride],
        W[::stride, ::stride],
        width=quiver_width,
        scale=quiver_scale,
        scale_units='xy',
        angles='uv',
        pivot='middle',
        color='k',
        alpha=alpha_grid[::stride, ::stride],
        zorder=100,
    )

    ax.set_xlabel(r'$x^1$')
    ax.set_ylabel(r'$x^2$')
    if title:
        ax.set_title(title)
    if panel_label:
        add_panel_label(ax, panel_label, **(panel_label_kw or {}))
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     location=colorbar_loc,label=r"$\ln \rho(\mathbf{x})$",)
    return ax


# --- convenience wrappers for your outputs ---
def plot_field_rho(edges, rho, nu, *, ax=None, log=False, obs=r'$\nu$', show_title=True, **qkw):
    """
    show_title=False suppresses the auto-built "ln(rho) with ... field" title,
    for use in a composed multi-panel figure where panel_label (a "(a)"-style
    corner tag, passed through **qkw to plot_rho_with_quiver) takes its place,
    per journal (PRE) convention.
    """
    if log:
        title = r'ln($\rho$) with '+obs+' field' if show_title else None
        vmin,vmax = -10,np.log(np.max(rho))
    else:
        title = r'$\rho$ with '+obs+' field' if show_title else None
        vmin,vmax = 0, np.max(rho)
    return plot_rho_with_quiver(edges, rho, nu, ax=ax, log=log, title=title, vmin=vmin, vmax=vmax, **qkw)
