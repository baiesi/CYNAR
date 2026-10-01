from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple

# ----- Traffic -----------------------------------------------------------------

# correl - parameters
@dataclass
class WindowCfg:
    i1: int = 2    # first bin used for the fit
    H: float = 0.2 # determines the window for fit, large H -> large window

# correl - parameters
@dataclass
class ExpPolyCfg:
    # Ne=1, Ng=1 (a single exp(-st-rt^2)(a0+a1t) term) is the recommended and
    # tested default: richer models (e.g. Ne=2, Ng=2) are prone to multimodal
    # fit instability on systems with more than one relevant timescale, with
    # no accuracy benefit observed when the simpler model is well-behaved
    # (see docs/considerations_on_CYNAR.tex, Sec. 1.2).
    Ne: int = 1
    Ng: int = 1
    max_iter: int = 2000
    clip_exp: float = 60.0

# traffic - parameters
@dataclass
class TrafficCfg:
    max_lag: int = 10_000
    window: WindowCfg = field(default_factory=WindowCfg)  # uses WindowCfg(i1, H)
    fit: ExpPolyCfg = field(default_factory=ExpPolyCfg)
    ridge: float = 1e-9
    plot: bool = False
    figsize_per_axis: float = 3.0
    point_size: float = 4.0
    alpha_points: float = 0.55
    line_width: float = 1.5
    share_axes: bool = False
    bin_alpha: float = 10.0 ** (1 / 25)  # >1.0 enables exponential binning

# ----- ******* -----------------------------------------------------------------




# ----- Inflow ------------------------------------------------------------------
@dataclass
class SmoothingCfg:
    enabled: bool = True
    w_min: float = 0.049787068367863944   ## exp(-3.0), exclude other points below w_min from neighbors

# Inflow - parameters
@dataclass
class InflowCfg:
    n_min: int = 1
    # Secondary safety net only: clips any surviving score component whose
    # magnitude exceeds thr_g. The *primary* defense against grid-boundary
    # artifacts is _grid.sparse_neighbor_mask (applied unconditionally in
    # grid_inflow), which excludes cells whose finite-difference score touches
    # a sparse/empty neighbor outright -- the artifact's magnitude scales as
    # ~log(1/eps)/(2*cell_size), so it *shrinks* as cells get coarser and can
    # slip under any fixed thr_g at large enough cell size, then get amplified
    # by D in the final sum (checked: this caused a ~2-order-of-magnitude blow
    # up in the inflow rate for the hair-bundle example, whose D~O(1000), at
    # cell sizes past a specific, computable crossover). Don't rely on thr_g
    # alone to catch this; it's a backstop for other noise, not the fix.
    thr_g: float = 200.0
    smoothing: SmoothingCfg = field(default_factory=SmoothingCfg)
    # Per-field smoothing bandwidth, as a *multiple of the grid's cell size*
    # (`side`, passed to grid_inflow/grid_v2) -- NOT a fraction of the raw
    # trajectory's spread. Keys: 'rho', 'nu', 'g' (grid_v2 only uses 'rho' and
    # 'nu'). Effective smoothing radius in cells ~= 2.45 * Q (2.45 = the
    # radius, in bandwidth units, admitted by SmoothingCfg.w_min=exp(-3)).
    # Default Q=0.5 (radius ~1.2 cells) was tuned empirically across
    # linear_vortex, hair, and Lorenz (2D and 3D, D-scales from O(1) to
    # O(1000)); it is not universal -- like the traffic fit's window (H, i1),
    # the right Q is somewhat system-dependent, and a *sparser or more
    # curved* density support (e.g. a chaotic attractor's fractal geometry)
    # will need re-checking. grid_inflow/grid_v2 additionally restrict the
    # smoothing neighbor search to occupied cells only, which removes the
    # (much larger) sensitivity to how sparse/curved that support is -- Q
    # mainly still needs retuning for dimension (a fixed Q admits a ball of
    # ~Q^N cells in N dimensions, so higher-D systems generally want a
    # smaller Q for a comparable effective neighbor count).
    Q: Optional[Dict[str, float]] = None
    PRINT: bool = False

    def __post_init__(self):
        assert self.n_min >= 1, "n_min must be >= 1"
# ----- ****** ------------------------------------------------------------------
