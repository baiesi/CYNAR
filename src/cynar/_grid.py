# =============================
# _grid.py — grid/binning utilities shared by inflow.py (G) and v2method.py (sigma_v2)
# =============================
from __future__ import annotations
from typing import Optional
import numpy as np


def make_grid_simple(X: np.ndarray, side: np.ndarray):
    """
    Build an N-D grid with cell centers on integer multiples of Dx[d].
    `side` may be a scalar (same cell size in every dimension) or an array of
    length N (per-dimension cell size); if shorter than N it is expanded using
    the per-dimension standard deviation ratio, as in the original.
    """
    N = X.shape[1]

    if np.ndim(side) == 0:                 # scalar like 2 or 2.0
        side = np.full(N, side, dtype=float)
    elif len(side) < N:
        s = side[0]
        side = np.full(N, s, dtype=float)
        ratio = np.std(X, axis=0)
        ratio /= ratio[0]
        for d in range(1, N):
            side[d] = side[0] * ratio[d]

    mins, maxs = X.min(axis=0), X.max(axis=0)

    edges, center = [], []
    for d in range(N):
        dx = float(side[d])
        k_min = int(np.floor((mins[d] - 1e-10) / dx + 0.5))
        k_max = int(np.ceil((maxs[d] + 1e-10) / dx - 0.5))
        centers_d = (np.arange(k_min, k_max + 1)) * dx
        edges_d = (np.arange(k_min - 0.5, k_max + 0.5 + 1)) * dx
        center.append(centers_d)
        edges.append(edges_d)

    # centers per dim, in math order: [x0_centers, x1_centers, ..., x{N-1}_centers]
    mesh = np.meshgrid(*center, indexing="ij")                     # shapes: (n0, n1, ..., nN-1)
    centers_cell = np.stack([m.reshape(-1) for m in mesh], axis=1) # C-order flatten
    # -> flat order = C-order: i0 * prod(n1..nN-1) + i1 * prod(n2..nN-1) + ... + i{N-1}

    return edges, centers_cell


def binning(points, values, edges, K):
    sizes = [len(e) - 1 for e in edges]              # (n0, n1, ..., n_{N-1})
    # C-order strides: last axis fastest
    strides = [int(np.prod(sizes[d + 1:])) for d in range(len(sizes))]
    idx_per_dim = [
        np.clip(np.digitize(points[:, d], edges[d]) - 1, 0, sizes[d] - 1)
        for d in range(points.shape[1])
    ]
    flat_idx = np.zeros(points.shape[0], dtype=int)
    for d, idxd in enumerate(idx_per_dim):
        flat_idx += idxd * strides[d]

    n_cells = int(np.prod(sizes))
    counts = np.bincount(flat_idx, minlength=n_cells).astype(float)
    means = None
    if values is not None:
        sums = np.zeros((n_cells, K), dtype=float)
        np.add.at(sums, flat_idx, values)
        with np.errstate(invalid='ignore'):
            means = sums / (counts[:, None] + 1e-12)
            means[counts == 0] = 0.0
    return counts, means


def sparse_neighbor_mask(occupied: np.ndarray) -> np.ndarray:
    """
    For an N-D boolean grid `occupied` (True = cell has enough samples, e.g.
    counts > n_min), return a same-shape boolean mask that is True for cells
    whose score/log-density-gradient estimate is contaminated by a sparse or
    empty neighbor along any axis, plus the sparse/empty cells themselves.

    Why this is needed: log(rho) is floored at a tiny eps for empty cells, so
    a finite-difference gradient (as used for the score in inflow.py) taken
    across an occupied/empty boundary produces an artifact of magnitude
    ~log(1/eps)/(2*cell_dx) -- not a real score. This artifact *shrinks* as
    cells get coarser, so a fixed-magnitude clip (InflowCfg.thr_g) alone
    cannot catch it at every resolution: past a resolution- and
    D-tensor-dependent crossover it slips under the clip and, weighted by D,
    can dominate the whole inflow-rate estimate (checked empirically on the
    hair-bundle example, where D~O(1000): the artifact matches the clip
    threshold exactly at the cell size where G suddenly blows up by ~2 orders
    of magnitude). Excluding the affected cells directly, rather than relying
    on a magnitude threshold, fixes this at any resolution or D scale.

    True grid-boundary cells (index 0 or -1 along an axis) are populated by
    construction (make_grid_simple sizes the grid from the data's own
    min/max), so they are not flagged just for lacking an outward neighbor;
    only genuinely sparse/empty *interior* neighbors count.
    """
    occupied = np.asarray(occupied, dtype=bool)
    touches_sparse = ~occupied.copy()
    for axis in range(occupied.ndim):
        fwd = np.ones_like(occupied)
        idx_src = [slice(None)] * occupied.ndim
        idx_dst = [slice(None)] * occupied.ndim
        idx_src[axis] = slice(1, None)
        idx_dst[axis] = slice(0, -1)
        fwd[tuple(idx_dst)] = occupied[tuple(idx_src)]

        bwd = np.ones_like(occupied)
        idx_src2 = [slice(None)] * occupied.ndim
        idx_dst2 = [slice(None)] * occupied.ndim
        idx_src2[axis] = slice(0, -1)
        idx_dst2[axis] = slice(1, None)
        bwd[tuple(idx_dst2)] = occupied[tuple(idx_src2)]

        touches_sparse |= ~fwd | ~bwd
    return touches_sparse


def covariance_matrix(V: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Cov(V) across rows. If weights given, compute weighted covariance:
      Cov_w = E_w[ (v - m)(v - m)^T ], where m = E_w[v].
    """
    if weights is None:
        m = V.mean(axis=0, keepdims=True)
        return (V - m).T @ (V - m) / max(1, V.shape[0])
    w = weights.reshape(-1, 1)
    w = w / (w.sum() + 1e-300)
    m = (w * V).sum(axis=0, keepdims=True)
    Vm = V - m
    return (w * Vm).T @ Vm
