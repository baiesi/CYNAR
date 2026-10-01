# =============================
# data_io.py
# =============================
from __future__ import annotations
from typing import Optional
import gzip
import numpy as np


def load_trajectory(path: str, has_time_column: bool, dt: Optional[float] = None):
    """
    Load a trajectory from a .csv/.txt/.npy file (optionally gzip-compressed,
    e.g. .csv.gz/.npy.gz) into the (T, N) array CYNAR expects.

    If has_time_column, column 0 is treated as time: it is stripped off,
    dt is taken as its median step, and a warning is printed if the
    sampling isn't close to uniform (the whole pipeline assumes a fixed
    dt). Otherwise dt must be supplied explicitly.
    """
    base = path[:-3] if path.endswith(".gz") else path  # strip .gz to see the real format
    if base.endswith(".npy"):
        # np.loadtxt/savetxt auto-(de)compress .gz by filename, but np.load doesn't,
        # so a gzipped .npy needs the file object opened explicitly.
        if path.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                raw = np.load(f)
        else:
            raw = np.load(path)
    else:
        raw = np.loadtxt(path, delimiter=None if base.endswith((".txt", ".dat")) else ",")
    raw = np.atleast_2d(raw)

    if has_time_column:
        t = raw[:, 0]
        X = raw[:, 1:]
        steps = np.diff(t)
        dt_inferred = float(np.median(steps))
        spread = float(np.std(steps) / dt_inferred) if dt_inferred != 0 else np.inf
        if spread > 0.01:
            print(f"Warning: time column is not uniformly spaced (relative std of "
                  f"steps = {spread:.2%}); CYNAR assumes a fixed dt, so estimates "
                  f"may be biased. Using the median step, dt={dt_inferred:.6g}.")
        dt = dt_inferred
    else:
        X = raw
        if dt is None:
            raise ValueError("has_time_column is False: dt must be supplied explicitly.")

    return X, float(dt)
