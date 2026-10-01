# Demo data for `template_your_data.ipynb`

Synthetic trajectories covering the different code paths of the generic template notebook.
Point the notebook at any of them directly (or use the `EXAMPLES` dict in its Configuration
cell to fill in the settings automatically), no need for `DATA_PATH=None`'s synthetic fallback.

All are gzip-compressed (`.gz`; `load_trajectory` decompresses transparently) to keep them small:
CSVs use 5 significant digits per coordinate (`%.5g`) and fixed 6-decimal time stamps (a time
column needs its own format -- see below); `.npy` files are stored as `float32`. 100,000 points
per file, ~6.5 MB total.

| File | dim | time column | suggested settings | exercises |
|---|---|---|---|---|
| `linear2d_with_time.csv.gz` | 2 | yes, uniform | `HAS_TIME_COLUMN=True` | default path: 2D fields, inflow sweep |
| `linear2d_no_time.csv.gz` | 2 | no | `HAS_TIME_COLUMN=False`, `DT=2e-3` | manual-`dt` path |
| `linear2d_no_time.npy.gz` | 2 | no | `HAS_TIME_COLUMN=False`, `DT=2e-3` | gzipped `.npy` loading, manual `dt` |
| `linear2d_jittered_time.csv.gz` | 2 | yes, non-uniform (~17% jitter) | `HAS_TIME_COLUMN=True` | non-uniform-sampling warning in `load_trajectory` |
| `lorenz3d_with_time.csv.gz` | 3 | yes, uniform | `HAS_TIME_COLUMN=True` | inflow sweep at d=3 (within default `MAX_DIM_FOR_INFLOW=4`), no 2D field plots since d != 2 |
| `highdim6d_no_time.csv.gz` | 6 | no | `HAS_TIME_COLUMN=False`, `DT=2e-3` | dimension-threshold skip: d=6 > `MAX_DIM_FOR_INFLOW=4`, traffic-only estimate |

## Reference (ground-truth) values

Each scenario above has a companion `<scenario>.truth.json` (e.g. `linear2d_with_time.truth.json`,
shared by the `.csv.gz` and `.npy.gz` variants of the same scenario since the template notebook
strips every extension down to the base name to look it up). It holds `Tr_true`, `sigma_true`,
`G_true`:

- For the `linear2d_*` files: exact, closed-form (`cynar.simul.linear.theory_linear_general`,
  solving the stationary-covariance Lyapunov equation -- validated against the paper's existing
  2D closed form and against long simulations).
- For `lorenz3d_with_time`: a Monte Carlo reference (the same along-trajectory Sekimoto average
  used as ground truth in `03_lorenz.ipynb`, computed on this same trajectory), since Lorenz is
  nonlinear/chaotic and has no closed form; see the file's `note` field.
- For `highdim6d_no_time`: exact (same Lyapunov approach), even though the template notebook
  skips the inflow step for this file (d=6), so only `Tr_true`/`sigma_true` are directly
  comparable to what the notebook reports.

With `SHOW_TRUE_VALUES=True` (the default) in the Configuration cell, these are loaded
automatically when present and overlaid on the traffic/inflow/sigma plots.

A note on the numbers themselves: at 100,000 points these are short enough that the fitted
estimates can sit fairly far from `sigma_true` (see the reference overlay) -- that's expected and
part of the point: the plots exist so you can see how much a given system's estimate actually
fluctuates, not to claim high accuracy from a short demo trajectory.
