# CYNAR quickstart

The short version. For the reasoning behind each setting, see `docs/considerations_on_CYNAR.tex`.

## 1. Install

```bash
pip install -e .
```

## 2. Get your data into shape

A trajectory as a `(T, d)` array: `T` time steps, `d` coordinates, one row per recorded state, fixed sampling interval `dt`.

- File formats: `.csv`/`.txt`/`.npy`, optionally gzip-compressed (`.csv.gz`, `.npy.gz`).
- A leading time column is fine — `cynar.data_io.load_trajectory` strips it and infers `dt` from it.

## 3. Open `notebooks/template_your_data.ipynb`

Run it once as-is first (no data set → it runs on a small synthetic example, so you see the expected output before touching anything). Then set, in the config cell:

| Variable | What it does |
|---|---|
| `DATA_PATH` | Path to your file (`None` → synthetic demo) |
| `HAS_TIME_COLUMN`, `DT` | How `dt` is obtained |
| `H_LIST` | Fit-window widths to sweep for the traffic estimate |
| `MAX_DIM_FOR_INFLOW` | Dimensions above this skip the grid/inflow step (grid cost grows exponentially with `d`) |
| `N_TARGET_BINS_PER_DIM`, `R_LIST` | Candidate grid cell sizes to sweep |

Then run all cells top to bottom.

## 4. Read off the result

- The traffic cell prints `H*` and `Tr*` (the stable fit-window choice and its traffic estimate).
- If `d ≤ MAX_DIM_FOR_INFLOW`, the inflow cell prints the analogous `Δx*`/`G*`.
- The summary cell prints `sigma_CYNAR = 4*Tr* + G*` — your entropy production rate estimate.
- If `d == 2`, you also get density/score/velocity field plots.

## 5. Minimal script, without the notebook

For a one-off, programmatic estimate with default settings:

```python
from cynar import data_io, cfg, traffic, inflow

X, dt = data_io.load_trajectory("mydata.csv", has_time_column=True)

out_traffic = traffic.compute_traffic(X, dt, cfg.TrafficCfg())
Tr = out_traffic["Tr"]

side0, R_list, _ = inflow.propose_cell_sizes(X)          # only for d small enough to grid
out_inflow = inflow.grid_inflow(X, dt, side0, out_traffic["D"], cfg=cfg.InflowCfg())
G = out_inflow["grid"]["G"]

sigma = 4 * Tr + G
```

This uses `TrafficCfg`'s and `InflowCfg`'s package defaults (`H=0.2`, `side0` from `propose_cell_sizes`) — fine for a quick look, but for a number you'd actually trust, sweep `H` and the cell size as the notebook does and pick the stable point, rather than a single fixed choice.

## 6. Optional: compare against the v² method

`cynar.v2method` implements the rival probability-flux estimator on the same grid, so you can sanity-check CYNAR against it cell-size-for-cell-size — `out_inflow["grid"]["sigma_v2"]` is already computed alongside `G` above.
