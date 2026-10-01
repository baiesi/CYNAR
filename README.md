# CYNAR

**C**omputation **Y**ields **N**onequilibrium **A**nalysis and **R**econstruction.

CYNAR estimates the steady-state entropy production rate σ of an overdamped diffusive system directly from trajectory data — no access to the underlying forces required. It implements the method of Di Terlizzi (PRL 2025, DOI: https://doi.org/10.1103/fsph-437v), which decomposes

```
σ = 4 · traffic + inflow_rate
```

- **traffic** (𝒯) is obtained from the curvature of the correlation matrix near t=0, by fitting each pairwise correlation with a sum of `exp(-s·t - r·t²) · polynomial(t)` terms and reading off analytic derivatives at t=0.
- **inflow rate** (𝒢) is obtained by binning trajectory data into a spatial grid, estimating the score ∇lnρ(x), and computing its diffusion-weighted covariance.

The package also provides an explicit implementation of a rival estimator, the **v²-method** (steady-state probability-flux estimator of Gnesotto & Broedersz, RPP 2018), in `cynar.v2method`, sharing the same spatial grid as the inflow-rate computation so the two can be compared cell-size-for-cell-size.

## Install

```bash
pip install -e .
```

## Layout

- `cynar.cfg` — dataclass configs (`TrafficCfg`, `InflowCfg`, `WindowCfg`, `ExpPolyCfg`, `SmoothingCfg`).
- `cynar.correl` — FFT correlation functions and exp-poly fitting.
- `cynar.traffic` — drives the correlation fits; computes 𝒯 and the estimated diffusion tensor D̂.
- `cynar.inflow` — grid construction, score estimation, computes 𝒢.
- `cynar.v2method` — the v²-method (probability-flux) comparison estimator, on the same grid as `inflow`.
- `cynar.smooth` — Gaussian-kernel/Nadaraya-Watson smoothing of grid fields.
- `cynar.plot` — plotting helpers.
- `cynar.simul` — example systems: `linear` (2D linear drift, closed-form theory), `SST` (stochastic switching traps, closed-form theory), `hair` (nonlinear active hair-bundle model), `Lorenz`, `Lorenz96` (high-dimensional chaotic examples).

## Using your own data

`notebooks/template_your_data.ipynb` is a generic template: point it at your own trajectory file
(a CSV/TXT/NPY array, optionally with a leading time column) and it adapts to the data's own
dimensionality, proposing candidate grid cell sizes and sweeping the traffic fit window
automatically. With no data set, it runs on a small synthetic dataset so you can see the expected
output first. See `QUICKSTART.md` for a short step-by-step version, or `docs/considerations_on_CYNAR.tex`
for the reasoning behind each setting.

## Examples

The notebooks in `notebooks/` reproduce the paper's examples, one per model: `01_linear_vortex` (with the noise-robustness study in `01b_noise_robustness`), `02_hair`, `03_lorenz`, `04_lorenz96`; `00_traffic_fit_illustration` illustrates the traffic fit. Shared helpers are in `notebooks/common.py`.

## Status

Version 0.1.0. Companion code for the paper "CYNAR: a trajectory-based estimator of entropy production" (Baiesi & Di Terlizzi, [arXiv:2609.39654](https://arxiv.org/abs/2609.39654)), comparing CYNAR against the v²-method across these example systems. The API may still change. See `CITATION.cff` for how to cite.
