"""
The 2D linear-drift ("linear_vortex") example has an exact closed-form
theory for sigma, traffic, and the diffusion tensor (see
cynar.simul.linear.theory_linear_2N). This is the main sanity check that
cynar.traffic (correlation-based) and cynar.inflow / cynar.v2method
(grid-based) recover the right numbers on a known nonequilibrium system.
"""
import numpy as np
import pytest

from cynar import cfg, traffic, inflow, v2method
from cynar.simul import linear


@pytest.fixture(scope="module")
def linear_vortex_trajectory():
    dt = 2e-3
    alph, Phi = 0.2, 3.0
    A = np.array([[-alph, -Phi], [alph * Phi, -1.0]])
    D = np.array([[1.0, 0.0], [0.0, 1.0]])
    X = linear.simulate_linear_diffusion(200_000, A, D, dt, oversampling=10, method="euler", seed=0)
    return X, dt, A, D


def test_traffic_recovers_theory(linear_vortex_trajectory):
    X, dt, A, D = linear_vortex_trajectory
    Tr_true, sigma_true, G_true = linear.theory_linear_2N(A, D)

    out = traffic.compute_traffic(X, dt, cfg.TrafficCfg())
    assert np.diag(out["D"]) == pytest.approx(np.diag(D), abs=0.1)
    assert out["Tr"] == pytest.approx(Tr_true, rel=0.1)


def test_inflow_and_v2method_agree_on_grid(linear_vortex_trajectory):
    X, dt, A, D = linear_vortex_trajectory
    Tr_true, sigma_true, G_true = linear.theory_linear_2N(A, D)

    side = np.array([0.4, 0.4])
    icfg = cfg.InflowCfg()

    out_inflow = inflow.grid_inflow(X, dt, side, D, cfg=icfg)
    out_v2 = v2method.grid_v2(X, dt, side, D, cfg=icfg)

    # inflow.grid_inflow's bundled v2-method result must match the standalone
    # cynar.v2method.grid_v2 result exactly: same grid, same formula.
    assert out_inflow["grid"]["sigma_v2"] == pytest.approx(out_v2["grid"]["sigma_v2"])

    # both G and sigma_v2 should be in the right ballpark of the true values
    assert out_inflow["grid"]["G"] == pytest.approx(G_true, rel=0.5)
    assert out_inflow["grid"]["sigma_v2"] == pytest.approx(sigma_true, rel=0.5)
