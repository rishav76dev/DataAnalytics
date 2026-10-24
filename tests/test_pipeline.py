import numpy as np
import pytest

from microgrid.controllers import deterministic_mpc, scenario_mpc
from microgrid.data import CHANNELS, audit, splits, synthetic
from microgrid.evaluation import degradation, dispatch, paired_block_bootstrap
from microgrid.forecasting import QuantileForecaster, features, residual_blocks
from microgrid.scenarios import sample_blocks


def test_data_audit_and_chronology():
    frame = synthetic(720)
    assert audit(frame)["rows"] == 720
    s = splits(len(frame))
    assert s["train"].stop == s["validation"].start
    assert s["calibration"].stop == s["test"].start
    with pytest.raises(ValueError):
        audit(frame.drop(index=3))
    frame.loc[3, "pv_kw"] = -1
    with pytest.raises(ValueError):
        audit(frame)


def test_forecasting_is_causal_and_shapes():
    frame = synthetic(720)
    modified = frame.copy()
    modified.loc[300:, CHANNELS] = 9999
    assert np.array_equal(features(frame, 300, 24), features(modified, 300, 24))
    model = QuantileForecaster(trees=5).fit(frame, 300, stride=24)
    a = model.predict(frame, 400)
    modified.loc[400:, CHANNELS] = 9999
    assert np.array_equal(a, model.predict(modified, 400))
    assert a.shape == (24, 3, 3) and (np.diff(a, axis=-1) >= 0).all()
    r = residual_blocks(model, frame, 300, 396)
    assert r.shape == (4, 24, 3)
    with pytest.raises(ValueError):
        model.predict(frame, 299)


def test_joint_sampler_preserves_blocks():
    q = np.ones((24, 3, 3)) * 100
    residual = np.arange(10)[:, None, None] * np.ones((10, 24, 3))
    s = sample_blocks(q, residual, 100)
    assert np.array_equal(s, sample_blocks(q, residual, 100))
    assert all(np.unique(x).size == 1 for x in s)
    independent = sample_blocks(q, residual, 100, independent=True)
    assert np.unique(independent[0]).size > 1


def test_mpc_service_and_common_first_action():
    x = np.array([[0, 0, 20], [30, 0, 0], [0, 0, 20]], dtype=float)
    s = np.stack([x, x.copy()])
    s[1, 1:] *= 0.5
    action = scenario_mpc(s, 50)
    assert 0 <= action <= 20
    assert deterministic_mpc(x, 50) == pytest.approx(20)
    frame = synthetic(24)
    trace, result = dispatch(
        frame, lambda t, e, x: deterministic_mpc(frame[CHANNELS].iloc[t : t + 24].to_numpy(), e)
    )
    assert trace.soc.between(0.1 - 1e-9, 0.9 + 1e-9).all()
    assert result["eens_kwh"] >= 0


def test_rainflow_known_cycle_and_paired_bootstrap():
    d = degradation([0.1, 0.9, 0.1], hours=0)
    assert d["equivalent_full_cycles"] == pytest.approx(0.8)
    assert d["cyclic_loss_fraction"] == pytest.approx(0.2 / 4000)
    result = paired_block_bootstrap(np.arange(100), np.arange(100), repeats=50)
    assert result["ci95"] == [0, 0]
