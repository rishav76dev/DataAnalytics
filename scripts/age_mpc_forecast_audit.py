"""Frozen, non-control marginal diagnostics by future lead and physical variable.

The controller uses the separately calibrated DC-deficit envelope, not these
diagnostic intervals. Marginal corrections use January--June 2024 exclusively.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from age_mpc_study import DATA, ROOT, bounds

from microgrid.data import CHANNELS
from microgrid.forecasting import metrics


def correction():
    frame = pd.read_parquet(DATA)
    b = bounds(frame)
    q = np.load(ROOT / "development_predictions.npy", mmap_mode="r")
    origins = np.arange(b["cal"], b["validation"] - 23)
    truth = frame[CHANNELS].to_numpy()[origins[:, None] + np.arange(24)]
    prediction = q[origins - b["fit"]]
    scores = np.maximum(prediction[..., 0] - truth, truth - prediction[..., 2])
    probability = min(1, np.ceil((len(origins) + 1) * 0.8) / len(origins))
    return np.maximum(0, np.quantile(scores, probability, axis=0, method="higher"))


def evaluate(label, corrections):
    f = pd.read_parquet(DATA)
    b = bounds(f)
    start, stop = (b["validation"], b["test"]) if label == "development" else (b["test"], len(f))
    offset = b["fit"] if label == "development" else b["test"]
    q = np.load(ROOT / f"{label}_predictions.npy", mmap_mode="r")
    origins = np.arange(start, stop - 23)
    prediction = np.array(q[origins - offset])
    truth = f[CHANNELS].to_numpy()[origins[:, None] + np.arange(24)]
    corrected = prediction.copy()
    corrected[..., 0] = np.maximum(0, corrected[..., 0] - corrections)
    corrected[..., 2] += corrections
    audit = json.loads(Path("data/india/age_mpc_v2/audit.json").read_text())
    for j, key in enumerate(["pv_capacity_kw", "wind_capacity_kw"]):
        cap = audit["generation_reference_capacities"][key] * 2
        corrected[:, :, j, 2] = np.minimum(cap, corrected[:, :, j, 2])
    # An identically zero PV median and both bounds identify the physical night mask.
    corrected[:, :, 0] = np.where(
        np.all(prediction[:, :, 0] == 0, axis=-1)[..., None], 0, corrected[:, :, 0]
    )
    persistence = f[CHANNELS].to_numpy()[origins[:, None] - 168 + np.arange(24)]
    reports = {
        "origins": len(origins),
        "future_leads_only": "1..23 hours; current observed stage excluded",
    }
    for name, pred in [("raw", prediction), ("fixed_marginal_diagnostic", corrected)]:
        reports[name] = {
            "pooled": metrics(truth[:, 1:], pred[:, 1:]),
            "by_source": {
                c: metrics(truth[:, 1:, j], pred[:, 1:, j]) for j, c in enumerate(CHANNELS)
            },
            "by_lead": {
                str(h): {c: metrics(truth[:, h, j], pred[:, h, j]) for j, c in enumerate(CHANNELS)}
                for h in range(1, 24)
            },
        }
    reports["weekly_persistence_future_mae_kw"] = float(
        np.abs(truth[:, 1:] - persistence[:, 1:]).mean()
    )
    return reports


if __name__ == "__main__":
    c = correction()
    reports = {label: evaluate(label, c) for label in ["development", "test"]}
    reports["note"] = (
        "Diagnostic only; fixed correction fitted before selection; does not change any frozen action or reserve"
    )
    (ROOT / "forecast_future_diagnostics.json").write_text(json.dumps(reports, indent=2))
