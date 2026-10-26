"""Run frozen-policy sensitivities, correcting only NumPy JSON serialization.

The frozen primary study source is preserved byte-for-byte. Its coupled-output
writer encountered a numpy.bool_ at terminal_full_current_capacity. This wrapper
changes serialization only; simulation, parameters and source hashes stay fixed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

import age_mpc_study as study


def dump_numpy(path, obj):
    def scalar(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported JSON value: {type(value)}")

    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=scalar))


# Applied on import in spawned workers as well as in the parent process.
study.dump = dump_numpy


if __name__ == "__main__":
    frozen = json.loads((study.ROOT / "frozen_protocol.json").read_text())
    for name, expected in frozen["code_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {name}")
    if (study.ROOT / "coupled_metrics.json").exists():
        raise ValueError("Completed sensitivity results already exist; preserve them")
    dump_numpy(
        study.ROOT / "coupled_serialization_receipt.json",
        {
            "failure": "First coupled run failed saving numpy.bool_ after numerical replay checks passed",
            "correction": "JSON conversion of numpy.generic to Python scalar; no numerical change",
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "primary_source_unchanged": True,
        },
    )
    study.coupled_sensitivity()
