"""Complete a pre-test receipt without changing validation-selected policies."""

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
from age_mpc_forecast_audit import correction
from age_mpc_study import ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    if (ROOT / "test_metrics.json").exists() or (ROOT / "test/rule_based.json").exists():
        raise ValueError("Test results exist; never reseal a completed holdout")
    if (ROOT / "freeze_receipts.json").exists():
        raise ValueError("Already sealed; preserve the original receipt")
    frozen_path = ROOT / "frozen_protocol.json"
    frozen = json.loads(frozen_path.read_text())
    for name, expected in frozen["code_sha256"].items():
        assert sha(name) == expected, name
    np.save(ROOT / "marginal_interval_correction.npy", correction())
    for name in ["development_independent_calibrator.pkl", "marginal_interval_correction.npy"]:
        frozen["model_hashes"][name] = sha(ROOT / name)
    frozen["artifact_receipt_note"] = (
        "Extra pre-test artifact hashes; no policy or selection changes"
    )
    frozen_path.write_text(json.dumps(frozen, indent=2))
    archive = ROOT / "frozen_source.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        paths = [
            *map(Path, frozen["code_sha256"]),
            Path("configs/age_mpc/protocol.yaml"),
            Path("pyproject.toml"),
            Path("uv.lock"),
            Path("src/microgrid/BLAST_LICENSE.txt"),
        ]
        for p in paths:
            z.write(p, str(p.resolve().relative_to(Path.cwd())))
    (ROOT / "freeze_receipts.json").write_text(
        json.dumps(
            {
                "frozen_protocol_sha256": sha(frozen_path),
                "frozen_source_sha256": sha(archive),
                "primary_comparator": frozen["primary_comparator"],
                "fresh_test_results_exist": False,
            },
            indent=2,
        )
    )
