"""Verify committed evidence and matching source without downloading or fitting."""

import hashlib
import json
from pathlib import Path

import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    root = Path("docs/evidence/age_mpc_v2")
    hashes = json.loads((root / "checksums.json").read_text())
    for name, expected in hashes.items():
        assert sha(root / name) == expected, name
    frozen = json.loads((root / "frozen_protocol.json").read_text())
    for name, expected in frozen["code_sha256"].items():
        # The original script path in the receipt was absolute in its cloud checkout.
        path = (
            Path("scripts/age_mpc_study.py")
            if name.endswith("scripts/age_mpc_study.py")
            else Path(name)
        )
        assert sha(path) == expected, str(path)
    assert sha("configs/age_mpc/protocol.yaml") == frozen["protocol_sha256"]
    metrics = json.loads((root / "test_metrics.json").read_text())
    for name, m in metrics.items():
        t = pd.read_parquet(root / f"trace_{name}.parquet")
        assert len(t) == 4200 and int(t.normalization.sum()) == 24
        assert abs(float(t.loc[~t.normalization].unserved_kw.sum()) - m["eens_kwh"]) < 1e-8
        assert (
            abs(float(t.damage_25c_24h.sum()) - m["ageing"]["25c_24h"]["total_loss_fraction"])
            < 1e-10
        )
    print(
        f"Verified {len(hashes)} evidence files, frozen source, and {len(metrics)} primary traces"
    )
