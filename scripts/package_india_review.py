"""Package computed traces/summaries for review, with hashes; omit model pickles."""

import hashlib
import json
import zipfile
from pathlib import Path


def main():
    out = Path("outputs/india_study")
    paths = []
    for folder in [out / "final", out / "rl"]:
        paths.extend(
            p for p in folder.iterdir() if p.suffix in {".json", ".csv", ".parquet", ".png", ".tex"}
        )
    paths.extend(
        [
            Path("data/india/reference/audit.json"),
            Path("data/india/weather/weather_manifest.json"),
            out / "final/frozen_source.zip",
            Path("docs/evidence/india_reference_2026-10-25/experimental_report.pdf"),
        ]
    )
    checksums = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out / "review_checksums.json").write_text(json.dumps(checksums, indent=2))
    with zipfile.ZipFile(out / "final/frozen_source.zip") as z:
        manifest = json.loads((out / "final/frozen_manifest.json").read_text())
        for path, expected in manifest["code_sha256"].items():
            # zipfile normalizes the leading slash of absolute __file__ paths.
            assert hashlib.sha256(z.read(path.lstrip("/"))).hexdigest() == expected, path
    paths.append(out / "review_checksums.json")
    dest = out / "india_reference_review.zip"
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, str(p))
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for path, expected in checksums.items():
            assert hashlib.sha256(z.read(path)).hexdigest() == expected, path
    print(
        f"{dest}: {len(paths)} files, {dest.stat().st_size} bytes; integrity and SHA-256 checks passed"
    )


if __name__ == "__main__":
    main()
