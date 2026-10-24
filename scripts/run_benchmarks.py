"""Explicit opt-in synthetic capacity/risk/seed sweep. Each run freezes its manifest.

Do not interpret repeated test sweeps as an untouched final test. Use validation
for tuning and a separately locked study before reporting scientific results.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacities", nargs="+", type=float, default=[100])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--hours", type=int, default=72)
    args = parser.parse_args()
    for capacity in args.capacities:
        for alpha in args.alphas:
            for seed in args.seeds:
                output = Path(f"outputs/sweep/c{capacity}_a{alpha}_s{seed}")
                output.mkdir(parents=True, exist_ok=True)
                config = yaml.safe_load(Path("configs/system.yaml").read_text())
                config["battery"]["capacity_kwh"] = capacity
                path = output / "system.yaml"
                path.write_text(yaml.safe_dump(config))
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "microgrid.cli",
                        "demo",
                        "--config",
                        str(path),
                        "--alpha",
                        str(alpha),
                        "--seed",
                        str(seed),
                        "--output",
                        str(output),
                        "--dispatch-hours",
                        str(args.hours),
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()
