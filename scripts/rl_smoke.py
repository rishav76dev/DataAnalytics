"""Small integration runs, NOT converged policies or research comparisons."""

import argparse
import json
from pathlib import Path

import pandas as pd

from microgrid.agents import evaluate, train, training_inputs
from microgrid.data import splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/demo/data/hourly.parquet")
    parser.add_argument("--reserves", default="outputs/demo/reserves.json")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frame = pd.read_parquet(args.data)
    parts = splits(len(frame))
    training = frame.iloc[parts["train"]].reset_index(drop=True)
    test = frame.iloc[parts["test"]].iloc[:72].reset_index(drop=True)
    rows = json.loads(Path(args.reserves).read_text())
    import numpy as np

    if len(rows) < len(test) or any(
        pd.Timestamp(r["timestamp"]) != ts for r, ts in zip(rows, test.timestamp)
    ):
        raise ValueError("Mismatched evaluation reserves")
    test_reserve = np.array(
        [
            np.nan if r["end_of_step_infeasible"][0] else r["end_of_step_targets_kwh"][0]
            for r in rows[: len(test)]
        ]
    )
    results = {}
    for variant in ["vanilla", "fixed", "adaptive", "constrained"]:
        selected, reserve = training_inputs(training, variant, args.seed)
        output = Path(f"outputs/agents/{variant}_{args.seed}")
        train(selected, output, variant, args.steps, args.seed, reserve)
        result = evaluate(
            output,
            test,
            str(output) + "_evaluation",
            variant,
            test_reserve if variant in {"adaptive", "constrained"} else None,
        )
        results[variant] = result
        print(
            f"{variant}: trained {args.steps} steps, held-out EENS={result['eens_kwh']:.3f}",
            flush=True,
        )
    Path("outputs/agents/smoke_summary.json").write_text(
        json.dumps(
            {
                "status": "integration_smoke_only_not_converged",
                "seed": args.seed,
                "steps": args.steps,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
