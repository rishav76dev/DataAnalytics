"""Staged degradation-aware MPC study. No SAC. Test refuses an unfrozen protocol."""

import argparse
import hashlib
import json
import multiprocessing
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from microgrid.age_mpc import age_mpc
from microgrid.ageing import LfpState, evaluate_lfp, update_lfp
from microgrid.data import CHANNELS
from microgrid.evaluation import paired_block_bootstrap, reliability
from microgrid.phase_forecasts import ResearchForecaster, daylight
from microgrid.physics import Battery, BatteryConfig
from microgrid.trajectory_calibration import JointScenarioPool, OnlineDeficitEnvelope

ROOT = Path("outputs/age_mpc_v2")
DATA = Path("data/india/age_mpc_v2/hourly.parquet")


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def bounds(frame):
    local = frame.timestamp.dt.tz_convert("Asia/Kolkata")
    return {
        "fit": int(np.flatnonzero(local.dt.year.to_numpy() == 2023)[0]),
        "cal": int(np.flatnonzero(local.dt.year.to_numpy() == 2024)[0]),
        "validation": int(
            np.flatnonzero((local.dt.year.to_numpy() == 2024) & (local.dt.month.to_numpy() == 7))[0]
        ),
        "test": int(np.flatnonzero(local.dt.year.to_numpy() == 2025)[0]),
        "end": len(frame),
    }


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "development_predictions.npy").exists():
        raise ValueError("Prepared development cache already exists")
    f = pd.read_parquet(DATA)
    b = bounds(f)
    data_audit = json.loads(Path("data/india/age_mpc_v2/audit.json").read_text())
    caps = [
        data_audit["generation_reference_capacities"]["pv_capacity_kw"] * 2,
        data_audit["generation_reference_capacities"]["wind_capacity_kw"] * 2,
    ]
    model = ResearchForecaster(generation_caps=caps).fit(f, b["fit"])
    q = model.predict_many(f, np.arange(b["fit"], b["test"]))
    (ROOT / "forecaster.pkl").write_bytes(pickle.dumps(model))
    np.save(ROOT / "development_predictions.npy", q.astype(np.float32))
    pool_origins = np.arange(b["fit"], b["cal"] - 24 + 1)
    truth = f[CHANNELS].to_numpy()[pool_origins[:, None] + np.arange(24)]
    pool = JointScenarioPool(
        truth,
        q[: len(pool_origins)],
        f.timestamp.iloc[pool_origins],
        model.scale_floor,
        generation_caps=caps,
    )
    (ROOT / "pool.pkl").write_bytes(pickle.dumps(pool))
    dump(
        ROOT / "development_data_manifest.json",
        {
            "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
            "partitions": b,
            "forecast_fit_stop": b["fit"],
            "pool_origins": len(pool_origins),
            "pool_phase_counts": {str(h): int((pool.hour == h).sum()) for h in range(24)},
            "night_pv_development_max_kw": float(
                f.pv_kw.iloc[: b["test"]].to_numpy()[~daylight(f.timestamp.iloc[: b["test"]])].max()
            ),
        },
    )
    generate_inputs(f, q, b["fit"], b["cal"], b["test"], "development")


def generate_inputs(f, q, q_offset, start, stop, label, calibrator=None, independent=False):
    pool = pickle.loads((ROOT / "pool.pkl").read_bytes())
    cal = calibrator or OnlineDeficitEnvelope()
    count = stop - start
    scenarios = np.lib.format.open_memmap(
        ROOT / f"{label}_scenarios.npy", mode="w+", dtype="float32", shape=(count, 100, 24, 3)
    )
    targets = np.lib.format.open_memmap(
        ROOT / f"{label}_usable_targets.npy", mode="w+", dtype="float64", shape=(count, 24)
    )
    maximum = np.lib.format.open_memmap(
        ROOT / f"{label}_max_interval.npy", mode="w+", dtype="float64", shape=(count, 24)
    )
    raw_targets = np.lib.format.open_memmap(
        ROOT / f"{label}_raw_targets.npy", mode="w+", dtype="float64", shape=(count, 24)
    )
    power = np.lib.format.open_memmap(
        ROOT / f"{label}_power_probability.npy", mode="w+", dtype="float64", shape=(count, 24)
    )
    records = []
    pipeline_seconds = np.zeros(count)
    for i, t in enumerate(range(start, stop)):
        pipeline_start = time.perf_counter()
        cal.advance(t, f)
        s = pool.sample(q[t - q_offset], f.timestamp.iloc[t], seed=42 + t, independent=independent)
        s[:, 0] = f[CHANNELS].iloc[t].to_numpy()
        _, d = cal.predict(t, f.timestamp.iloc[t], s, 400, 0, 320)
        pipeline_seconds[i] = time.perf_counter() - pipeline_start
        scenarios[i] = s
        targets[i] = d["usable_targets_dc_kwh"]
        maximum[i] = d["max_interval_deficit_dc_kwh"]
        raw = np.empty(24)
        for h in range(23):
            raw[h] = max(0, float(np.max(d["raw_bounds"][d["interval_i"] == h])))
        raw[-1] = 0
        raw_targets[i] = raw
        net = s[:, 1:, 2] - s[:, 1:, 0] - s[:, 1:, 1]
        power[i] = np.r_[
            np.mean(np.maximum.accumulate(net[:, ::-1], axis=1)[:, ::-1] > 50 + 1e-9, axis=0), 0
        ]
        if t + 24 <= stop:
            coverage = cal.coverage(f[CHANNELS].iloc[t : t + 24].to_numpy(), d)
            records.append(
                {
                    "origin": t,
                    "timestamp": str(f.timestamp.iloc[t]),
                    "phase": pd.Timestamp(f.timestamp.iloc[t]).tz_convert("Asia/Kolkata").hour,
                    "ready": d["ready"],
                    "correction": d["correction_dc_kwh_per_sqrt_hour"],
                    **coverage,
                }
            )
        if (i + 1) % 720 == 0:
            print(f"{label}: calibrated {i + 1}/{count} hours", flush=True)
    for arr in [scenarios, targets, maximum, raw_targets, power]:
        arr.flush()
    np.save(ROOT / f"{label}_pipeline_seconds.npy", pipeline_seconds)
    (ROOT / f"{label}_calibrator.pkl").write_bytes(pickle.dumps(cal))
    dump(ROOT / f"{label}_calibration.json", records)
    dump(ROOT / f"{label}_range.json", {"start": start, "stop": stop, "q_offset": q_offset})


def summarize(trace, c, normalization=24):
    op = trace.iloc[:-normalization]
    result = reliability(op)
    result["operational_hours"] = len(op)
    result["normalization_hours"] = normalization
    result["normalization_eens_kwh"] = float(trace.unserved_kw.iloc[-normalization:].sum())
    result["terminal_energy_kwh"] = float(trace.energy_kwh.iloc[-1])
    result["terminal_full"] = abs(result["terminal_energy_kwh"] - c.maximum) <= 1e-5
    result["initial_energy_kwh"] = c.initial_soc * c.capacity
    result["controller_seconds"] = float(trace.controller_seconds.sum())
    result["controller_p95_ms"] = float(np.quantile(trace.controller_seconds, 0.95) * 1000)
    result["surplus_mean_soc"] = float(
        trace.loc[trace.pv_kw + trace.wind_kw > trace.load_kw, "soc"].mean()
    )
    all_losses = {}
    for temperature in [25, 35]:
        for window in [12, 24, 48]:
            age = evaluate_lfp(
                [c.initial_soc, *trace.soc],
                temperature_c=temperature,
                chunk_hours=window,
                integration="effective_age_exact",
            )
            increments = np.zeros(len(trace))
            position = 0
            for item in age.pop("windows"):
                step = window
                stop = min(position + step, len(trace))
                if len(trace) - stop == 1:
                    stop += 1
                increments[stop - 1] = item["calendar_increment"] + item["cycling_increment"]
                position = stop
            trace[f"damage_{temperature}c_{window}h"] = increments
            all_losses[f"{temperature}c_{window}h"] = age
    result["ageing"] = all_losses
    return result


def simulate(candidate, label, capacity, coupled=False, temperature=25.0):
    f = pd.read_parquet(DATA)
    b = bounds(f)
    span = json.loads((ROOT / f"{label}_range.json").read_text())
    start = max(b["validation"], span["start"]) if label == "development" else span["start"]
    stop = span["stop"]
    offset = start - span["start"]
    c = BatteryConfig(capacity_kwh=capacity, power_kw=50)
    initial_config = c
    battery = Battery(c)
    ageing_state = LfpState()
    soc_window = [battery.soc]
    input_label = label + "_independent" if candidate["kind"] == "adaptive_independent" else label
    s_cache = np.load(ROOT / f"{input_label}_scenarios.npy", mmap_mode="r")
    needed = np.load(ROOT / f"{input_label}_usable_targets.npy", mmap_mode="r")
    maximum = np.load(ROOT / f"{input_label}_max_interval.npy", mmap_mode="r")
    raw = np.load(ROOT / f"{input_label}_raw_targets.npy", mmap_mode="r")
    power = np.load(ROOT / f"{input_label}_power_probability.npy", mmap_mode="r")
    q = np.load(
        ROOT
        / ("development_predictions.npy" if label == "development" else "test_predictions.npy"),
        mmap_mode="r",
    )
    q_offset = b["fit"] if label == "development" else b["test"]
    rows = []
    for i, t in enumerate(range(start, stop)):
        c = battery.config
        x = f[CHANNELS].iloc[t].to_numpy()
        k = offset + i
        point = np.array(q[t - q_offset, :, :, 1], float)
        if candidate.get("forecast") == "persistence":
            point = f[CHANNELS].iloc[t - 168 : t - 144].to_numpy().copy()
            point[:, 0] *= daylight(pd.date_range(f.timestamp.iloc[t], periods=24, freq="h"))
        point[0] = x
        current_target = c.minimum
        infeasible = False
        begin = time.perf_counter()
        if i >= stop - start - 24:
            action = -max(0, float(x[0] + x[1] - x[2]))
        elif candidate["kind"] == "rule":
            action = float(x[2] - x[0] - x[1])
        else:
            use_scenarios = candidate["kind"].startswith("scenario")
            scenarios = np.array(s_cache[k], float) if use_scenarios else point[None]
            scenarios[:, 0] = x
            reserve = None
            if candidate["kind"] == "fixed":
                reserve = np.full(24, candidate["fraction"] * c.capacity)
            if candidate["kind"] in {
                "adaptive",
                "adaptive_no_calendar",
                "adaptive_raw",
                "adaptive_independent",
                "scenario_envelope",
            }:
                risk_max = (
                    np.maximum.accumulate(raw[k, ::-1])[::-1]
                    if candidate["kind"] == "adaptive_raw"
                    else maximum[k]
                )
                flags = (risk_max > c.maximum - c.minimum + 1e-8) | (power[k] > 0.05)
                target = raw[k] if candidate["kind"] == "adaptive_raw" else needed[k]
                target = target + candidate.get("buffer_fraction", 0.0) * (c.maximum - c.minimum)
                reserve = c.minimum + target
                reserve = np.where(flags, c.maximum, np.minimum(reserve, c.maximum))
                reserve[-1] = c.minimum
                current_target = float(reserve[0])
                infeasible = bool(flags[0])
            action = age_mpc(
                scenarios,
                battery.energy,
                c,
                reserve=reserve,
                eens_slack_kwh=candidate.get("slack", 0),
                calendar=candidate["kind"] != "adaptive_no_calendar",
            )
        row = battery.step(action, *x)
        fade_energy = 0.0
        if coupled:
            soc_window.append(battery.soc)
            if len(soc_window) == 25 or i == stop - start - 1:
                history = np.array(soc_window)
                times = np.arange(len(history), dtype=float)
                if len(history) == 2:
                    history = np.array([history[0], history.mean(), history[1]])
                    times = np.array([0.0, 0.5, 1.0])
                update_lfp(
                    ageing_state,
                    times,
                    history,
                    np.full(len(history), temperature),
                    integration="effective_age_exact",
                )
                new_config = replace(initial_config, soh=ageing_state.soh)
                new_energy = battery.energy * new_config.capacity / battery.config.capacity
                fade_energy = battery.energy - new_energy
                battery.config = new_config
                battery.energy = new_energy
                soc_window = [battery.soc]
                row["energy_kwh"] = battery.energy
        row["capacity_fade_energy_loss_kwh"] = fade_energy
        row["physical_soh"] = battery.config.soh
        row.update(
            {
                "timestamp": f.timestamp.iloc[t],
                "controller_seconds": time.perf_counter() - begin,
                "requested_reserve_kwh": current_target,
                "reserve_infeasible": infeasible,
                "normalization": i >= stop - start - 24,
            }
        )
        rows.append(row)
    trace = pd.DataFrame(rows)
    result = summarize(trace, initial_config)
    result.update(
        {
            "candidate": candidate,
            "capacity_kwh": capacity,
            "infeasible_fraction": float(trace.reserve_infeasible.mean()),
        }
    )
    result["capacity_feedback"] = coupled
    if coupled:
        result.update(
            {
                "physical_temperature_c": temperature,
                "live_final_soh": ageing_state.soh,
                "live_calendar_loss_fraction": ageing_state.calendar_loss,
                "live_cycling_loss_fraction": ageing_state.cycling_loss,
                "terminal_full_current_capacity": abs(
                    trace.energy_kwh.iloc[-1] - battery.config.maximum
                )
                <= 1e-5,
                "capacity_fade_energy_loss_kwh": float(trace.capacity_fade_energy_loss_kwh.sum()),
            }
        )
        replay = result["ageing"][f"{int(temperature)}c_24h"]["total_loss_fraction"]
        if abs(replay - (1 - ageing_state.soh)) > 1e-8:
            raise ValueError("Live/offline ageing replay disagreement")
    if label == "test":
        forecast_runtime = json.loads((ROOT / "test_forecast_runtime.json").read_text())
        forecast_cost = (
            forecast_runtime["seconds"] * (stop - start - 24) / forecast_runtime["origins"]
        )
        prep = np.load(ROOT / f"{input_label}_pipeline_seconds.npy")
        if candidate["kind"] == "rule":
            shared = 0.0
        elif candidate["kind"] in {"point", "fixed"}:
            shared = forecast_cost
        else:
            shared = forecast_cost + float(prep[:-24].sum())
        result["shared_pipeline_seconds"] = shared
        result["amortized_end_to_end_seconds"] = shared + result["controller_seconds"]
        result["runtime_scope"] = (
            "Batch forecasting plus measured joint sampling/envelope updates and controller; common envelope overhead conservatively charged to all scenario variants. Not live deployed latency."
        )
    dest = ROOT / (f"coupled_{int(temperature)}c" if coupled else label)
    dest.mkdir(exist_ok=True)
    trace.to_parquet(dest / f"trace_{candidate['name']}.parquet", index=False)
    dump(dest / f"{candidate['name']}.json", result)
    return result


def calibration_summary(label, start=0):
    records = [
        r
        for r in json.loads((ROOT / f"{label}_calibration.json").read_text())
        if r["ready"] and r["origin"] >= start
    ]
    df = pd.DataFrame(records)

    def group(rows):
        return {
            "origins": len(rows),
            "raw_joint_path_coverage": float(rows.raw_joint_path_covered.mean()),
            "calibrated_joint_path_coverage": float(rows.corrected_joint_path_covered.mean()),
        }

    return {
        "overall": group(df),
        "by_month": {
            str(k): group(v)
            for k, v in df.groupby(
                pd.to_datetime(df.timestamp, utc=True)
                .dt.tz_convert("Asia/Kolkata")
                .dt.strftime("%Y-%m")
            )
        },
        "by_origin_hour": {str(k): group(v) for k, v in df.groupby("phase")},
    }


def select():
    f = pd.read_parquet(DATA)
    b = bounds(f)
    baseline = None
    for capacity in [400, 800]:
        result = simulate({"name": f"rule_{capacity}", "kind": "rule"}, "development", capacity)
        if result["lpsp_energy"] <= 0.01 and result["terminal_full"]:
            baseline = result
            break
    if baseline is None:
        raise ValueError(
            "Validation adequacy or terminal restoration failed for both predeclared designs"
        )
    capacity = baseline["capacity_kwh"]
    margin = 0.0001 * baseline["demand_kwh"]
    candidates = [{"name": "deterministic_age", "kind": "point", "family": "deterministic_age"}]
    candidates += [
        {
            "name": f"adaptive_buffer_{v}",
            "kind": "adaptive",
            "buffer_fraction": v,
            "family": "adaptive_age",
        }
        for v in [0.0, 0.1, 0.25, 0.5, 1.0]
    ]
    candidates += [
        {"name": f"fixed_{v}", "kind": "fixed", "fraction": v, "family": "fixed_age"}
        for v in [0.2, 0.4, 0.6, 0.8, 0.9]
    ]
    candidates += [
        {"name": f"scenario_{v}", "kind": "scenario", "slack": v, "family": "scenario"}
        for v in [0.0, 1.0, 3.0]
    ]
    candidates += [
        {
            "name": f"scenario_envelope_buffer_{v}",
            "kind": "scenario_envelope",
            "buffer_fraction": v,
            "slack": 0.0,
            "family": "scenario_envelope",
        }
        for v in [0.0, 0.25, 0.5, 1.0]
    ]
    candidates += [
        {
            "name": f"scenario_envelope_slack_{v}",
            "kind": "scenario_envelope",
            "slack": v,
            "family": "scenario_envelope",
        }
        for v in [1.0, 3.0]
    ]
    results = []
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
    ) as pool:
        jobs = {pool.submit(simulate, c, "development", capacity): c for c in candidates}
        for job in as_completed(jobs):
            r = job.result()
            results.append(r)
            print(
                f"validation {r['candidate']['name']}: EENS={r['eens_kwh']:.3f}, damage={r['ageing']['25c_24h']['total_loss_fraction']:.6f}, full_endpoint={r['terminal_full']}",
                flush=True,
            )
    selections = {}
    qualification = {}
    for family in sorted(set(c["family"] for c in candidates)):
        group = [r for r in results if r["candidate"]["family"] == family]
        eligible = [r for r in group if validation_eligible(r, baseline, margin)]
        chosen = min(
            eligible or group,
            key=lambda r: (
                r["ageing"]["25c_24h"]["total_loss_fraction"] if eligible else r["eens_kwh"]
            ),
        )
        selections[family] = chosen["candidate"]
        qualification[family] = bool(eligible)
    # Ablations use exactly the proposed policy's selected safety buffer.
    for kind in ["adaptive_no_calendar", "adaptive_raw"]:
        candidate = {
            "name": kind,
            "kind": kind,
            "buffer_fraction": selections["adaptive_age"]["buffer_fraction"],
            "family": kind,
        }
        ablation = simulate(candidate, "development", capacity)
        selections[kind] = candidate
        qualification[kind] = (
            ablation["eens_kwh"] <= baseline["eens_kwh"] + margin and ablation["terminal_full"]
        )
    selections["rule_based"] = {"name": "rule_based", "kind": "rule"}
    cal = calibration_summary("development", b["validation"])
    dump(ROOT / "validation_calibration_summary.json", cal)
    protocol = Path("configs/age_mpc/protocol.yaml")
    frozen = {
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "capacity_kwh": capacity,
        "battery": BatteryConfig(capacity_kwh=capacity, power_kw=50).__dict__,
        "selection": selections,
        "validation_qualified": qualification,
        "validation_baseline": baseline,
        "validation_matching_margin_kwh": margin,
        "calibration_validation_gate_pass": cal["overall"]["calibrated_joint_path_coverage"] >= 0.9,
        "calibration_summary": cal,
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *Path("src/microgrid").glob("*.py")]
        },
        "test_evaluated": False,
        "claim_rule": "No degradation superiority if reliability/endpoint/calibration/applicability gates fail",
    }
    if (ROOT / "frozen_protocol.json").exists():
        raise ValueError("Do not overwrite frozen test protocol")
    dump(ROOT / "frozen_protocol.json", frozen)
    print(
        "FROZEN",
        json.dumps(
            {
                "capacity": capacity,
                "qualified": qualification,
                "calibration": frozen["calibration_validation_gate_pass"],
            }
        ),
        flush=True,
    )


def validation_eligible(row, baseline, margin):
    if row["eens_kwh"] > baseline["eens_kwh"] + margin or not row["terminal_full"]:
        return False
    trace = pd.read_parquet(ROOT / "development" / f"trace_{row['candidate']['name']}.parquet")
    ref = pd.read_parquet(ROOT / "development" / f"trace_{baseline['candidate']['name']}.parquet")
    return all(
        paired_block_bootstrap(
            trace.unserved_kw.iloc[:-24], ref.unserved_kw.iloc[:-24], block=block
        )["ci95"][1]
        <= margin / baseline["operational_hours"]
        for block in [24, 168]
    )


def refine_family(family, capacity, baseline, margin, selection):
    existing = []
    for path in (ROOT / "development").glob("*.json"):
        row = json.loads(path.read_text())
        if row["candidate"].get("family") == family:
            existing.append(row)
    parameter = "fraction" if family in {"fixed_age", "fixed_persistence"} else "buffer_fraction"
    existing.sort(key=lambda r: r["candidate"][parameter])

    def valid(r):
        return validation_eligible(r, baseline, margin)

    qualified = [r for r in existing if valid(r)]
    if not qualified:
        return selection
    high = min(r["candidate"][parameter] for r in qualified)
    low = max(
        [
            r["candidate"][parameter]
            for r in existing
            if not valid(r) and r["candidate"][parameter] < high
        ],
        default=high,
    )
    factor = capacity if family in {"fixed_age", "fixed_persistence"} else capacity * 0.8
    while (high - low) * factor > 2.0:
        value = (low + high) / 2
        candidate = {
            "name": f"{family}_refined_{value:.8f}",
            "kind": "fixed"
            if family in {"fixed_age", "fixed_persistence"}
            else ("adaptive_independent" if family == "adaptive_independent" else "adaptive"),
            parameter: value,
            "forecast": selection.get("forecast", "research"),
            "family": family,
        }
        row = simulate(candidate, "development", capacity)
        existing.append(row)
        print(
            f"refine {family} {value:.6f}: EENS={row['eens_kwh']:.3f}, damage={row['ageing']['25c_24h']['total_loss_fraction']:.6f}",
            flush=True,
        )
        if valid(row):
            high = value
        else:
            low = value
    best = min(
        [r for r in existing if valid(r)],
        key=lambda r: r["ageing"]["25c_24h"]["total_loss_fraction"],
    )
    return best["candidate"]


def refine():
    if (ROOT / "test_metrics.json").exists():
        raise ValueError("Never refine after fresh-test evaluation")
    path = ROOT / "frozen_protocol.json"
    frozen = json.loads(path.read_text())
    if frozen.get("refined"):
        raise ValueError("Refined selection is already frozen")
    archive = ROOT / "coarse_protocol_before_refinement.json"
    if archive.exists():
        raise ValueError("Refinement already started; inspect saved state")
    archive.write_text(path.read_text())
    capacity = frozen["capacity_kwh"]
    baseline = frozen["validation_baseline"]
    margin = frozen["validation_matching_margin_kwh"]
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
    ) as pool:
        jobs = {
            pool.submit(refine_family, f, capacity, baseline, margin, frozen["selection"][f]): f
            for f in ["fixed_age", "adaptive_age"]
        }
        for job in as_completed(jobs):
            frozen["selection"][jobs[job]] = job.result()
    persistence_candidates = [
        {
            "name": f"fixed_persistence_{v}",
            "kind": "fixed",
            "fraction": v,
            "forecast": "persistence",
            "family": "fixed_persistence",
        }
        for v in [0.2, 0.4, 0.6, 0.8, 0.9]
    ]
    persistence_candidates.append(
        {
            "name": "persistence_age",
            "kind": "point",
            "forecast": "persistence",
            "family": "persistence_age",
        }
    )
    persistence_results = []
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
    ) as pool:
        jobs = [pool.submit(simulate, c, "development", capacity) for c in persistence_candidates]
        for job in as_completed(jobs):
            row = job.result()
            persistence_results.append(row)
            print("persistence validation", row["candidate"]["name"], row["eens_kwh"], flush=True)
    base_group = [r for r in persistence_results if r["candidate"]["family"] == "fixed_persistence"]
    valid_group = [r for r in base_group if validation_eligible(r, baseline, margin)]
    initial = min(
        valid_group or base_group,
        key=lambda r: (
            r["ageing"]["25c_24h"]["total_loss_fraction"] if valid_group else r["eens_kwh"]
        ),
    )["candidate"]
    frozen["selection"]["fixed_persistence"] = refine_family(
        "fixed_persistence", capacity, baseline, margin, initial
    )
    frozen["selection"]["persistence_age"] = next(
        r for r in persistence_results if r["candidate"]["family"] == "persistence_age"
    )["candidate"]
    f = pd.read_parquet(DATA)
    b = bounds(f)
    q = np.load(ROOT / "development_predictions.npy", mmap_mode="r")
    generate_inputs(
        f, q, b["fit"], b["cal"], b["test"], "development_independent", independent=True
    )
    independent_candidates = [
        {
            "name": f"independent_buffer_{v}",
            "kind": "adaptive_independent",
            "buffer_fraction": v,
            "family": "adaptive_independent",
        }
        for v in [0.0, 0.1, 0.25, 0.5, 1.0]
    ]
    independent_results = []
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
    ) as pool:
        jobs = [pool.submit(simulate, c, "development", capacity) for c in independent_candidates]
        for job in as_completed(jobs):
            row = job.result()
            independent_results.append(row)
            print("independent validation", row["candidate"]["name"], row["eens_kwh"], flush=True)
    eligible = [r for r in independent_results if validation_eligible(r, baseline, margin)]
    initial = min(
        eligible or independent_results,
        key=lambda r: r["ageing"]["25c_24h"]["total_loss_fraction"] if eligible else r["eens_kwh"],
    )["candidate"]
    frozen["selection"]["adaptive_independent"] = refine_family(
        "adaptive_independent", capacity, baseline, margin, initial
    )
    beta = frozen["selection"]["adaptive_age"].get("buffer_fraction", 0.0)
    candidates = [
        {
            "name": f"scenario_same_buffer_{v}",
            "kind": "scenario_envelope",
            "buffer_fraction": beta,
            "slack": v,
            "family": "scenario_envelope",
        }
        for v in [0.0, 1.0, 3.0]
    ]
    candidates += [
        {"name": k + "_matched_buffer", "kind": k, "buffer_fraction": beta, "family": k}
        for k in ["adaptive_no_calendar", "adaptive_raw"]
    ]
    rerun = []
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
    ) as pool:
        jobs = {pool.submit(simulate, c, "development", capacity): c for c in candidates}
        for job in as_completed(jobs):
            row = job.result()
            rerun.append(row)
            print("matched buffer", row["candidate"]["name"], row["eens_kwh"], flush=True)
    group = []
    for result_path in (ROOT / "development").glob("*.json"):
        row = json.loads(result_path.read_text())
        if row["candidate"].get("family") == "scenario_envelope":
            group.append(row)
    valid = [r for r in group if validation_eligible(r, baseline, margin)]
    chosen = min(
        valid or group,
        key=lambda r: r["ageing"]["25c_24h"]["total_loss_fraction"] if valid else r["eens_kwh"],
    )
    frozen["selection"]["scenario_envelope"] = chosen["candidate"]
    frozen["validation_qualified"]["scenario_envelope"] = bool(valid)
    for k in ["adaptive_no_calendar", "adaptive_raw"]:
        chosen = next(r for r in rerun if r["candidate"]["kind"] == k)
        frozen["selection"][k] = chosen["candidate"]
        frozen["validation_qualified"][k] = (
            chosen["eens_kwh"] <= baseline["eens_kwh"] + margin and chosen["terminal_full"]
        )
    for name, candidate in frozen["selection"].items():
        if name == "rule_based":
            continue
        row = json.loads((ROOT / "development" / f"{candidate['name']}.json").read_text())
        frozen["validation_qualified"][name] = validation_eligible(row, baseline, margin)
    baseline_families = [
        name
        for name in [
            "fixed_age",
            "scenario",
            "scenario_envelope",
            "deterministic_age",
            "adaptive_independent",
            "fixed_persistence",
            "persistence_age",
        ]
        if frozen["validation_qualified"].get(name)
    ]
    frozen["primary_comparator"] = min(
        baseline_families,
        key=lambda name: json.loads(
            (ROOT / "development" / f"{frozen['selection'][name]['name']}.json").read_text()
        )["ageing"]["25c_24h"]["total_loss_fraction"],
    )
    frozen["independent_validation_calibration"] = calibration_summary(
        "development_independent", b["validation"]
    )
    frozen["refined"] = True
    frozen["refinement_resolution_dc_kwh"] = 2.0
    frozen["refinement_reason"] = (
        "Equal energy precision for fixed/adaptive reserve prevents a coarse fixed baseline from creating an apparent advantage; no fresh-test outcomes used."
    )
    frozen["protocol_sha256"] = hashlib.sha256(
        Path("configs/age_mpc/protocol.yaml").read_bytes()
    ).hexdigest()
    frozen["code_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), *Path("src/microgrid").glob("*.py")]
    }
    frozen["model_hashes"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [ROOT / "forecaster.pkl", ROOT / "pool.pkl", ROOT / "development_calibrator.pkl"]
    }
    dump(path, frozen)
    print("REFINED FROZEN", json.dumps(frozen["selection"]), flush=True)


def test():
    frozen = json.loads((ROOT / "frozen_protocol.json").read_text())
    if (ROOT / "test" / "rule_based.json").exists():
        raise ValueError("Final test already run; preserve it")
    f = pd.read_parquet(DATA)
    b = bounds(f)
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != frozen["data_sha256"]:
        raise ValueError("Data changed after freezing")
    for p, h in frozen["code_sha256"].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise ValueError(f"Frozen source changed: {p}")
    if (
        hashlib.sha256(Path("configs/age_mpc/protocol.yaml").read_bytes()).hexdigest()
        != frozen["protocol_sha256"]
    ):
        raise ValueError("Protocol changed after freezing")
    for name, expected in frozen.get("model_hashes", {}).items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen model changed")
    model = pickle.loads((ROOT / "forecaster.pkl").read_bytes())
    forecast_start = time.perf_counter()
    q = model.predict_many(f, np.arange(b["test"], len(f)))
    dump(
        ROOT / "test_forecast_runtime.json",
        {
            "seconds": time.perf_counter() - forecast_start,
            "origins": len(q),
            "mode": "batched inference; not a live single-call latency",
        },
    )
    np.save(ROOT / "test_predictions.npy", q.astype(np.float32))
    cal = pickle.loads((ROOT / "development_calibrator.pkl").read_bytes())
    generate_inputs(f, q, b["test"], b["test"], len(f), "test", cal)
    independent_cal = pickle.loads((ROOT / "development_independent_calibrator.pkl").read_bytes())
    generate_inputs(
        f, q, b["test"], b["test"], len(f), "test_independent", independent_cal, independent=True
    )

    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
    ) as pool:
        jobs = {
            pool.submit(simulate, c, "test", frozen["capacity_kwh"]): name
            for name, c in frozen["selection"].items()
        }
        results = {}
        for job in as_completed(jobs):
            name = jobs[job]
            r = job.result()
            results[name] = r
            print(
                f"test {name}: EENS={r['eens_kwh']:.3f}, damage={r['ageing']['25c_24h']['total_loss_fraction']:.6f}",
                flush=True,
            )
    dump(ROOT / "test_metrics.json", results)
    report(results, frozen)


def coupled_sensitivity():
    frozen = json.loads((ROOT / "frozen_protocol.json").read_text())
    names = list(dict.fromkeys(["rule_based", "adaptive_age", frozen["primary_comparator"]]))
    results = {}
    for temperature in [25.0, 35.0]:
        with ProcessPoolExecutor(
            max_workers=2, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1
        ) as pool:
            jobs = {
                pool.submit(
                    simulate,
                    frozen["selection"][name],
                    "test",
                    frozen["capacity_kwh"],
                    True,
                    temperature,
                ): name
                for name in names
            }
            for job in as_completed(jobs):
                name = jobs[job]
                r = job.result()
                results[f"{name}_{int(temperature)}c"] = r
                print(
                    f"coupled {name} {temperature}C EENS={r['eens_kwh']:.3f}, SOH={r['live_final_soh']:.6f}",
                    flush=True,
                )
    dump(ROOT / "coupled_metrics.json", results)


def report(results=None, frozen=None):
    results = results or json.loads((ROOT / "test_metrics.json").read_text())
    frozen = frozen or json.loads((ROOT / "frozen_protocol.json").read_text())
    reference = results["rule_based"]
    margin = 0.0001 * reference["demand_kwh"]
    comparisons = {}
    for name, r in results.items():
        if name == "rule_based":
            continue
        path = ROOT / "test" / f"trace_{r['candidate']['name']}.parquet"
        ours = pd.read_parquet(path)
        base = pd.read_parquet(ROOT / "test/trace_rule_based.parquet")
        intervals = {}
        for block in [24, 168]:
            eens = paired_block_bootstrap(
                ours.unserved_kw.iloc[:-24], base.unserved_kw.iloc[:-24], block=block
            )
            damage = paired_block_bootstrap(ours.damage_25c_24h, base.damage_25c_24h, block=block)
            intervals[str(block)] = {
                "EENS_mean_hourly_difference": eens,
                "damage_mean_hourly_difference": damage,
            }
        noninferior = r["eens_kwh"] <= reference["eens_kwh"] + margin and all(
            v["EENS_mean_hourly_difference"]["ci95"][1] <= margin / reference["operational_hours"]
            for v in intervals.values()
        )
        comparisons[name] = {
            "point_reliability_match": r["eens_kwh"] <= reference["eens_kwh"] + margin,
            "bootstrap_noninferiority": noninferior,
            "endpoint_match": r["terminal_full"] and reference["terminal_full"],
            "model_damage_reduction_fraction": 1
            - r["ageing"]["25c_24h"]["total_loss_fraction"]
            / reference["ageing"]["25c_24h"]["total_loss_fraction"],
            "damage_sign_robust_across_windows_and_temperatures": all(
                r["ageing"][k]["total_loss_fraction"]
                < reference["ageing"][k]["total_loss_fraction"]
                for k in reference["ageing"]
            ),
            "paired_intervals": intervals,
        }
    strong = {}
    proposed = results["adaptive_age"]
    proposed_trace = pd.read_parquet(
        ROOT / "test" / f"trace_{proposed['candidate']['name']}.parquet"
    )
    for name in [
        "fixed_age",
        "scenario",
        "scenario_envelope",
        "deterministic_age",
        "adaptive_independent",
        "fixed_persistence",
        "persistence_age",
    ]:
        other = results[name]
        other_trace = pd.read_parquet(ROOT / "test" / f"trace_{other['candidate']['name']}.parquet")
        pairs = {}
        for block in [24, 168]:
            pairs[str(block)] = {
                "EENS": paired_block_bootstrap(
                    proposed_trace.unserved_kw.iloc[:-24],
                    other_trace.unserved_kw.iloc[:-24],
                    block=block,
                ),
                "damage": paired_block_bootstrap(
                    proposed_trace.damage_25c_24h, other_trace.damage_25c_24h, block=block
                ),
            }
        reduction = (
            1
            - proposed["ageing"]["25c_24h"]["total_loss_fraction"]
            / other["ageing"]["25c_24h"]["total_loss_fraction"]
        )
        eligible = (
            comparisons["adaptive_age"]["bootstrap_noninferiority"]
            and comparisons[name]["bootstrap_noninferiority"]
            and comparisons["adaptive_age"]["endpoint_match"]
            and comparisons[name]["endpoint_match"]
        )
        strong[name] = {
            "both_match_reference_reliability_and_inventory": eligible,
            "model_damage_reduction_fraction": reduction,
            "practical_reduction_at_least_one_percent": reduction >= 0.01,
            "paired_intervals": pairs,
            "stable_damage_sign": all(
                proposed["ageing"][k]["total_loss_fraction"]
                < other["ageing"][k]["total_loss_fraction"]
                for k in other["ageing"]
            ),
        }
    cal = calibration_summary("test")
    dump(ROOT / "test_calibration_summary.json", cal)
    dump(
        ROOT / "matched_reliability.json",
        {
            "primary_comparator": frozen.get("primary_comparator"),
            "matching_reference": "rule_based",
            "margin_kwh": margin,
            "comparisons": comparisons,
            "proposed_vs_strong_baselines": strong,
            "calibration": cal,
            "scope": "published-model predictions, not empirical pack lifetime",
            "normalization": "Last 24h common recharge-only endpoint normalization; operational EENS excludes this day, ageing includes it",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "select", "refine", "test", "coupled", "report"]
    )
    stage = parser.parse_args().stage
    {
        "prepare": prepare,
        "select": select,
        "refine": refine,
        "test": test,
        "coupled": coupled_sensitivity,
        "report": report,
    }[stage]()
