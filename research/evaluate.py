"""Portable end-to-end hard-mode evaluation; never contacts the competition server.

Input data must already contain solar scaled by 12 exactly once. ``load_data``
uses the historical published-data loader and performs that one scaling.
Scores are computed from aggregate totals, without intermediate rounding.
Python exception monitoring detects errors swallowed by an agent's fallbacks;
disable it only for a separately labelled latency benchmark after a clean run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from starter_kit.environment import CampusBatteryEnv


def load_data(repo=REPO, cache=None):
    """Load public data (solar x12); optional local pickle cache with source hashes."""
    repo = Path(repo).resolve()
    sources = sorted((repo / "data/train").glob("*.csv"))
    fingerprint = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    fingerprint["loader"] = hashlib.sha256((repo / "eval_local.py").read_bytes()).hexdigest()
    fingerprint["hard_loader"] = hashlib.sha256((repo / "eval_dificil.py").read_bytes()).hexdigest()
    if cache is not None:
        cache = Path(cache)
        meta = cache.with_suffix(cache.suffix + ".json")
        if cache.exists() and meta.exists() and json.loads(meta.read_text()) == fingerprint:
            return pd.read_pickle(cache)
    spec = importlib.util.spec_from_file_location("evaluation_dataset_loader", repo / "eval_dificil.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.carregar_dades(repo)
    data.attrs["solar_scale"] = 12
    data.attrs["source_fingerprint"] = fingerprint
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        data.to_pickle(cache)
        meta.write_text(json.dumps(fingerprint, indent=2))
    return data


def totals(grid, base, price, carbon, violations=0):
    """Aggregate first; never average block scores or rounded percentages."""
    cost = float(np.dot(grid, price))
    cost_base = float(np.dot(base, price))
    co2 = float(np.dot(grid, carbon))
    co2_base = float(np.dot(base, carbon))
    dcost = 100 * (cost_base - cost) / cost_base if cost_base > 0 else 0.0
    dco2 = 100 * (co2_base - co2) / co2_base if co2_base > 0 else 0.0
    raw = 0.7 * dco2 + 0.3 * dcost - 0.20 * int(violations)
    return {
        "score": max(0.0, raw), "score_unclipped": raw,
        "co2_saved_pct": dco2, "cost_saved_pct": dcost,
        "co2_baseline_kg": co2_base, "co2_participant_kg": co2,
        "cost_baseline_eur": cost_base, "cost_participant_eur": cost,
        "co2_saved_kg": co2_base - co2, "cost_saved_eur": cost_base - cost,
        "violations": int(violations),
        # Compatible labels for the earlier scripts, now intentionally unrounded.
        "co2_estalviat_pct": dco2, "cost_estalviat_pct": dcost,
        "infraccions": int(violations),
    }


class HardEnv(CampusBatteryEnv):
    """Supplied physical simulator with the observable hard-mode interface."""

    def __init__(self, data, initial_soc=3000.0, market_len=3, telemetry_mode="kit"):
        self.market_len = market_len
        self.telemetry_mode = telemetry_mode
        self.ts = pd.to_datetime(data.Fecha).reset_index(drop=True)
        self.starts = np.r_[True, self.ts.diff().iloc[1:].ne(pd.Timedelta(hours=1)).to_numpy()]
        self.block_ids = np.cumsum(self.starts) - 1
        self.block_end = np.empty(len(data), dtype=int)
        ends = np.r_[np.flatnonzero(self.starts)[1:], len(data)]
        for start, end in zip(np.flatnonzero(self.starts), ends):
            self.block_end[start:end] = end
        self.arrays = {k: data[k].to_numpy(dtype=float) for k in
                       ("load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh")}
        for key in ("irr", "tamb", "tpan", "wind"):
            self.arrays[key] = data[key].to_numpy(dtype=float) if key in data else np.full(len(data), np.nan)
        super().__init__(data, spec=dict(capacity_kwh=6000, power_kw=1500,
            efficiency=.90, soc_init=initial_soc/6000, soc_min=0, soc_max=1, dt_hours=1),
            export_price=0.0)

    def _get_obs(self):
        if self.t >= self.n:
            return None
        i, arrays, ts = self.t, self.arrays, self.ts.iloc[self.t]
        end = min(i + self.market_len, self.block_end[i])
        previous = max(0, i - 1)
        if self.telemetry_mode == "current_at_starts" and self.starts[i]:
            previous = i
        missing = self.telemetry_mode == "missing_at_starts" and self.starts[i]
        telemetry = {out: float("nan") if missing else float(arrays[key][previous]) for out, key in (
            ("ultim_consum_kw", "load_kw"), ("ultima_solar_kw", "pv_kw"),
            ("irradiancia_wm2", "irr"), ("temp_ambient_c", "tamb"),
            ("temp_panel_c", "tpan"), ("vent_ms", "wind"))}
        return dict(timestamp=ts, hour=ts.hour, day_of_week=ts.dayofweek,
            soc_kwh=float(self.soc), soc_pct=float(self.soc / self.cap),
            soc_percent=float(100 * self.soc / self.cap), cap_max_kwh=self.cap,
            power_max_kw=self.power, preu_actual=float(arrays["precio_eur_per_kwh"][i]),
            co2_actual=float(arrays["co2_kg_per_kwh"][i]),
            preus_properes_24h=arrays["precio_eur_per_kwh"][i:end].tolist(),
            co2_properes_24h=arrays["co2_kg_per_kwh"][i:end].tolist(),
            noves_dades_5min=telemetry)


class ExceptionAudit:
    """Observe caught Python exceptions in submitted code, without patching it.

    Libraries themselves are not traced. Exceptions from them propagate to an
    agent frame and are still observed there even when the agent catches them.
    Normal iterator completion is excluded. One exception propagated through
    several agent frames is recorded once.
    """

    def __init__(self, directory, enabled=True):
        self.prefix = str(Path(directory).resolve())
        self.normalized_prefix = os.path.normcase(os.path.normpath(self.prefix)) + os.sep
        self.enabled, self.errors, self.step = enabled, [], None
        self._seen, self._references = set(), []

    def trace(self, frame, event, arg):
        if not os.path.normcase(os.path.normpath(frame.f_code.co_filename)).startswith(self.normalized_prefix):
            return None
        if event == "exception":
            typ, value, _ = arg
            if not issubclass(typ, (StopIteration, StopAsyncIteration, GeneratorExit)) and id(value) not in self._seen:
                self._seen.add(id(value))
                self._references.append(value)
                self.errors.append(dict(step=self.step, type=typ.__name__, message=str(value),
                    file=str(Path(frame.f_code.co_filename).relative_to(self.prefix)),
                    function=frame.f_code.co_name, line=frame.f_lineno))
        # Suppress line events while retaining exception/return events.
        frame.f_trace_lines = False
        frame.f_trace_opcodes = False
        return self.trace

    def __enter__(self):
        self.old = sys.gettrace()
        if self.enabled:
            sys.settrace(self.trace)
        return self

    def __exit__(self, *exc):
        if self.enabled:
            sys.settrace(self.old)


def artifact_inventory(agent_dir):
    return {str(p.relative_to(agent_dir)): {"bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(Path(agent_dir).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".txt", ".json", ".npz")}


def run(data, agent_dir, gap_soc="carry", initial_soc=3000.0, market_len=3,
        reset_history_every=None, telemetry_mode="kit", label=None, trace_path=None,
        strict=True, audit_exceptions=True, expected_models=None):
    """Execute a complete delivery agent; return unrounded metrics and provenance.

    ``gap_soc='reset'`` injects/removes energy to restore ``initial_soc`` at each
    discontinuity, explicitly reported as ``soc_reset_energy_kwh``. Normal carry
    does not change SOC at gaps. ``reset_history_every`` is a cold-start stress
    test, independent of SOC. There is no terminal liquidation or SOC credit.
    ``data`` is already scaled and may contain disjoint chronological periods.
    No hidden future weather/load/PV or market beyond the legal window is passed.
    """
    if gap_soc not in ("carry", "reset"):
        raise ValueError("gap_soc must be carry or reset")
    if market_len not in (1, 2, 3):
        raise ValueError("Only legal 1-, 2-, or 3-hour market vectors are permitted")
    if not 0 <= initial_soc <= 6000:
        raise ValueError("initial_soc must be in [0, 6000] kWh")
    if telemetry_mode not in ("kit", "missing_at_starts", "current_at_starts"):
        raise ValueError("Unknown telemetry mode")
    if reset_history_every is not None and reset_history_every < 1:
        raise ValueError("reset_history_every must be positive")
    agent_dir = Path(agent_dir).resolve()
    data = data.copy().reset_index(drop=True)
    if data.empty:
        raise ValueError("Cannot evaluate an empty period")
    data["Fecha"] = pd.to_datetime(data.Fecha)
    if not data.Fecha.is_monotonic_increasing or data.Fecha.duplicated().any():
        raise ValueError("Timestamps must be strictly increasing")
    required = data[["load_kw", "pv_kw", "precio_eur_per_kwh", "co2_kg_per_kwh"]].to_numpy(float)
    if not np.isfinite(required).all():
        raise ValueError("Non-finite physical or market inputs")
    inventory = artifact_inventory(agent_dir)
    env = HardEnv(data, initial_soc=initial_soc, market_len=market_len, telemetry_mode=telemetry_mode)
    audit = ExceptionAudit(agent_dir, enabled=audit_exceptions)
    module_name = "battery_evaluated_" + hashlib.sha256(str(agent_dir).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, agent_dir / "agent.py")
    module = importlib.util.module_from_spec(spec)
    start = time.perf_counter()
    with audit:
        import_start = time.perf_counter()
        spec.loader.exec_module(module)
        import_seconds = time.perf_counter() - import_start
        init_start = time.perf_counter()
        agent = module.BatteryAgent()
        init_seconds = time.perf_counter() - init_start
        if strict and audit.errors:
            raise RuntimeError("Agent import/initialization swallowed an exception: " + json.dumps(audit.errors))
        model_objects = getattr(agent, "models", {})
        if expected_models is not None and set(expected_models) - set(model_objects):
            raise RuntimeError("Missing required loaded models: " + str(sorted(set(expected_models) - set(model_objects))))
        obs = env.reset()
        actions, startsoc, elapsed_act, resets, reset_energy = [], [], [], [], []
        while obs is not None:
            i = env.t
            if i > 0 and env.starts[i] and gap_soc == "reset":
                reset_energy.append(float(initial_soc - env.soc))
                env.soc = float(initial_soc)
                obs = env._get_obs()
            if i > 0 and reset_history_every and i % reset_history_every == 0:
                hist = getattr(agent, "hist", getattr(agent, "history", None))
                if hist is None or not hasattr(hist, "reset"):
                    raise ValueError("Explicit history reset requested but agent has no resettable history")
                hist.reset()
                resets.append(i)
            startsoc.append(float(env.soc))
            audit.step = i
            before = len(audit.errors)
            t0 = time.perf_counter()
            action = float(agent.act(obs))
            elapsed_act.append(time.perf_counter() - t0)
            if not math.isfinite(action):
                raise ValueError(f"Non-finite action at row {i}")
            if strict and len(audit.errors) > before:
                raise RuntimeError("Agent swallowed an exception: " + json.dumps(audit.errors[before:]))
            actions.append(action)
            obs, _ = env.step(action)
    total_seconds = time.perf_counter() - start
    grid = np.asarray(env.grid, float)
    baseline = np.maximum(data.load_kw.to_numpy() - data.pv_kw.to_numpy(), 0.)
    price, carbon = data.precio_eur_per_kwh.to_numpy(), data.co2_kg_per_kwh.to_numpy()
    violations = np.asarray(env.infraccions_per_step, int)
    result = totals(grid, baseline, price, carbon, violations.sum())
    model_status = {}
    for name, model in getattr(agent, "models", {}).items():
        model_status[name] = {"trees": int(model.num_trees()) if hasattr(model, "num_trees") else None,
                              "features": int(model.num_feature()) if hasattr(model, "num_feature") else None}
    blocks = []
    for bid in range(int(env.block_ids.max()) + 1):
        sel = env.block_ids == bid
        idx = np.flatnonzero(sel)
        blocks.append({"block": bid, "start": str(data.Fecha.iloc[idx[0]]),
            "end": str(data.Fecha.iloc[idx[-1]]), "hours": len(idx),
            "soc_initial_kwh": startsoc[idx[0]], "soc_final_kwh": float(env.soc_series[idx[-1]]),
            **totals(grid[sel], baseline[sel], price[sel], carbon[sel], violations[sel].sum())})
    result.update(label=label or agent_dir.name, hours=len(data), start=str(data.Fecha.iloc[0]),
        end=str(data.Fecha.iloc[-1]), periods=blocks,
        config=dict(gap_soc=gap_soc, initial_soc_kwh=initial_soc, market_len=market_len,
            reset_history_every=reset_history_every, telemetry_mode=telemetry_mode,
            terminal_credit=False, exception_monitoring=audit_exceptions, strict=strict),
        soc_initial_kwh=initial_soc, soc_final_kwh=float(env.soc),
        soc_min_kwh=float(min(startsoc + env.soc_series)), soc_max_kwh=float(max(startsoc + env.soc_series)),
        soc_reset_energy_kwh=float(sum(reset_energy)), soc_resets=len(reset_energy),
        explicit_history_resets=resets, errors=len(audit.errors), error_details=audit.errors,
        model_status=model_status, market_loaded=getattr(agent, "market", None) is not None,
        artifact_inventory=inventory,
        runtime=dict(import_seconds=import_seconds, init_seconds=init_seconds,
            act_total_seconds=float(sum(elapsed_act)), act_mean_ms=float(np.mean(elapsed_act)*1000),
            act_max_ms=float(max(elapsed_act)*1000), total_seconds=total_seconds),
        grid_kwh=float(grid.sum()), baseline_grid_kwh=float(baseline.sum()), export_kwh=float(sum(env.exports)),
        # Compatibility fields used in historical comparisons.
        hores=len(data), temps_s=total_seconds, errors_planificacio=len(audit.errors),
        models_carregats=list(model_status), mercat_analog_carregat=getattr(agent, "market", None) is not None)
    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": data.Fecha, "block": env.block_ids, "soc_before_kwh": startsoc,
            "action_kw": actions, "soc_after_kwh": env.soc_series, "load_kw": data.load_kw,
            "pv_scaled_kw": data.pv_kw, "grid_kwh": grid, "baseline_grid_kwh": baseline,
            "price_eur_per_kwh": price, "carbon_kg_per_kwh": carbon,
            "violations": violations, "act_seconds": elapsed_act}).to_csv(trace_path, index=False)
        result["trace"] = str(trace_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="Optional prebuilt hourly CSV; PV must already be x12")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end", help="Exclusive end timestamp")
    parser.add_argument("--max-hours", type=int)
    parser.add_argument("--gap-soc", choices=["carry", "reset"], default="carry")
    parser.add_argument("--initial-soc", type=float, default=3000)
    parser.add_argument("--market-len", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--reset-history-every", type=int)
    parser.add_argument("--telemetry-mode", choices=["kit", "missing_at_starts", "current_at_starts"], default="kit")
    parser.add_argument("--no-exception-audit", action="store_true", help="Latency benchmark only after an audited run")
    parser.add_argument("--allow-errors", action="store_true")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.data, parse_dates=["Fecha"]) if args.data else load_data(args.repo, args.cache)
    if args.start:
        data = data.loc[data.Fecha >= pd.Timestamp(args.start)]
    if args.end:
        data = data.loc[data.Fecha < pd.Timestamp(args.end)]
    if args.max_hours:
        data = data.iloc[:args.max_hours]
    result = run(data, args.agent_dir, gap_soc=args.gap_soc, initial_soc=args.initial_soc,
        market_len=args.market_len, reset_history_every=args.reset_history_every,
        telemetry_mode=args.telemetry_mode, trace_path=args.trace,
        audit_exceptions=not args.no_exception_audit, strict=not args.allow_errors)
    text = json.dumps(result, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text)


if __name__ == "__main__":
    main()
