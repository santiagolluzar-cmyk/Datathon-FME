"""Explicit, reproducible observation/initial-state sensitivity experiments.

Use each fold's own fitted bundle on its own held-out data. These experiments
do not make an in-sample bundle out-of-sample and do not consume locked periods
automatically. Default CLI scope is the three development periods only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .evaluate import REPO, load_data, run
except ImportError:
    from evaluate import REPO, load_data, run


SCENARIOS = {
    "reference": {},
    "soc_reset_at_gaps": {"gap_soc": "reset"},
    "initial_soc_empty": {"initial_soc": 0.0},
    "initial_soc_full": {"initial_soc": 6000.0},
    "one_known_market_hour": {"market_len": 1},
    "two_known_market_hours": {"market_len": 2},
    "history_reset_daily": {"reset_history_every": 24},
    "history_reset_every_6h": {"reset_history_every": 6},
    "missing_telemetry_at_starts": {"telemetry_mode": "missing_at_starts"},
    "current_telemetry_at_starts": {"telemetry_mode": "current_at_starts"},
}


def inventory_diagnostic(result, data):
    """Ex-post accounting sensitivity, NOT official score or deployable policy.

    Value surplus terminal stored energy at discharge efficiency and charge
    a terminal inventory deficit at recharge efficiency. Valuation uses mean
    observed price/carbon over the evaluated period, so this is a transparent
    accounting convention, not a guarantee of recoverable future savings.
    Reset energy is accounted for explicitly and not counted as earned energy.
    """
    delta = (result["soc_final_kwh"] - result["soc_initial_kwh"]
             - result["soc_reset_energy_kwh"])
    equivalent = delta * math.sqrt(.9) if delta >= 0 else delta / math.sqrt(.9)
    carbon_unit = (100 * .7 * float(data.co2_kg_per_kwh.mean()) / result["co2_baseline_kg"]
                   if result["co2_baseline_kg"] > 0 else 0.)
    price_unit = (100 * .3 * float(data.precio_eur_per_kwh.mean()) / result["cost_baseline_eur"]
                  if result["cost_baseline_eur"] > 0 else 0.)
    adjustment = equivalent * (carbon_unit + price_unit)
    return dict(label="Ex-post mean-market inventory accounting; not official score",
                net_stored_energy_change_kwh=delta, grid_equivalent_kwh=equivalent,
                score_adjustment=adjustment,
                inventory_neutral_score=max(0., result["score_unclipped"] + adjustment))


def run_scenarios(data, variants, scenario_names=None, output=None, traces=None,
                  expected_models=None, audit_exceptions=True, progress=True):
    """Run fixed policies under predeclared sensitivities, saving after each run.

    ``variants`` maps labels to self-contained agent directories. Root caller
    owns split/model alignment; no automatic model training or selection occurs.
    The first reference run is always included for matched deltas.
    """
    names = ["reference"] + [n for n in (scenario_names or list(SCENARIOS)) if n != "reference"]
    if set(names) - set(SCENARIOS):
        raise ValueError("Unknown scenarios: " + str(sorted(set(names) - set(SCENARIOS))))
    results = {"method": "Complete agents; predeclared sensitivity scenarios; aggregate totals",
               "hours": len(data), "results": {}, "scenarios": {n: SCENARIOS[n] for n in names}}
    for label, directory in variants.items():
        results["results"][label] = {}
        for scenario in names:
            trace = Path(traces) / f"{label}__{scenario}.csv" if traces else None
            result = run(data, directory, label=label, trace_path=trace,
                         audit_exceptions=audit_exceptions,
                         expected_models=(expected_models or {}).get(label), **SCENARIOS[scenario])
            result["terminal_inventory_diagnostic"] = inventory_diagnostic(result, data)
            result["delta_to_same_agent_reference"] = (
                result["score"] - results["results"][label]["reference"]["score"] if scenario != "reference" else 0.)
            results["results"][label][scenario] = result
            if progress:
                print(label, scenario, f"score={result['score']:.6f}",
                      f"SOC={result['soc_final_kwh']:.1f}", f"errors={result['errors']}", flush=True)
            if output:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                Path(output).write_text(json.dumps(results, indent=2))
    return results


def aggregate_runs(runs):
    """Aggregate independently evaluated fold runs by totals, never mean score."""
    fields = ["cost_participant_eur", "cost_baseline_eur", "co2_participant_kg",
              "co2_baseline_kg", "violations", "hours", "errors"]
    output = {key: sum(r[key] for r in runs) for key in fields}
    output["co2_saved_pct"] = 100 * (1-output["co2_participant_kg"]/output["co2_baseline_kg"])
    output["cost_saved_pct"] = 100 * (1-output["cost_participant_eur"]/output["cost_baseline_eur"])
    output["score"] = max(0., .7*output["co2_saved_pct"]+.3*output["cost_saved_pct"]-.2*output["violations"])
    output["fold_scores"] = [r["score"] for r in runs]
    output["soc_final_kwh_by_fold"] = [r["soc_final_kwh"] for r in runs]
    output["runtime_seconds"] = sum(r["runtime"]["total_seconds"] for r in runs)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="append", required=True, help="LABEL=/absolute/or/relative/agent_directory")
    parser.add_argument("--data", type=Path, help="Hourly CSV already scaled to hard-mode solar x12")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--start")
    parser.add_argument("--end", help="Exclusive")
    parser.add_argument("--development", action="store_true", help="Select development periods from split_manifest.json")
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", type=Path)
    args = parser.parse_args()
    variants = dict(item.split("=", 1) for item in args.agent)
    data = pd.read_csv(args.data, parse_dates=["Fecha"]) if args.data else load_data(cache=args.cache)
    if args.development:
        splits = json.loads((REPO/"research/split_manifest.json").read_text())
        mask = np.zeros(len(data), bool)
        for part in splits["development"]:
            mask |= (data.Fecha >= part["start"]) & (data.Fecha < part["end"])
        data = data.loc[mask]
    elif not args.start or not args.end:
        parser.error("Specify --development or both --start and --end; locked data is never opened implicitly")
    if args.start:
        data = data.loc[data.Fecha >= args.start]
    if args.end:
        data = data.loc[data.Fecha < args.end]
    run_scenarios(data, variants, args.scenario, output=args.output, traces=args.traces)


if __name__ == "__main__":
    main()
