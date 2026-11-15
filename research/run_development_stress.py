"""Run the preselected four policies under seven development-only sensitivities."""
import json
from pathlib import Path

try:
    from .evaluate import REPO, load_data
    from .stress import run_scenarios, aggregate_runs
except ImportError:
    from evaluate import REPO, load_data
    from stress import run_scenarios, aggregate_runs


NAMES = ["lgb_direct_empirical", "analog_residual_quantiles", "blend_market_0.5", "market75_k16"]
SCENARIOS = ["reference", "initial_soc_empty", "initial_soc_full", "one_known_market_hour",
             "two_known_market_hours", "history_reset_daily", "history_reset_every_6h"]


def main():
    path = REPO/"research/results/development_stress.json"
    data = load_data(cache=REPO/"research/data_cache.pkl")
    parts = json.loads((REPO/"research/split_manifest.json").read_text())["development"]
    output = dict(method="Own independently fitted fold models; complete agents; scores from aggregate totals",
                  scenarios=SCENARIOS, names=NAMES, folds={}, aggregate={})
    for part in parts:
        name = part["name"]
        selected = data.loc[(data.Fecha >= part["start"]) & (data.Fecha < part["end"])]
        dirs = {label: REPO/"research/agents"/name/label for label in NAMES}
        output["folds"][name] = run_scenarios(selected, dirs, SCENARIOS,
            output=REPO/f"research/results/development_stress_{name}.json")
        path.write_text(json.dumps(output, indent=2))
    for scenario in SCENARIOS:
        output["aggregate"][scenario] = {}
        for name in NAMES:
            runs = [output["folds"][p["name"]]["results"][name][scenario] for p in parts]
            output["aggregate"][scenario][name] = aggregate_runs(runs)
        reference = output["aggregate"][scenario][NAMES[0]]["score"]
        prototype = output["aggregate"][scenario][NAMES[1]]["score"]
        for name in NAMES:
            output["aggregate"][scenario][name]["delta_vs_baseline"] = output["aggregate"][scenario][name]["score"]-reference
            output["aggregate"][scenario][name]["delta_vs_prototype"] = output["aggregate"][scenario][name]["score"]-prototype
        print("AGGREGATE", scenario, {k: round(v["score"], 6) for k,v in output["aggregate"][scenario].items()}, flush=True)
    path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
