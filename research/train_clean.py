"""Fixed-recipe, purged training with train-only profiles and OOF residuals.

All paths are portable relative to this project. There are no web operations.
Input ``data`` must have hourly ``pv_kw`` already multiplied by 12 exactly once,
as returned by ``hard_work.training.load_data``. Evaluation labels are never
used to fit models, profiles, residual tables, or analogs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hard_work"))
from training import load_data, module_at, build_features, rebuild_prior
from market_analog import build_library

QUANTILE_ROUNDS = (142, 122, 81, 106, 374, 155, 76)
RECIPES = {
    "direct": dict(rounds=600, learning_rate=.03, num_leaves=63,
                   min_data_in_leaf=40, lambda_l2=1, seed=0,
                   feature_fraction=.8, bagging_fraction=.8),
    "residual": dict(rounds=79, learning_rate=.035, num_leaves=15,
                     min_data_in_leaf=500, lambda_l2=5, seed=83,
                     feature_fraction=.85, bagging_fraction=.85),
    "price": dict(rounds=500, learning_rate=.03, num_leaves=31,
                  min_data_in_leaf=100, lambda_l2=5, seed=0,
                  feature_fraction=.8, bagging_fraction=.8),
    "co2": dict(rounds=500, learning_rate=.03, num_leaves=31,
                min_data_in_leaf=100, lambda_l2=5, seed=0,
                feature_fraction=.8, bagging_fraction=.8),
}


def net_baseline(x):
    return np.where(np.isfinite(x[..., 25]), x[..., 25], x[..., 6]) - np.where(
        np.isfinite(x[..., 23]), x[..., 23], x[..., 7])


def span_records(data, mask):
    """Record exact disjoint continuous selected spans without hiding gaps."""
    mask = np.asarray(mask, bool)
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    records = []
    start = prev = int(indices[0])
    for i in indices[1:]:
        i = int(i)
        if i != prev + 1 or data.Fecha.iloc[i] - data.Fecha.iloc[prev] != pd.Timedelta(hours=1):
            records.append(dict(start=str(data.Fecha.iloc[start]), end=str(data.Fecha.iloc[prev]), hours=prev-start+1))
            start = i
        prev = i
    records.append(dict(start=str(data.Fecha.iloc[start]), end=str(data.Fecha.iloc[prev]), hours=prev-start+1))
    return records


def choose_inner_calibration(data, train_mask):
    """Two complete training-only weeks chosen by calendar position, not scores."""
    weeks = data.Fecha.dt.to_period("W-SUN")
    eligible = []
    for week in sorted(weeks[train_mask].unique()):
        ii = np.flatnonzero((weeks == week).to_numpy())
        if (len(ii) == 168 and np.asarray(train_mask)[ii].all()
                and data.Fecha.iloc[ii[0]] == week.start_time
                and data.Fecha.iloc[ii[-1]] == week.end_time.floor("h")
                and data.Fecha.iloc[ii].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()):
            eligible.append(week)
    if len(eligible) < 3:
        raise ValueError("Need at least three complete training weeks for independent residual calibration")
    positions = sorted(set([len(eligible)//3, 2*len(eligible)//3]))
    chosen = [eligible[p] for p in positions]
    return weeks.isin(chosen).to_numpy() & np.asarray(train_mask, bool)


def eligible_origins(features, train_mask):
    """Every target in the complete 24h trajectory must be permitted."""
    n = len(train_mask)
    origins = np.arange(n)
    good = np.asarray(train_mask, bool).copy()
    blocks = features["block_id"]
    for h in range(24):
        target = np.minimum(origins+h, n-1)
        good &= (origins+h < n) & train_mask[target] & (blocks == blocks[target])
    return good


def make_features(data, train_mask, original, module):
    prior = rebuild_prior(data, train_mask, module, original)
    # Splitting on mask changes prevents validation telemetry from entering
    # training rows after a holdout, and truncates known market vectors at splits.
    features = build_features(data, module, prior, split_mask=np.asarray(train_mask, bool))
    return prior, features


def fit_model(kind, features, train_mask, threads):
    n = len(train_mask)
    allowed = eligible_origins(features, train_mask)
    origins = np.repeat(np.flatnonzero(allowed), 24)
    leads = np.tile(np.arange(24), int(allowed.sum()))
    if not len(origins):
        raise ValueError("No complete training trajectories")
    key = "Xn" if kind in ("direct", "residual") else "Xm"
    x = features[key][origins, leads]
    if kind in ("direct", "residual"):
        y = (features["load"]-features["pv"])[origins+leads]
        if kind == "residual":
            y = y-net_baseline(x)
    else:
        y = features["price" if kind == "price" else "carbon"][origins+leads]
        y = y-x[:, 27 if kind == "price" else 28]
        keep = leads >= 3
        x, y = x[keep], y[keep]
    recipe = RECIPES[kind].copy()
    rounds = recipe.pop("rounds")
    params = dict(objective="regression", metric="l2", verbosity=-1,
                  num_threads=threads, bagging_freq=1, **recipe)
    model = lgb.train(params, lgb.Dataset(x, label=y), num_boost_round=rounds)
    return model, dict(rows=len(y), origins=int(allowed.sum()),
                       trees=model.num_trees(), recipe=RECIPES[kind])


def residual_table(features, model, calibration_mask, residual=False):
    n = len(calibration_mask)
    quantiles = (np.arange(15)+.5)/15
    x = features["Xn"]
    pred = model.predict(x.reshape(-1, x.shape[-1]), num_threads=1).reshape(n, 24)
    if residual:
        pred += net_baseline(x)
    actual = features["load"]-features["pv"]
    table, metrics = {}, {}
    for h in range(24):
        origins = np.arange(n-h)
        allowed = calibration_mask[origins] & calibration_mask[origins+h]
        allowed &= features["block_id"][origins] == features["block_id"][origins+h]
        error = actual[origins+h]-pred[origins, h]
        for day in (0, 1):
            selected = error[allowed & ((x[origins, h, 8] > 50) == day)]
            fallback = False
            if not len(selected):
                selected = error[allowed]
                fallback = True
            if not len(selected):
                raise ValueError("No independent calibration residuals at horizon %s" % h)
            key = f"{h}_{day}"
            table[key] = np.quantile(selected, quantiles).tolist()
            metrics[key] = dict(n=len(selected), mean=float(selected.mean()),
                                rmse=float(np.mean(selected**2)**.5),
                                std=float(selected.std()), pooled_daylight_fallback=fallback)
    return table, metrics


def train_bundle(data, train_mask, out_dir, *, calibration_mask=None, threads=2):
    """Write all fixed architecture artifacts; return the output directory Path.

    ``calibration_mask``, if supplied, is a subset of outer training data used
    only by separate inner models to create fresh residual scenario tables.
    Outer models are refit on all ``train_mask`` rows at fixed counts. No
    evaluation early stopping or evaluation-label-dependent choice occurs.
    """
    started = time.perf_counter()
    data = data.reset_index(drop=True)
    train_mask = np.asarray(train_mask, bool)
    if train_mask.shape != (len(data),):
        raise ValueError("Mask must have one boolean per hourly row")
    if not train_mask.any():
        raise ValueError("Empty training set")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    module = module_at(ROOT/"mode_dificil/agent.py", "clean_training_features")
    original = json.loads((ROOT/"hard_work/original/prior.json").read_text())
    inner_cal = (choose_inner_calibration(data, train_mask) if calibration_mask is None
                 else np.asarray(calibration_mask, bool))
    if inner_cal.shape != train_mask.shape or np.any(inner_cal & ~train_mask):
        raise ValueError("Calibration must be a training-only subset")
    inner_train = train_mask & ~inner_cal
    print(f"Training {out_dir.name}: outer={train_mask.sum()}, inner residual calibration={inner_cal.sum()}", flush=True)
    inner_prior, inner_features = make_features(data, inner_train, original, module)
    tables, residual_metrics = {}, {}
    inner_training = {}
    for kind in ("direct", "residual"):
        model, stats = fit_model(kind, inner_features, inner_train, threads)
        tables[kind], residual_metrics[kind] = residual_table(inner_features, model, inner_cal, kind == "residual")
        inner_training[kind] = stats
    del inner_features, inner_prior, model
    prior, features = make_features(data, train_mask, original, module)
    stats = {}
    for kind in ("direct", "residual", "price", "co2"):
        model, stats[kind] = fit_model(kind, features, train_mask, threads)
        model.save_model(str(out_dir/f"model_{kind}.txt"))
        print(f"Saved {kind}: {stats[kind]['rows']} rows, {stats[kind]['trees']} trees", flush=True)
    x0 = features["Xn"][:, 0, :]
    y0 = features["load"]-features["pv"]-net_baseline(x0)
    for j, (q, rounds) in enumerate(zip((np.arange(7)+.5)/7, QUANTILE_ROUNDS)):
        params = dict(objective="quantile", alpha=float(q), metric="quantile", verbosity=-1,
                      num_threads=threads, num_leaves=31, min_data_in_leaf=50,
                      learning_rate=.05, lambda_l2=5, feature_fraction=.9,
                      bagging_fraction=.9, bagging_freq=1, seed=83)
        model = lgb.train(params, lgb.Dataset(x0[train_mask], label=y0[train_mask]), num_boost_round=rounds)
        model.save_model(str(out_dir/f"net_q{j}.txt"))
        stats[f"q{j}"] = dict(rows=int(train_mask.sum()), quantile=float(q), trees=model.num_trees(), recipe=params)
    analog = build_library(features, exclude=~train_mask)
    np.savez_compressed(out_dir/"analog_library.npz", **analog)
    prior["qtab_original"] = original["qtab"]
    prior["qtab_direct"] = tables["direct"]
    prior["qtab_residual"] = tables["residual"]
    prior["qtab"] = tables["direct"]
    prior["version"] = "clean_fixed_recipes_v1"
    prior.pop("_cache", None)
    (out_dir/"prior.json").write_text(json.dumps(prior))
    metadata = dict(
        method="Fixed official architecture recipes, full-horizon purging, train-only profiles, independent inner residual calibration; no evaluation early stopping",
        data_hours=len(data), training_hours=int(train_mask.sum()),
        training_spans=span_records(data, train_mask),
        excluded_spans=span_records(data, ~train_mask),
        inner_calibration_spans=span_records(data, inner_cal),
        inner_training=inner_training, models=stats,
        analog_trajectories=int(len(analog["origins"])),
        residual_metrics=residual_metrics,
        original_residual_table="Retained as qtab_original for historical diagnostics only; qtab defaults to fresh direct residuals",
        solar_scaling="Input pv_kw already hard-mode x12; this trainer performs no additional scaling",
        clear_sky="Deterministic astronomical table retained from original prior; no observed target values",
        lightgbm_version=lgb.__version__, numpy_version=np.__version__, pandas_version=pd.__version__,
        duration_s=time.perf_counter()-started,
        hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out_dir.iterdir()) if p.suffix in (".txt", ".json", ".npz") and p.name != "training_metadata.json"},
    )
    (out_dir/"training_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Bundle complete: {out_dir} in {metadata['duration_s']:.1f}s", flush=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-before", help="Exclusive timestamp boundary; omit for final full-data refit")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    data = load_data(ROOT)
    mask = np.ones(len(data), bool) if args.train_before is None else (data.Fecha < pd.Timestamp(args.train_before)).to_numpy()
    train_bundle(data, mask, args.out, threads=args.threads)


if __name__ == "__main__":
    main()
