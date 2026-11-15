"""Train a clean seasonal blocked OOF bundle, excluding the locked final set."""
import json
from pathlib import Path

import pandas as pd

from train_clean import ROOT, load_data, train_bundle, span_records
from training import validation_mask


def main():
    data = load_data(ROOT)
    split = json.loads((ROOT / "research/split_manifest.json").read_text())
    cutoff = pd.Timestamp(split["locked_final"]["start"])
    before_final = (data.Fecha < cutoff).to_numpy()
    historical, _ = validation_mask(data)
    evaluate = before_final & historical
    train = before_final & ~historical
    assert int(evaluate.sum()) == 1512, "Expected nine complete seasonal weeks"
    assert not (train & evaluate).any()
    assert not (train & ~before_final).any()
    destination = ROOT / "research/folds/blocked"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        name="blocked_seasonal_development",
        method="Seasonal blocked OOF development; future-within-pre-final training is allowed, unlike forward folds",
        locked_final_start=str(cutoff),
        locked_final_hours_used=0,
        training_hours=int(train.sum()),
        evaluation_hours=int(evaluate.sum()),
        evaluation_weeks=[str(w) for w in sorted(data.Fecha[evaluate].dt.to_period("W-SUN").unique())],
        training_spans=span_records(data, train),
        evaluation_spans=span_records(data, evaluate),
        train_indices=[int(i) for i in data.index[train]],
        evaluation_indices=[int(i) for i in data.index[evaluate]],
        role="Development robustness only; not untouched final evidence",
        original_residual_table_used=False,
    )
    (destination / "blocked_split.json").write_text(json.dumps(manifest, indent=2))
    train_bundle(data, train, destination, threads=2)


if __name__ == "__main__":
    main()
