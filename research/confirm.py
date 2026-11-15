"""One-shot check of a frozen policy on the declared last 1008 public hours.

Never edits selection.json. Refuses to overwrite a completed result. This is a
public-data audit, not a server score or a historically pristine hidden test.
"""
import argparse
import hashlib
import json
from pathlib import Path

from build_agent import build
from evaluate import run,load_data

HERE=Path(__file__).resolve().parent


def confirm(out):
    out=Path(out)
    if out.exists():
        raise FileExistsError('Confirmation already exists; retain it. For deliberate reproduction use a new --out path.')
    selection=json.loads((HERE/'selection.json').read_text())
    manifest=json.loads((HERE/'split_manifest.json').read_text())
    final=manifest['locked_final']
    data=load_data(cache=HERE/'data_cache.pkl')
    data=data.loc[(data.Fecha>=final['start']) & (data.Fecha<final['end'])].copy()
    assert len(data)==1008
    configs={
        'baseline_clean':dict(market_analog_weight=0,net_residual_weight=0,current_quantiles=False),
        'prototype_clean':dict(market_analog_weight=1,analog_k=4,net_residual_weight=1,current_quantiles=True),
        'selected':selection['config'],
    }
    result=dict(method='Predeclared one-shot final current-search check, identical training cutoff for all agents; no policy reselection using these outcomes.',
                limitation=final['independence_limit'],selection=selection,
                selection_sha256=hashlib.sha256((HERE/'selection.json').read_bytes()).hexdigest(),
                periods=final,results={})
    for name,c in configs.items():
        directory=build(HERE/'folds/locked',HERE/'agents/locked'/name,c)
        result['results'][name]={}
        for gap in ('carry','reset'):
            row=run(data,directory,gap_soc=gap,label=name,
                    trace_path=out.parent/f'locked_{name}_{gap}.csv')
            result['results'][name][gap]=row
            out.write_text(json.dumps(result,indent=2))
            print(name,gap,'score',round(row['score'],6),'CO2',round(row['co2_saved_pct'],4),
                  'cost',round(row['cost_saved_pct'],4),'errors',row['errors'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'results/locked_final.json')
    a=p.parse_args();confirm(a.out)
