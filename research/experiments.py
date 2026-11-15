"""Complete-agent temporal experiments; all results retained, no server access."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

from build_agent import build, DEFAULT
from evaluate import run, load_data

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'research'


def aggregate(rows):
    keys = ('co2_baseline_kg','co2_participant_kg','cost_baseline_eur','cost_participant_eur','violations','errors','hours')
    out = {k:sum(r[k] for r in rows) for k in keys}
    out['co2_saved_pct'] = 100*(1-out['co2_participant_kg']/out['co2_baseline_kg'])
    out['cost_saved_pct'] = 100*(1-out['cost_participant_eur']/out['cost_baseline_eur'])
    out['score_unclipped'] = .7*out['co2_saved_pct']+.3*out['cost_saved_pct']-.2*out['violations']
    out['score'] = max(0,out['score_unclipped'])
    out['runtime_s'] = sum(r['runtime']['total_seconds'] for r in rows)
    out['soc_final_kwh'] = [r['soc_final_kwh'] for r in rows]
    return out


def factorial():
    configs = {}
    for market in (0,1):
        for net in (0,1):
            for quantiles in (False,True):
                name = ('analog' if market else 'lgb')+'_'+('residual' if net else 'direct')+'_'+('quantiles' if quantiles else 'empirical')
                configs[name] = dict(market_analog_weight=market,net_residual_weight=net,current_quantiles=quantiles)
    return configs


def evaluate_configs(configs, output, *, folds=None, options=None, data=None):
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((HERE/'split_manifest.json').read_text())
    folds = folds or manifest['development']
    data = load_data(cache=HERE/'data_cache.pkl') if data is None else data
    result = json.loads(output.read_text()) if output.exists() else dict(method='Complete agents, forward chronological train-only refits, aggregate totals; separate 3000kWh starts per development fold.',results={})
    provenance = {'scripts':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('build_agent.py','evaluate.py','experiments.py')},
                  'bundles':{f['name']:hashlib.sha256((HERE/'folds'/f['name']/'training_metadata.json').read_bytes()).hexdigest() for f in folds},
                  'options':options or {},'folds':folds,'source_fingerprint':data.attrs.get('source_fingerprint')}
    for name,config in configs.items():
        if name in result['results']:
            prior = result['results'][name]
            if prior.get('provenance') != provenance or prior['config'] != {**DEFAULT,**config}:
                raise ValueError('Existing result has different or missing provenance; use a new output path or --fresh: '+name)
            continue
        records = []
        for fold in folds:
            c = {**DEFAULT,**config}
            destination = HERE/'agents'/fold['name']/name
            build(HERE/'folds'/fold['name'],destination,c)
            selected = data.loc[(data.Fecha>=fold['start']) & (data.Fecha<fold['end'])].copy()
            r = run(selected,destination,label=name,**(options or {}))
            r['fold'] = fold['name']
            r['policy_config'] = c
            records.append(r)
            print(name,fold['name'],round(r['score'],5),'errors',r['errors'],'time',round(r['runtime']['total_seconds'],2),flush=True)
        result['results'][name] = dict(config={**DEFAULT,**config},aggregate=aggregate(records),folds=records,provenance=provenance)
        output.write_text(json.dumps(result,indent=2))
        print('AGGREGATE',name,round(result['results'][name]['aggregate']['score'],5),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs',type=Path,help='JSON mapping of names to configuration')
    p.add_argument('--out',type=Path,default=HERE/'results/factorial.json')
    p.add_argument('--no-exception-audit',action='store_true')
    p.add_argument('--fresh',action='store_true',help='Replace this output JSON and rerun every requested configuration')
    a=p.parse_args()
    configs=json.loads(a.configs.read_text()) if a.configs else factorial()
    if a.fresh:
        a.out.unlink(missing_ok=True)
    evaluate_configs(configs,a.out,options=dict(audit_exceptions=not a.no_exception_audit))


if __name__=='__main__':
    main()
