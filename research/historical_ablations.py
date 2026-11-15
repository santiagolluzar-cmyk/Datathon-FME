"""Diagnostic factorial on previously exposed historical holdout.

These results MUST NOT select future final-holdout variants. Historical early
stopping and shared residual tables contaminate independence. No web actions.
"""
from pathlib import Path
import json,sys,shutil
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'research'),str(ROOT/'hard_work'),str(ROOT)]
from build_agent import build,DEFAULT
from evaluate import run,load_data
from training import validation_mask
HERE=ROOT/'research'

def main():
 out=HERE/'results/historical_ablations.json';out.parent.mkdir(parents=True,exist_ok=True)
 bundle=HERE/'historical_bundle';bundle.mkdir(exist_ok=True)
 old=ROOT/'hard_work/oof_baseline';new=ROOT/'hard_work/oof_delivery'
 for src,name in [(old/'model_net.txt','model_direct.txt'),(new/'model_net.txt','model_residual.txt'),(old/'model_price.txt','model_price.txt'),(old/'model_co2.txt','model_co2.txt'),(new/'analog_library.npz','analog_library.npz')]+[(new/f'net_q{j}.txt',f'net_q{j}.txt') for j in range(7)]:shutil.copy2(src,bundle/name)
 prior=json.loads((old/'prior.json').read_text())
 for key in ['qtab_original','qtab_direct','qtab_residual']:prior[key]=prior['qtab']
 (bundle/'prior.json').write_text(json.dumps(prior))
 data=load_data(cache=HERE/'data_cache.pkl');mask,weeks=validation_mask(data);selected=data.loc[mask].reset_index(drop=True)
 result={'warning':'Diagnostic only; historical holdout already exposed to earlier training-selection and shared residual tables. Never use for final-holdout selection.', 'method':'Complete delivered agents, strict swallowed-exception monitoring, 3 market hours, SOC carries across timestamp gaps and History resets. Aggregate totals, no terminal credit.', 'validation_weeks':weeks,'results':{},'corner_parity':{}}
 for market in [0,1]:
  for net in [0,1]:
   for quantiles in [False,True]:
    name=('analog' if market else 'lgb')+'_'+('residual' if net else 'direct')+'_'+('quantiles' if quantiles else 'empirical')
    config={**DEFAULT,'market_analog_weight':market,'net_residual_weight':net,'current_quantiles':quantiles,'residual_scale':2.0,'residual_source':'original','analog_k':4,'ratio_co2_cost':.6345}
    dest=HERE/'agents/historical'/name;build(bundle,dest,config)
    trace=HERE/f'results/historical_{name}.csv';r=run(selected,dest,label=name,trace_path=trace,strict=True,audit_exceptions=True)
    result['results'][name]={'policy_config':config,**r};out.write_text(json.dumps(result,indent=2))
    print(name,r['score'],'CO2%',r['co2_saved_pct'],'cost%',r['cost_saved_pct'],'errors',r['errors'],'violations',r['violations'],flush=True)
 for name,directory,corner in [('baseline',old,'lgb_direct_empirical'),('prototype',new,'analog_residual_quantiles')]:
  trace=HERE/f'results/historical_source_{name}.csv';r=run(selected,directory,label='source_'+name,trace_path=trace,strict=True,audit_exceptions=True)
  a=pd.read_csv(trace);b=pd.read_csv(HERE/f'results/historical_{corner}.csv')
  result['corner_parity'][name]={'source_result':r,'corner':corner,'score_delta':result['results'][corner]['score']-r['score'],'max_abs_action_delta_kw':float(np.max(abs(a.action_kw-b.action_kw))),'max_abs_soc_delta_kwh':float(np.max(abs(a.soc_after_kwh-b.soc_after_kwh))),'max_abs_grid_delta_kwh':float(np.max(abs(a.grid_kwh-b.grid_kwh)))}
  out.write_text(json.dumps(result,indent=2));print('PARITY',name,result['corner_parity'][name]['score_delta'],result['corner_parity'][name]['max_abs_action_delta_kw'],flush=True)
 print('Saved',out,flush=True)
if __name__=='__main__':main()
