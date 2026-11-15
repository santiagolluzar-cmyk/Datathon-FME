"""Calibration diagnostics of preserved current-hour uncertainty models."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'hard_work')]
from training import load_data,build_features,module_at,validation_mask,calibration_masks
D=load_data(ROOT);holdout,_=validation_mask(D);cal,con=calibration_masks(D,holdout)
result={}
for label,path,df in [('full_in_sample',ROOT/'mode_dificil',D),('historical_oof',ROOT/'hard_work/oof_delivery',D.loc[holdout].reset_index(drop=True))]:
 mod=module_at(path/'agent.py',f'q_{label}');agent=mod.BatteryAgent();f=build_features(df,mod,agent.prior);X=f['Xn'][:,0,:]
 base=np.where(np.isfinite(X[:,25]),X[:,25],X[:,6])-np.where(np.isfinite(X[:,23]),X[:,23],X[:,7]);y=f['load']-f['pv']
 raw=np.array([base+agent.models[f'q{j}'].predict(X,num_threads=1) for j in range(7)]).T;qs=np.sort(raw,axis=1)
 pred=base+agent.models['net'].predict(X,num_threads=1)
 known=f['night'][:,0]&np.isfinite(f['lastload']);evalmask=~known
 rows={}
 for title,mask in [('all_uncertain_hours',evalmask),('daylight',evalmask&(X[:,8]>50)),('dark',evalmask&(X[:,8]<=50))]:
  if not mask.any():continue
  a=qs[mask];b=y[mask]
  rows[title]={'n':int(mask.sum()),'coverage_q071_to_q929':float(np.mean((b>=a[:,0])&(b<=a[:,-1]))),'below_q071':float(np.mean(b<a[:,0])),'above_q929':float(np.mean(b>a[:,-1])),'mean_width_kw':float(np.mean(a[:,-1]-a[:,0])),'quantile_crossing_fraction_before_sort':float(np.mean(np.any(np.diff(raw[mask],axis=1)<0,axis=1))),'median_mae_kw':float(np.mean(abs(a[:,3]-b))),'net_model_mae_kw':float(np.mean(abs(pred[mask]-b)))}
 result[label]=rows
(ROOT/'research/audit_agents_quantiles.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
