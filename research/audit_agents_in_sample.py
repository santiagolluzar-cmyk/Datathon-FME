"""Exact ZIP reproduction and IN-SAMPLE diagnostic, never a test-score claim."""
from pathlib import Path
from zipfile import ZipFile
import sys,json,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'hard_work')]
from training import load_data,module_at,build_features,validation_mask,calibration_masks
import eval_dificil
OUT=ROOT/'research'

def main():
 result={'warning':'All published-train results for official ZIPs are IN-SAMPLE; not unbiased generalization estimates. No server access.','exact_packages':{},'reproduction':{}}
 data=load_data(ROOT)
 for label,path in [('baseline',ROOT/'hard_work/original'),('prototype',ROOT/'mode_dificil')]:
  dest=OUT/f'audit_agents_exact_{label}';dest.mkdir(exist_ok=True)
  with ZipFile(path/'submission.zip') as z:z.extractall(dest)
  result['exact_packages'][label]={'zip_sha256':hashlib.sha256((path/'submission.zip').read_bytes()).hexdigest()}
  for suite,frame in [('all_published_train',data),('historical_selected_weeks',eval_dificil.seleccionar(data,'setmanes'))]:
   r=eval_dificil.simular(frame,dest,ROOT)
   result['exact_packages'][label][suite]=r
   print(label,suite,r['score'],r['temps_s'],flush=True)
   (OUT/'audit_agents_in_sample.json').write_text(json.dumps(result,indent=2))
 # Reproduce the preserved out-of-fold complete-agent evaluation, without refitting or overwriting historical evidence.
 holdout,_=validation_mask(data);cal,con=calibration_masks(data,holdout)
 for label,path in [('baseline',ROOT/'hard_work/oof_baseline'),('prototype',ROOT/'hard_work/oof_delivery')]:
  result['reproduction'][label]={}
  for suite,mask in [('calibration',cal),('confirmation',con),('all',holdout)]:
   r=eval_dificil.simular(data.loc[mask].reset_index(drop=True),path,ROOT)
   result['reproduction'][label][suite]=r
   print('reproduction',label,suite,r['score'],r['temps_s'],flush=True)
   (OUT/'audit_agents_in_sample.json').write_text(json.dumps(result,indent=2))
 # Forecast-only diagnostic on the exact published data used for full training.
 old=module_at(OUT/'audit_agents_exact_baseline/agent.py','old_audit').BatteryAgent()
 new=module_at(OUT/'audit_agents_exact_prototype/agent.py','new_audit').BatteryAgent()
 module=module_at(OUT/'audit_agents_exact_baseline/agent.py','features_audit')
 f=build_features(data,module,old.prior)
 n=len(data);xn=f['Xn'];xm=f['Xm'];origin=np.arange(n)[:,None];lead=np.arange(24)[None,:];target=np.minimum(origin+lead,n-1)
 safe=(origin+lead<n)&(f['block_id'][origin]==f['block_id'][target])
 base=np.where(np.isfinite(xn[:,:,25]),xn[:,:,25],xn[:,:,6])-np.where(np.isfinite(xn[:,:,23]),xn[:,:,23],xn[:,:,7])
 pred={'baseline':{},'prototype':{}}
 pred['baseline']['net']=old.models['net'].predict(xn.reshape(-1,28),num_threads=1).reshape(n,24)
 pred['prototype']['net']=base+new.models['net'].predict(xn.reshape(-1,28),num_threads=1).reshape(n,24)
 for name,ix in [('price',27),('co2',28)]:pred['baseline'][name]=xm[:,:,ix]+old.models[name].predict(xm.reshape(-1,44),num_threads=1).reshape(n,24)
 an=np.array([new.market.predict(x) for x in xm]);pred['prototype']['price']=an[:,0,:];pred['prototype']['co2']=an[:,1,:]
 truths={'net':f['load']-f['pv'],'price':f['price'],'co2':f['carbon']}
 stats={}
 for name in pred:
  stats[name]={}
  for key in truths:
   stats[name][key]={}
   for lo,hi in ([(0,1),(1,3),(3,12),(12,24)] if key=='net' else [(3,12),(12,24),(3,24)]):
    mask=safe&(lead>=lo)&(lead<hi);err=(pred[name][key]-truths[key][target])[mask]
    stats[name][key][f'h{lo}-{hi-1}']={'n':int(mask.sum()),'rmse':float(np.sqrt(np.mean(err**2))),'mae':float(np.mean(abs(err))),'bias':float(np.mean(err))}
   if key!='net':
    known=np.isfinite(xm[:,:,8]) # overwrite current + next two only for negative-weight diagnostics
    for j in range(3):pred[name][key][:,j]=np.where(safe[:,j],truths[key][target[:,j]],pred[name][key][:,j])
  w=.7*pred[name]['co2']/.6345+.3*pred[name]['price']
  stats[name]['negative_weight_forecast_hours']=int((w<0).sum())
  stats[name]['min_weight_forecast']=float(w.min())
 # Exact source origin self-neighbor exposure in the full-data analog library.
 nf=module_at(OUT/'audit_agents_exact_prototype/agent.py','newfeat_audit').market_features(xm)
 lib=new.market.library;nf=np.where(np.isfinite(nf),nf,lib['median'])
 self_rows=[]
 for j,i in enumerate(lib['origins']):
  pool=new.market.groups[int(lib['hour'][j])];d=np.sum((lib['x'][pool]-nf[int(i)])**2,axis=1);sel=np.argpartition(d,min(4,len(pool))-1)[:min(4,len(pool))];nn=pool[sel];weights=1/(.2+d[sel]);weights/=weights.sum()
  matches=np.flatnonzero(nn==j)
  self_rows.append((float(d[np.flatnonzero(pool==j)[0]]),float(weights[matches[0]]) if len(matches) else 0))
 result['in_sample_forecast_metrics']=stats
 result['analog_self_neighbor_diagnostic']={'source_origins':len(self_rows),'self_in_top4_fraction':float(np.mean(np.array(self_rows)[:,1]>0)),'mean_self_weight':float(np.mean(np.array(self_rows)[:,1])),'max_self_feature_distance_sq':float(np.max(np.array(self_rows)[:,0]))}
 (OUT/'audit_agents_in_sample.json').write_text(json.dumps(result,indent=2));print(json.dumps({'forecast_metrics':stats,'analog_self_neighbor':result['analog_self_neighbor_diagnostic']},indent=2),flush=True)
if __name__=='__main__':main()
