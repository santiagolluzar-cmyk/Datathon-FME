"""Static/numeric training audit. Read-only original inputs; local JSON output."""
from pathlib import Path
import sys,json,re,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hard_work'))
from training import load_data,validation_mask,calibration_masks,module_at,rebuild_prior

def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def run():
 data=load_data(ROOT);ho,weeks=validation_mask(data);cal,con=calibration_masks(data,ho)
 report={'hours':len(data),'holdout_hours':int(ho.sum()),'calibration_hours':int(cal.sum()),'confirmation_hours':int(con.sum()),'weeks':weeks,'models':{},'priors':{}}
 for directory in ['hard_work/original','mode_dificil','hard_work/oof_baseline','hard_work/oof_delivery']:
  folder=ROOT/directory;prior=json.loads((folder/'prior.json').read_text())
  report['priors'][directory]={'qtab_hash':canonical(prior['qtab']),'version':prior['version'],'rk_train':prior['rk_train'],'rc_train':prior['rc_train']}
  for file in folder.glob('*.txt'):
   txt=file.read_text();params=dict(re.findall(r'^\[([^:]+): (.*?)\]$',txt,flags=re.M))
   report['models'][directory+'/'+file.name]={'trees':txt.count('Tree='),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'params':{key:params.get(key) for key in ['objective','num_iterations','learning_rate','num_leaves','min_data_in_leaf','lambda_l2','feature_fraction','bagging_fraction','seed']}}
 old=json.loads((ROOT/'hard_work/original/prior.json').read_text());new=json.loads((ROOT/'hard_work/oof_baseline/prior.json').read_text())
 report['profile_shift']={}
 for key in ['load','pv','price','co2','irr']:
  cells=set(old[key]['mth'])&set(new[key]['mth']);a=np.array([old[key]['mth'][c] for c in cells]);b=np.array([new[key]['mth'][c] for c in cells]);d=a-b
  report['profile_shift'][key]={'cells':len(cells),'rmse':float(np.mean(d*d)**.5),'max_abs':float(abs(d).max()),'mean_abs':float(abs(d).mean())}
 # Explicitly count training origins that could consume heldout lag features when
 # build_features(data,module,prior) is called without split_mask in training.main.
 ts=data.Fecha.to_numpy();n=len(data);cont=np.r_[False,np.diff(ts)==np.timedelta64(1,'h')]
 contaminated=np.zeros(n,dtype=bool)
 for lag in range(1,49):
  for i in range(lag,n):
   if not ho[i] and ho[i-lag] and cont[i-lag+1:i+1].all(): contaminated[i]=True
 report['warm_feature_holdout_overlap']={'train_origins_with_heldout_observations_in_previous48h':int(contaminated.sum()),'timestamps':data.Fecha[contaminated].astype(str).tolist()}
 # Verify excluded-data profiles from provided temporal prior exactly correspond
 # to current loader; original cs/qtab are deliberately retained by rebuild_prior.
 module=module_at(ROOT/'mode_dificil/agent.py','training_audit_module');rebuilt=rebuild_prior(data,~ho,module,old)
 report['temporal_profile_reconstruction']={}
 for key in ['load','pv','price','co2','irr']:
  cells=set(rebuilt[key]['mth'])&set(new[key]['mth']);d=[abs(rebuilt[key]['mth'][c]-new[key]['mth'][c]) for c in cells]
  report['temporal_profile_reconstruction'][key]={'max_abs':max(d),'keys_equal':set(rebuilt[key]['mth'])==set(new[key]['mth'])}
 out=ROOT/'research/training_audit.json';out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['models','warm_feature_holdout_overlap']},indent=2));print('Lag leakage origins:',contaminated.sum())
if __name__=='__main__':run()
