"""Refit selected demand forecasts on all published training data.

The selected number of trees is fixed by calibration experiments. This script
does not read hidden-test data, upload solutions, or alter the source repository.
"""
import json
import hashlib
import numpy as np
import lightgbm as lgb
from training import REPO, OUT, module_at, load_data, build_features


def net_baseline(X):
    """Reconstruct the physical baseline for an array ending in 28 features."""
    return np.where(np.isfinite(X[...,25]),X[...,25],X[...,6])-np.where(
        np.isfinite(X[...,23]),X[...,23],X[...,7])


def main():
    target=OUT/'final_models';target.mkdir(exist_ok=True)
    module=module_at(REPO/'mode_dificil/agent.py')
    prior=json.loads((REPO/'mode_dificil/prior.json').read_text());prior['_cache']={}
    data=load_data(REPO)
    features=build_features(data,module,prior)
    n=len(data);X=features['Xn'].reshape(-1,28)
    origins=np.repeat(np.arange(n),24);leads=np.tile(np.arange(24),n)
    future=np.minimum(origins+leads,n-1)
    safe=(origins+leads<n)&(features['block_id'][origins]==features['block_id'][future])
    actual=features['load']-features['pv']
    baseline=net_baseline(X)
    y=actual[future]-baseline
    records=json.loads((OUT/'candidates/scores.json').read_text())
    selected=next(r for r in records if r['name']=='residual_regression_15_500')
    rounds=selected['iterations']
    params=dict(objective='regression',metric='l2',verbosity=-1,num_threads=4,
                learning_rate=.035,num_leaves=15,min_data_in_leaf=500,
                feature_fraction=.85,bagging_fraction=.85,bagging_freq=1,lambda_l2=5,seed=83)
    model=lgb.train(params,lgb.Dataset(X[safe],label=y[safe]),num_boost_round=rounds)
    model.save_model(str(target/'model_net.txt'))
    pred=model.predict(X,num_threads=1).reshape(n,24)+baseline.reshape(n,24)
    print('Final net',int(safe.sum()),'rows',rounds,'trees',flush=True)
    qmetadata=json.loads((OUT/'current_quantiles/metadata_leaves31.json').read_text())
    X0=features['Xn'][:,0,:];base0=net_baseline(X0);y0=actual-base0
    scenes=[]
    for j,row in enumerate(qmetadata['models']):
        params=dict(objective='quantile',alpha=row['quantile'],metric='quantile',verbosity=-1,
                    num_threads=4,num_leaves=31,min_data_in_leaf=50,learning_rate=.05,
                    lambda_l2=5,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,seed=83)
        model=lgb.train(params,lgb.Dataset(X0,label=y0),num_boost_round=row['iterations'])
        model.save_model(str(target/f'net_q{j}.txt'))
        scenes.append(base0+model.predict(X0,num_threads=1))
        print('Final quantile',j,row['quantile'],row['iterations'],'trees',flush=True)
    scenes=np.sort(np.asarray(scenes).T,axis=1)
    np.savez_compressed(target/'forecast_cache.npz',net=pred,current_scenarios=scenes)
    prior.pop('_cache',None)
    (target/'prior.json').write_text(json.dumps(prior))
    metadata={'training_hours':n,'net_training_rows':int(safe.sum()),
              'net':{'filename':'model_net.txt','objective':'residual_regression',
                     'num_leaves':15,'min_data_in_leaf':500,'iterations':rounds},
              'quantiles':[{**r,'filename':f'net_q{j}.txt'} for j,r in enumerate(qmetadata['models'])],
              'baseline':'where(isfinite(X[...,25]),X[...,25],X[...,6]) - where(isfinite(X[...,23]),X[...,23],X[...,7])',
              'forecast_net':'net_baseline(Xn) + model_net.predict(Xn)',
              'forecast_current_scenarios':'sort(net_baseline(Xn[0]) + [net_qj.predict(Xn[0]) for j in range(7)])',
              'calibration':'Selected tree counts fixed before full-data fit; no hidden-test labels used',
              'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.glob('*.txt')}}
    (target/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Final full-data demand models ready',flush=True)


if __name__=='__main__':main()
