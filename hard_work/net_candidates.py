"""Net forecast experiments with cold-start heldout weeks."""
import json
import time
import numpy as np
import lightgbm as lgb
from training import *


def run():
    data=load_data();module=module_at(REPO/'mode_dificil/agent.py')
    original=json.loads((REPO/'mode_dificil/prior.json').read_text());original['_cache']={}
    holdout,weeks=validation_mask(data);cal,confirm=calibration_masks(data,holdout)
    coldold=build_features(data,module,original,holdout)
    oldmodels={k:lgb.Booster(model_str=(REPO/f'mode_dificil/model_{k}.txt').read_text()) for k in ('net','price','co2')}
    coldcache=forecast_cache(coldold,oldmodels)
    coldcache.update(validation=holdout,calibration=cal,confirmation=confirm)
    np.savez_compressed(OUT/'cache_cold.npz',**coldcache)
    prior=json.loads((OUT/'temporal/prior.json').read_text());prior['_cache']={}
    features=build_features(data,module,prior,holdout)
    np.savez_compressed(OUT/'temporal/features_cold.npz',**features)
    models={k:lgb.Booster(model_str=(OUT/f'temporal/model_{k}.txt').read_text()) for k in ('net','price','co2')}
    cache=forecast_cache(features,models)
    cache.update(validation=holdout,calibration=cal,confirmation=confirm)
    np.savez_compressed(OUT/'temporal/cache_cold.npz',**cache)
    print('Cold caches ready',flush=True)
    n=len(data);o=np.repeat(np.arange(n),24);h=np.tile(np.arange(24),n);t=np.minimum(o+h,n-1)
    safe=(o+h<n)&(features['block_id'][o]==features['block_id'][t])
    train=safe & ~holdout[o] & ~holdout[t]
    valid=safe & cal[o] & cal[t]
    test=safe & confirm[o] & confirm[t]
    X=features['Xn'].reshape(n*24,-1)
    y=(features['load']-features['pv'])[t]
    base=np.where(np.isfinite(X[:,25]),X[:,25],X[:,6])-np.where(np.isfinite(X[:,23]),X[:,23],X[:,7])
    out=OUT/'candidates';out.mkdir(exist_ok=True)
    scores=[]
    for target in ('residual','direct'):
        for objective in ('regression','regression_l1','huber'):
            for leaves,minimum in ((31,100),(15,500),(63,100)):
                name=f'{target}_{objective}_{leaves}_{minimum}'
                yy=y-base if target=='residual' else y
                params=dict(objective=objective,metric='l2',verbosity=-1,num_threads=4,
                            learning_rate=.035,num_leaves=leaves,min_data_in_leaf=minimum,
                            feature_fraction=.85,bagging_fraction=.85,bagging_freq=1,lambda_l2=5,seed=83)
                if objective=='huber':
                    params['alpha']=.9
                model=lgb.train(params,lgb.Dataset(X[train],label=yy[train]),num_boost_round=600,
                                valid_sets=[lgb.Dataset(X[valid],label=yy[valid])],
                                callbacks=[lgb.early_stopping(60,verbose=False)])
                p=model.predict(X,num_threads=1)+(base if target=='residual' else 0)
                row={'name':name,'iterations':model.best_iteration,
                     'cal_rmse':float(np.mean((p[valid]-y[valid])**2)**.5),
                     'confirmation_rmse':float(np.mean((p[test]-y[test])**2)**.5),
                     'confirmation_h0_rmse':float(np.mean((p[test&(h==0)]-y[test&(h==0)])**2)**.5)}
                model.save_model(str(out/f'{name}.txt'))
                # Save only replacement matrix and coldstart grouping; parent uses existing market cache.
                np.savez_compressed(out/f'{name}.npz',net=p.reshape(n,24))
                scores.append(row);(out/'scores.json').write_text(json.dumps(scores,indent=2))
                print(row,flush=True)
    print('All candidates ready',flush=True)


if __name__=='__main__':
    run()
