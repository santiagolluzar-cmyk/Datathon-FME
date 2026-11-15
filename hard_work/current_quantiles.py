"""Conditional current-hour residual quantiles, selected without confirmation tuning."""
import json
import numpy as np
import lightgbm as lgb
from training import *
import control
from sweep import subset


def main():
    root=OUT
    features=dict(np.load(root/'temporal/features_cold.npz'))
    cache=dict(np.load(root/'temporal/cache_cold.npz'))
    cal,confirm=cache['calibration'],cache['confirmation']
    train=~cache['validation']
    X=features['Xn'][:,0,:]
    base=np.where(np.isfinite(X[:,25]),X[:,25],X[:,6])-np.where(np.isfinite(X[:,23]),X[:,23],X[:,7])
    y=cache['load']-cache['pv']-base
    modeldir=root/'current_quantiles';modeldir.mkdir(exist_ok=True)
    results=[]
    for leaves in (31,15):
        scenarios=[];metadata=[]
        for j,q in enumerate((np.arange(7)+.5)/7):
            params=dict(objective='quantile',alpha=float(q),metric='quantile',verbosity=-1,
                        num_threads=4,num_leaves=leaves,min_data_in_leaf=50,
                        learning_rate=.05,lambda_l2=5,feature_fraction=.9,bagging_fraction=.9,
                        bagging_freq=1,seed=83)
            model=lgb.train(params,lgb.Dataset(X[train],label=y[train]),num_boost_round=400,
                            valid_sets=[lgb.Dataset(X[cal],label=y[cal])],
                            callbacks=[lgb.early_stopping(60,verbose=False)])
            pred=base+model.predict(X,num_threads=1)
            scenarios.append(pred)
            model.save_model(str(modeldir/f'net_q{j}_leaves{leaves}.txt'))
            metadata.append({'quantile':float(q),'iterations':model.best_iteration})
        scenarios=np.sort(np.asarray(scenarios).T,axis=1)
        np.savez_compressed(modeldir/f'scenarios_leaves{leaves}.npz',current_scenarios=scenarios)
        (modeldir/f'metadata_leaves{leaves}.json').write_text(json.dumps({'models':metadata,
            'baseline':'finite Xn[:,25] else Xn[:,6], minus finite Xn[:,23] else Xn[:,7]',
            'features':'same28 features as current net model, horizon0 only'},indent=2))
        print('Saved current scenarios',leaves,flush=True)
        mod=module_at(root/'original/agent.py')
        net=np.load(root/'candidates/residual_regression_15_500.npz')['net']
        ca=subset({**cache,'net':net},cal)
        prep=control.prepare(ca,mod,{'current_scale':1})
        for i in range(len(ca['load'])):
            if not prep['current_known'][i]:
                prep['current'][i]=scenarios[cal][i].tolist()
        score=control.replay(ca,mod,prep)['score']
        results.append({'name':f'quantiles_leaves{leaves}','calibration':score})
        print('Calibration controller',leaves,score,flush=True)
    huber=np.load(root/'candidates/residual_huber_63_100.npz')['net']
    net=np.load(root/'candidates/residual_regression_15_500.npz')['net'].copy()
    net[:,0]=huber[:,0]
    ca=subset({**cache,'net':net},cal)
    score=control.run(ca,module_at(root/'original/agent.py'),{'current_scale':1})['score']
    results.append({'name':'hybrid_huber0_regression_future','calibration':score})
    np.savez_compressed(modeldir/'hybrid.npz',net=net)
    (modeldir/'calibration_scores.json').write_text(json.dumps(results,indent=2))
    print('Hybrid calibration',score,flush=True)


if __name__=='__main__':main()
