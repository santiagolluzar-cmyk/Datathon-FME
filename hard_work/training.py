"""Reproducible local hard-mode forecasts; no web requests or submissions."""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def module_at(path, name='hard_agent'):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_data(repo=REPO):
    sys.path.insert(0, str(repo))
    from eval_local import construir_dataset_horari
    data = construir_dataset_horari(repo / 'data/train').set_index('Fecha')
    data['pv_kw'] *= 12
    raw = pd.read_csv(repo / 'data/train/meteo.csv', parse_dates=['Fecha'])
    cols = {}
    for col in raw.columns:
        if 'TEMPERATURA AMBIENTE (SENSOR 1)' in col:
            cols[col] = 'tamb'
        elif 'TEMPERATURA PANEL (SENSOR 1)' in col:
            cols[col] = 'tpan'
        elif 'VELOCIDAD VIENTO (SENSOR 1)' in col:
            cols[col] = 'wind'
        elif col == 'Atersa - IRRADIANCIA (W/m2)':
            cols[col] = 'irr'
    meteo = raw.set_index('Fecha')[list(cols)].apply(pd.to_numeric, errors='coerce').rename(columns=cols).resample('1h').mean()
    return data.join(meteo).reset_index()


def make_obs(data, i, bid):
    row = data.iloc[i]
    last = data.iloc[i-1] if i > 0 and bid[i] == bid[i-1] else row
    horizon = min(i+3, len(data))
    while horizon > i+1 and bid[horizon-1] != bid[i]:
        horizon -= 1
    return {'timestamp': row.Fecha, 'preu_actual': row.precio_eur_per_kwh,
            'co2_actual': row.co2_kg_per_kwh,
            'preus_properes_24h': data.precio_eur_per_kwh.iloc[i:horizon].tolist(),
            'co2_properes_24h': data.co2_kg_per_kwh.iloc[i:horizon].tolist(),
            'noves_dades_5min': {'ultim_consum_kw': last.load_kw, 'ultima_solar_kw': last.pv_kw,
                               'irradiancia_wm2': last.irr, 'temp_ambient_c': last.tamb,
                               'temp_panel_c': last.tpan, 'vent_ms': last.wind}}


def build_features(data, module, prior, split_mask=None):
    breaks = data.Fecha.diff().ne(pd.Timedelta(hours=1)).to_numpy().copy()
    if split_mask is not None:
        breaks |= np.r_[True, split_mask[1:] != split_mask[:-1]]
    bid = np.cumsum(breaks)
    hist = module.History(prior)
    xn, xm, night, last = [], [], [], []
    for i in range(len(data)):
        if i and bid[i] != bid[i-1]:
            hist.reset()
        hist.update(make_obs(data, i, bid))
        a, b, c, d = module.features(hist)
        xn.append(a); xm.append(b); night.append(c); last.append(d)
    return {'Xn': np.asarray(xn), 'Xm': np.asarray(xm), 'night': np.asarray(night, bool),
            'lastload': np.asarray(last), 'block_id': bid,
            'timestamps': data.Fecha.astype(str).to_numpy(dtype=str),
            'load': data.load_kw.to_numpy(), 'pv': data.pv_kw.to_numpy(),
            'price': data.precio_eur_per_kwh.to_numpy(), 'carbon': data.co2_kg_per_kwh.to_numpy()}


def forecast_cache(features, models):
    n = len(features['load'])
    xn, xm = features['Xn'], features['Xm']
    cache = {k: v for k, v in features.items() if k not in ('Xn', 'Xm')}
    cache['bp'], cache['bc'] = xm[:, :, 27], xm[:, :, 28]
    cache['daylight'] = xn[:, :, 8] > 50
    for name, X, base in [('net', xn, None), ('prices', xm, cache['bp']), ('co2', xm, cache['bc'])]:
        mod = models['price' if name == 'prices' else name]
        pred = mod.predict(X.reshape(-1, X.shape[-1]), num_threads=1).reshape(n, 24)
        cache[name] = pred if base is None else base + pred
    # Only the current hour and next two known market hours are overwritten.
    # Horizons across a time gap use a profile; no future labels are consumed.
    for i in range(n):
        for h in range(3):
            if i+h < n and features['block_id'][i+h] == features['block_id'][i]:
                cache['prices'][i,h] = cache['price'][i+h]
                cache['co2'][i,h] = cache['carbon'][i+h]
    return cache


def validation_mask(data):
    dates = data.Fecha
    weeks = dates.dt.to_period('W-SUN')
    selected = []
    for month, group in data.groupby(dates.dt.to_period('M')):
        candidates = []
        for week, w in group.groupby(group.Fecha.dt.to_period('W-SUN')):
            if len(w) == 168 and w.Fecha.iloc[0] == week.start_time and w.Fecha.iloc[-1] == week.end_time.floor('h'):
                candidates.append(week)
        if candidates:
            selected.append(max(candidates))
    return weeks.isin(selected).to_numpy(), [str(w) for w in selected]


def calibration_masks(data, holdout):
    weeks = data.Fecha.dt.to_period('W-SUN')
    selected = sorted(weeks[holdout].unique())
    return (weeks.isin(selected[::2]).to_numpy(), weeks.isin(selected[1::2]).to_numpy())


def residual_tables(cache, calibration):
    """Equal probability residual scenarios from calibration origins/targets only."""
    n = len(cache['load'])
    quantiles = (np.arange(15) + .5) / 15
    table, summary = {}, {}
    for h in range(24):
        j = np.arange(n-h)
        good = calibration[j] & calibration[j+h] & (cache['block_id'][j] == cache['block_id'][j+h])
        residual = cache['load'][j+h] - cache['pv'][j+h] - cache['net'][j,h]
        for day in (0,1):
            selected = residual[good & (cache['daylight'][j,h] == day)]
            table[f'{h}_{day}'] = np.quantile(selected, quantiles).tolist()
            summary[f'{h}_{day}'] = {'n':len(selected), 'bias':float(np.mean(selected)),
                                   'std':float(np.std(selected)), 'rmse':float(np.mean(selected**2)**.5),
                                   'q95_minus_q05':float(np.quantile(selected,.95)-np.quantile(selected,.05))}
    return table, summary


def calibrate_existing(out=OUT):
    data=load_data()
    old=dict(np.load(out/'cache.npz'))
    new=dict(np.load(out/'temporal/cache.npz'))
    cal,confirm=calibration_masks(data,new['validation'])
    table,summary=residual_tables(new,cal)
    _,oldsummary=residual_tables(old,cal)
    (out/'temporal/qtab_oof.json').write_text(json.dumps(table))
    (out/'temporal/residual_comparison.json').write_text(json.dumps({'oof':summary,'old_in_sample':oldsummary},indent=2))
    for cache,path in [(old,out/'cache.npz'),(new,out/'temporal/cache.npz')]:
        cache['calibration'],cache['confirmation']=cal,confirm
        np.savez_compressed(path,**cache)
    print('Calibration',int(cal.sum()),'Confirmation',int(confirm.sum()),flush=True)
    for key in ('0_0','0_1','6_0','6_1','23_0','23_1'):
        print(key,'old RMSE',oldsummary[key]['rmse'],'OOF RMSE',summary[key]['rmse'],flush=True)


def train_full(repo, out, rounds=None):
    """Refit on all published training data, using rounds selected on heldout weeks."""
    import lightgbm as lgb
    module=module_at(repo/'mode_dificil/agent.py')
    original=json.loads((repo/'mode_dificil/prior.json').read_text())
    data=load_data(repo)
    prior=rebuild_prior(data,np.ones(len(data),bool),module,original)
    prior['qtab']=json.loads((out/'temporal/qtab_oof.json').read_text())
    features=build_features(data,module,prior)
    metadata=json.loads((out/'temporal/validation.json').read_text())
    n=len(data); origins=np.repeat(np.arange(n),24); leads=np.tile(np.arange(24),n)
    targets=np.minimum(origins+leads,n-1)
    good=(origins+leads<n)&(features['block_id'][origins]==features['block_id'][targets])
    full=out/'full';full.mkdir(exist_ok=True)
    models={}
    for name,key,values,baseidx in [('net','Xn',features['load']-features['pv'],None),
                                  ('price','Xm',features['price'],27),('co2','Xm',features['carbon'],28)]:
        X=features[key].reshape(n*24,-1); y=values[targets].copy(); mask=good.copy()
        if baseidx is not None:
            y-=X[:,baseidx];mask &= leads>=3
        params=dict(objective='regression',metric='l2',verbosity=-1,num_threads=4,
                    learning_rate=.035,num_leaves=47 if name=='net' else 31,
                    min_data_in_leaf=100,feature_fraction=.85,bagging_fraction=.85,
                    bagging_freq=1,lambda_l2=5,seed=83)
        count=rounds or metadata['metrics'][name]['best_iteration']
        model=lgb.train(params,lgb.Dataset(X[mask],label=y[mask]),num_boost_round=count)
        model.save_model(str(full/f'model_{name}.txt'));models[name]=model
        print('Full-data model',name,count,flush=True)
    prior.pop('_cache',None)
    (full/'prior.json').write_text(json.dumps(prior))
    cache=forecast_cache(features,models)
    np.savez_compressed(full/'cache.npz',**cache)
    return full


def rebuild_prior(data, train_mask, module, original):
    # Sky table is a deterministic clear-sky astronomical curve, not targets.
    pr = {'cs': original['cs'], 'qtab': original['qtab'], 'version': 'temporal_holdout', '_cache': {}}
    train = data.loc[train_mask].copy()
    ts = train.Fecha
    train['month'] = ts.dt.month
    train['hour'] = ts.dt.hour
    train['td'] = [module.tipus_dia(t) for t in ts]
    train['tm'] = [module.tipus_mercat(t) for t in ts]
    for key, col in [('load','load_kw'), ('pv','pv_kw'), ('price','precio_eur_per_kwh'), ('co2','co2_kg_per_kwh'), ('irr','irr')]:
        typ = 'tm' if key in ('price','co2') else 'td'
        table = train.groupby(['month',typ,'hour'])[col].mean()
        broad = train.groupby([typ,'hour'])[col].mean()
        pr[key] = {'mth': {'_'.join(map(str,k)):float(v) for k,v in table.items()},
                   'th': {'_'.join(map(str,k)):float(v) for k,v in broad.items()}}
    baseline = np.maximum(train.load_kw-train.pv_kw,0)
    pr['rk_train'] = float(np.sum(baseline*train.precio_eur_per_kwh)/np.sum(baseline))
    pr['rc_train'] = float(np.sum(baseline*train.co2_kg_per_kwh)/np.sum(baseline))
    return pr


def train_models(data, features, holdout, out, rounds=500):
    import lightgbm as lgb
    n = len(data)
    origins = np.repeat(np.arange(n), 24)
    leads = np.tile(np.arange(24), n)
    targets = origins+leads
    safe = targets<n
    targets = np.minimum(targets,n-1)
    safe &= features['block_id'][origins] == features['block_id'][targets]
    train = safe & ~holdout[origins] & ~holdout[targets]
    valid = safe & holdout[origins] & holdout[targets]
    # Purge the 24 hours preceding every validation start so no horizon touches holdout.
    for i in np.flatnonzero(holdout & ~np.r_[False,holdout[:-1]]):
        train &= ~((origins>=i-24)&(origins<i))
    models, metrics = {}, {}
    for name, key, values, baseidx in [('net','Xn',(features['load']-features['pv']),None),
                                      ('price','Xm',features['price'],27), ('co2','Xm',features['carbon'],28)]:
        X = features[key].reshape(n*24,-1)
        y = values[targets].copy()
        if baseidx is not None:
            y -= X[:,baseidx]
            mt = train & (leads>=3)
            mv = valid & (leads>=3)
        else:
            mt,mv = train,valid
        params = dict(objective='regression', metric='l2', verbosity=-1, num_threads=4,
                      learning_rate=0.035, num_leaves=31 if name!='net' else 47,
                      min_data_in_leaf=100, feature_fraction=0.85, bagging_fraction=0.85,
                      bagging_freq=1, lambda_l2=5, seed=83)
        model=lgb.train(params,lgb.Dataset(X[mt],label=y[mt]),num_boost_round=rounds,
                        valid_sets=[lgb.Dataset(X[mv],label=y[mv])],
                        callbacks=[lgb.early_stopping(70,verbose=False)])
        pred=model.predict(X[mv],num_threads=1)
        metrics[name]={'rows_train':int(mt.sum()),'rows_validation':int(mv.sum()),
                       'best_iteration':model.best_iteration,'rmse':float(np.mean((pred-y[mv])**2)**0.5),
                       'mae':float(np.mean(abs(pred-y[mv])))}
        model.save_model(str(out/f'model_{name}.txt'))
        models[name]=model
        print(name,metrics[name],flush=True)
    return models, metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,default=REPO)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--baseline-only',action='store_true')
    parser.add_argument('--rounds',type=int,default=500)
    parser.add_argument('--calibrate-only',action='store_true')
    parser.add_argument('--final-full',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    if args.calibrate_only:
        calibrate_existing(args.out)
        return
    if args.final_full:
        train_full(args.repo,args.out)
        return
    module=module_at(args.repo/'mode_dificil/agent.py')
    original=json.loads((args.repo/'mode_dificil/prior.json').read_text()); original['_cache']={}
    data=load_data(args.repo)
    print('Data',len(data),'hours',flush=True)
    features=build_features(data,module,original)
    np.savez_compressed(args.out/'features.npz',**features)
    import lightgbm as lgb
    old={k:lgb.Booster(model_str=(args.repo/f'mode_dificil/model_{k}.txt').read_text()) for k in ('net','price','co2')}
    cache=forecast_cache(features,old)
    holdout,weeks=validation_mask(data)
    cache['validation']=holdout
    np.savez_compressed(args.out/'cache.npz',**cache)
    print('Baseline cache ready',flush=True)
    if args.baseline_only:
        return
    out=args.out/'temporal';out.mkdir(exist_ok=True)
    prior=rebuild_prior(data,~holdout,module,original)
    f=build_features(data,module,prior)
    models,metrics=train_models(data,f,holdout,out,args.rounds)
    cache=forecast_cache(f,models);cache['validation']=holdout
    np.savez_compressed(out/'cache.npz',**cache)
    np.savez_compressed(out/'features.npz',**f)
    prior.pop('_cache',None)
    (out/'prior.json').write_text(json.dumps(prior))
    (out/'validation.json').write_text(json.dumps({'weeks':weeks,'metrics':metrics,
        'note':'Models and profiles exclude heldout labels. Historical observed telemetry before an origin is permitted. Existing residual qtab retained temporarily, must recalibrate.'},indent=2))
    print('Temporal cache ready',flush=True)
    calibrate_existing(args.out)


if __name__=='__main__':
    main()
