"""Experiment harness: training-data generation through the runtime feature code, models, agent, evaluation."""
import math, pickle, sys, time, warnings
from datetime import timedelta
warnings.filterwarnings('ignore')
sys.path.insert(0, r'C:\Users\Núria\AppData\Local\Temp\claude\3ab9b5b62eda92f3\73fe9ce0-e95f-4694-8210-72e4d2985bdb\scratchpad\hd')
import numpy as np
import pandas as pd
import lightgbm as lgb
from base import dataset, month_seq, run, CAP, PMAX, SP
import feats
import dp

HD = SP / 'hd'
D = dataset()
SEQS = {p: month_seq(D, p) for p in ('last', 'first', 'mid')}


def segments(df):
    df = df.sort_values('Fecha').reset_index(drop=True)
    gap = df.Fecha.diff().ne(pd.Timedelta(hours=1)).cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(gap)]


def obs_from(seg, i):
    r = seg.iloc[i]
    fut = seg.iloc[i:i + 3]
    prev = seg.iloc[i - 1] if i > 0 else r
    return {'timestamp': r.Fecha, 'preu_actual': r.precio_eur_per_kwh, 'co2_actual': r.co2_kg_per_kwh,
            'preus_properes_24h': fut.precio_eur_per_kwh.tolist(), 'co2_properes_24h': fut.co2_kg_per_kwh.tolist(),
            'noves_dades_5min': {'ultim_consum_kw': prev.load_kw, 'ultima_solar_kw': prev.pv_kw,
                                 'irradiancia_wm2': prev.irradiancia_wm2, 'temp_ambient_c': prev.temp_ambient_c,
                                 'temp_panel_c': prev.temp_panel_c, 'vent_ms': prev.vent_ms}}


def generate(train, prior, block_len=168, phases=(0, 84)):
    """Simulate the agent's view over train with pseudo-blocks of 168h (block starts = cold starts)."""
    Xn, yn, Xm, yp, yc, meta = [], [], [], [], [], []
    for phase in phases:
        bid = 0
        for seg in segments(train):
            n = len(seg)
            net = (seg.load_kw - seg.pv_kw).to_numpy()
            pr, co = seg.precio_eur_per_kwh.to_numpy(), seg.co2_kg_per_kwh.to_numpy()
            hist = feats.History(prior)
            for i in range(n):
                if (i - phase) % block_len == 0 or i == 0:
                    hist.reset(); bid += 1
                hist.update(obs_from(seg, i))
                nr, mr = feats.features(hist)
                for h in range(feats.H):
                    if i + h >= n:
                        break
                    Xn.append(nr[h]); yn.append(net[i + h])
                    if h >= 3:
                        Xm.append(mr[h]); yp.append(pr[i + h]); yc.append(co[i + h])
                    meta.append((bid, h, phase))
    return (np.array(Xn, float), np.array(yn), np.array(Xm, float), np.array(yp), np.array(yc), np.array(meta))


def train_models(train, cache_key, params=None):
    f = HD / f'models_{cache_key}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    t0 = time.time()
    prior = feats.build_prior(train)
    Xn, yn, Xm, yp, yc, meta = generate(train, prior)
    print(f'  [{cache_key}] dades: net {Xn.shape} mkt {Xm.shape} ({time.time()-t0:.0f}s)', flush=True)
    P = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    if params:
        P.update(params)
    # out-of-fold residuals for the net model (GroupKFold by pseudo-block)
    groups = meta[:, 0]
    folds = groups % 5
    oof = np.full(len(yn), np.nan)
    for fo in range(5):
        tr, va = folds != fo, folds == fo
        m = lgb.LGBMRegressor(**P).fit(Xn[tr], yn[tr])
        oof[va] = m.predict(Xn[va])
    res = yn - oof
    hcol = Xn[:, 0].astype(int)
    day = (Xn[:, feats.NET_NAMES.index('cs')] > 50).astype(int)
    m_net = lgb.LGBMRegressor(**P).fit(Xn, yn)
    m_p = lgb.LGBMRegressor(**P).fit(Xm, yp)
    m_c = lgb.LGBMRegressor(**P).fit(Xm, yc)
    base = (train.load_kw - train.pv_kw).clip(lower=0)
    out = dict(prior=prior, net=m_net, price=m_p, co2=m_c, res=res, res_h=hcol, res_day=day,
               rc=float(base @ train.co2_kg_per_kwh / base.sum()), rk=float(base @ train.precio_eur_per_kwh / base.sum()),
               mae_net=float(np.nanmean(np.abs(res))))
    pickle.dump(out, open(f, 'wb'))
    print(f'  [{cache_key}] models ({time.time()-t0:.0f}s) MAE net oof={out["mae_net"]:.0f} kW', flush=True)
    return out


def residual_quantiles(M, K):
    qs = [(i + 0.5) / K for i in range(K)]
    tab = {}
    for h in range(feats.H):
        for day in (0, 1):
            sel = (M['res_h'] == h) & (M['res_day'] == day)
            r = M['res'][sel]
            r = r[np.isfinite(r)]
            tab[(h, day)] = list(np.quantile(r, qs)) if len(r) > 50 else [0.0] * K
    return tab


class Agent:
    def __init__(self, M, K=9, H=24, use_net='model', use_mkt='model', v_term=0.0, ratio=None, res_scale=1.0):
        self.M, self.K, self.H, self.use_net, self.use_mkt, self.v = M, K, H, use_net, use_mkt, v_term
        self.hist = feats.History(M['prior'])
        self.rc, self.rk = M['rc'], M['rk']
        if ratio:
            self.rc = ratio * self.rk
        self.qtab = residual_quantiles(M, K) if K > 1 else None
        self.res_scale = res_scale
        self.oracle = None  # optional dict with true arrays for ablations

    def act(self, obs):
        cap = float(obs.get('cap_max_kwh', CAP)); pmax = float(obs.get('power_max_kw', PMAX))
        self.hist.update(obs)
        nr, mr = feats.features(self.hist)
        H = self.H
        known = self.hist.known
        if self.use_mkt == 'model':
            Xm = np.array(mr, float)
            pp, pc = self.M['price'].predict(Xm), self.M['co2'].predict(Xm)
        else:
            pp, pc = self.use_mkt(obs)
        prices = [known[j][0] if j < len(known) else pp[j] for j in range(H)]
        co2s = [known[j][1] if j < len(known) else pc[j] for j in range(H)]
        ws = [0.7 * c / self.rc + 0.3 * p / self.rk for p, c in zip(prices, co2s)]
        if self.use_net == 'model':
            npred = self.M['net'].predict(np.array(nr, float))
        else:
            npred = self.use_net(obs)
        netss = []
        for h in range(H):
            if self.qtab is None:
                netss.append([npred[h]])
            else:
                day = 1 if nr[h][feats.NET_NAMES.index('cs')] > 50 else 0
                netss.append([npred[h] + self.res_scale * q for q in self.qtab[(h, day)]])
        soc = float(obs['soc_kwh'])
        p = dp.plan(ws, netss, soc, cap, pmax, self.v)
        mc = max(0.0, min(pmax, (cap - soc) / dp.ETA - 1e-4)); md = max(0.0, min(pmax, soc * dp.ETA - 1e-4))
        return float(max(-md, min(mc, p)))


def models_for(pos):
    seq = SEQS[pos]
    train = D[~D.Fecha.isin(seq.Fecha)].reset_index(drop=True)
    return train_models(train, f'{feats.VERSION}_{pos}')


def evaluate(name, make_agent, poss=('last', 'first', 'mid')):
    out = {}
    for pos in poss:
        M = models_for(pos)
        out[pos] = run(make_agent(M, SEQS[pos]), SEQS[pos])
    mean = np.mean([r['score'] for r in out.values()])
    print(f"{name:46s} " + ' '.join(f"{p}={r['score']:.2f}" for p, r in out.items())
          + f" | MITJANA={mean:.3f} export={np.mean([r['export_mwh'] for r in out.values()]):.0f}MWh "
          f"inf={sum(r['inf'] for r in out.values())} t={max(r['t'] for r in out.values()):.0f}s/seq", flush=True)
    return mean
