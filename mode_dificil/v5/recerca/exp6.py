"""Market v3: short-term dynamics features. Evaluate via cached net forecasts + new market forecasts."""
from fcache import *  # noqa

N3 = feats.MKT3_NAMES
IB3, IC3 = N3.index('bp'), N3.index('bc')


def gen_mkt3(train, prior, block_len=168, phases=(0, 84)):
    X, yp, yc, meta = [], [], [], []
    for phase in phases:
        bid = 0
        for seg in segments(train):
            n = len(seg)
            pr, co = seg.precio_eur_per_kwh.to_numpy(), seg.co2_kg_per_kwh.to_numpy()
            hist = feats.History(prior)
            for i in range(n):
                if (i - phase) % block_len == 0 or i == 0:
                    hist.reset(); bid += 1
                hist.update(obs_from(seg, i))
                _, mr = feats.features_v3(hist)
                for h in range(3, feats.H):
                    if i + h >= n:
                        break
                    X.append(mr[h]); yp.append(pr[i + h]); yc.append(co[i + h]); meta.append((bid, h))
    return np.array(X, float), np.array(yp), np.array(yc), np.array(meta)


def train_mkt3(pos, P=None, tag='m3'):
    f = HD / f'{tag}_{feats.VERSION}_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    if pos == 'ALL':
        train = D.copy().reset_index(drop=True)
        M = train_models(train, f'{feats.VERSION}_ALL')
    else:
        M = models_for(pos)
        train = D[~D.Fecha.isin(SEQS[pos].Fecha)].reset_index(drop=True)
    X, yp, yc, meta = gen_mkt3(train, M['prior'])
    PP = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    if P:
        PP.update(P)
    out = dict(price=lgb.LGBMRegressor(**PP).fit(X, yp - X[:, IB3]), co2=lgb.LGBMRegressor(**PP).fit(X, yc - X[:, IC3]))
    pickle.dump(out, open(f, 'wb'))
    print(f'  [{tag} {pos}] {X.shape}', flush=True)
    return out


def market_cache(pos, M3, tag='m3'):
    f = HD / f'fcm_{tag}_{feats.VERSION}_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    M = models_for(pos)
    seq = SEQS[pos]
    hist = feats.History(M['prior'])
    T = len(seq)
    P = np.zeros((T, 24)); C = np.zeros((T, 24))
    for i in range(T):
        r = seq.iloc[i]; prev = seq.iloc[i - 1] if i > 0 else r; fut = seq.iloc[i:i + 3]
        obs = {'timestamp': r.Fecha, 'preu_actual': r.precio_eur_per_kwh, 'co2_actual': r.co2_kg_per_kwh,
               'preus_properes_24h': fut.precio_eur_per_kwh.tolist(), 'co2_properes_24h': fut.co2_kg_per_kwh.tolist(),
               'noves_dades_5min': {'ultim_consum_kw': prev.load_kw, 'ultima_solar_kw': prev.pv_kw,
                                    'irradiancia_wm2': prev.irradiancia_wm2, 'temp_ambient_c': prev.temp_ambient_c,
                                    'temp_panel_c': prev.temp_panel_c, 'vent_ms': prev.vent_ms}}
        hist.update(obs)
        _, mr = feats.features_v3(hist)
        X = np.array(mr, float)
        pp = X[:, IB3] + M3['price'].predict(X); pc = X[:, IC3] + M3['co2'].predict(X)
        for j, (p, c) in enumerate(hist.known):
            pp[j], pc[j] = p, c
        P[i], C[i] = pp, pc
    out = dict(P=P, C=C)
    pickle.dump(out, open(f, 'wb'))
    return out


if __name__ == '__main__':
    FC = {p: build_cache(p) for p in SEQS}
    MM = {p: models_for(p) for p in SEQS}
    fast_eval('mercat v2 (base actual)', lambda p: FastAgent(FC[p], MM[p], K=15, res_scale=2.0))
    for tag, P in (('m3', None), ('m3b', dict(num_leaves=15, n_estimators=400))):
        FC3 = {}
        for p in SEQS:
            mc = market_cache(p, train_mkt3(p, P, tag), tag)
            fc = dict(FC[p]); fc['P'], fc['C'] = mc['P'], mc['C']; FC3[p] = fc
        fast_eval(f'mercat v3 ({tag})', lambda p: FastAgent(FC3[p], MM[p], K=15, res_scale=2.0))
