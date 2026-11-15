"""Generic market-model training / caching by feature function (v3, v4, ...)."""
from v4 import *  # noqa

FEATFN = {'v3': (feats.features_v3, feats.MKT3_NAMES), 'v4': (feats.features_v4, feats.MKT4_NAMES)}


def gen_mkt(pos, fname):
    f = HD / f'gm_{fname}_{pos}.npz'
    if f.exists():
        return dict(np.load(f))
    fn, _ = FEATFN[fname]
    M, _ = base_models(pos)
    train = train_of(pos)
    X, yp, yc, meta = [], [], [], []
    for phase in (0, 84):
        for si, seg in enumerate(segments(train)):
            n = len(seg)
            pr, co = seg.precio_eur_per_kwh.to_numpy(), seg.co2_kg_per_kwh.to_numpy()
            hist = feats.History(M['prior'])
            for i in range(n):
                if (i - phase) % 168 == 0 or i == 0:
                    hist.reset()
                hist.update(obs_from(seg, i))
                _, mr = fn(hist)
                for h in range(3, 24):
                    if i + h >= n:
                        break
                    X.append(mr[h]); yp.append(pr[i + h]); yc.append(co[i + h]); meta.append((hist.ts.toordinal(), h))
    out = dict(X=np.array(X, np.float32), yp=np.array(yp), yc=np.array(yc), meta=np.array(meta))
    np.savez(f, **out)
    return out


def train_mkt(pos, fname, tag, P=None, hsel=None):
    f = HD / f'mk_{tag}_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    d = gen_mkt(pos, fname)
    names = FEATFN[fname][1]
    ib, ic = names.index('bp'), names.index('bc')
    PP = dict(PM)
    if P:
        PP.update(P)
    X, yp, yc = d['X'], d['yp'], d['yc']
    if hsel is not None:
        m = (d['meta'][:, 1] >= hsel[0]) & (d['meta'][:, 1] <= hsel[1])
        X, yp, yc = X[m], yp[m], yc[m]
    out = dict(price=lgb.LGBMRegressor(**PP).fit(X, yp - X[:, ib]), co2=lgb.LGBMRegressor(**PP).fit(X, yc - X[:, ic]),
               fname=fname, ib=ib, ic=ic)
    pickle.dump(out, open(f, 'wb'))
    return out


def mkt_feat_cache(pos, fname):
    """Feature rows of the held-out sequence for every step (T, 24, F) + known values."""
    f = HD / f'mf_{fname}_{pos}.npz'
    if f.exists():
        return dict(np.load(f))
    fn, _ = FEATFN[fname]
    M, _ = base_models(pos)
    seq = SEQS[pos]
    hist = feats.History(M['prior'])
    X, KP, KC, NKn = [], [], [], []
    for i in range(len(seq)):
        hist.update(obs_from(seq, i))
        _, mr = fn(hist)
        X.append(mr)
        kp = [x[0] for x in hist.known] + [np.nan] * 3; kc = [x[1] for x in hist.known] + [np.nan] * 3
        KP.append(kp[:3]); KC.append(kc[:3]); NKn.append(len(hist.known))
    out = dict(X=np.array(X, np.float32), KP=np.array(KP), KC=np.array(KC), NK=np.array(NKn))
    np.savez(f, **out)
    return out


def predict_mkt(models, mf):
    """models: list of (model_dict, hlo, hhi). Returns P, C arrays (T,24) with known hours filled."""
    T = mf['X'].shape[0]
    X = mf['X'].reshape(T * 24, -1)
    P = np.zeros(T * 24); C = np.zeros(T * 24)
    hcol = X[:, 0]
    for md, lo, hi in models:
        m = (hcol >= lo) & (hcol <= hi)
        P[m] = X[m, md['ib']] + md['price'].predict(X[m])
        C[m] = X[m, md['ic']] + md['co2'].predict(X[m])
    P, C = P.reshape(T, 24), C.reshape(T, 24)
    for j in range(3):
        ok = mf['NK'] > j
        P[ok, j] = mf['KP'][ok, j]; C[ok, j] = mf['KC'][ok, j]
    return P, C


def fc_mkt(pos, models_spec):
    """models_spec: list of (tag, fname, P, hlo, hhi)."""
    fc = dict(build_cache(pos))
    fname = models_spec[0][1]
    mf = mkt_feat_cache(pos, fname)
    models = [(train_mkt(pos, fn_, tag, P, (lo, hi) if (lo, hi) != (3, 23) else None), lo, hi) for tag, fn_, P, lo, hi in models_spec]
    fc['P'], fc['C'] = predict_mkt(models, mf)
    return fc


def mkt_mae(pos, fc):
    seq = SEQS[pos]
    co, pr = seq.co2_kg_per_kwh.to_numpy(), seq.precio_eur_per_kwh.to_numpy()
    T = len(seq)
    M = models_for(pos)
    out = {}
    for name, lo, hi in (('h3-5', 3, 5), ('h6-11', 6, 11), ('h12-23', 12, 23)):
        e = []
        for h in range(lo, hi + 1):
            idx = np.arange(T - h)
            wt = 0.7 * co[idx + h] / M['rc'] + 0.3 * pr[idx + h] / M['rk']
            wp = 0.7 * fc['C'][idx, h] / M['rc'] + 0.3 * fc['P'][idx, h] / M['rk']
            e.append(np.abs(wt - wp))
        out[name] = float(np.mean(np.concatenate(e)))
    return out
