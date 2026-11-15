"""v6: horizon-specialised net-demand models with extended features (on top of v5 market bands)."""
from v5 import *  # noqa

N28 = len(feats.NET_NAMES)
N6 = len(feats.NET6_NAMES)


def gen6(pos):
    return gen4(pos, featfn=feats.features_v6, tag='g6')


def _fitn(X, y, P=None):
    PP = dict(PN)
    if P:
        PP.update(P)
    return lgb.LGBMRegressor(**PP).fit(X, y)


def net_models6(pos, nspec, tag, P=None):
    """nspec: list of (lo, hi, ncols). Hours not covered use the pooled base model."""
    f = HD / f'nm6_{tag}_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    g = gen6(pos)
    nD = len(g['meta'])
    Xn = g['Xn'].reshape(nD * 24, -1); yn = g['Yn'].ravel(); h = Xn[:, 0]
    out = []
    for lo, hi, nc in nspec:
        m = (h >= lo) & (h <= hi) & np.isfinite(yn)
        out.append(dict(lo=lo, hi=hi, nc=nc, model=_fitn(Xn[m][:, :nc], yn[m], P)))
    pickle.dump(out, open(f, 'wb'))
    return out


def oofnet6(pos, nspec, tag, P=None):
    f = HD / f'oofn6_{tag}_{pos}.npy'
    if f.exists():
        return np.load(f)
    g = gen6(pos)
    nD = len(g['meta'])
    Xn = g['Xn'].reshape(nD * 24, -1); yn = g['Yn'].ravel(); h = Xn[:, 0]
    fold = np.repeat((g['meta'][:, 3] // 7) % FOLDS, 24)
    f4 = HD / f'oof4_{pos}.npz'
    net = (np.load(f4)['net'] if f4.exists() else np.load(HD / f'oof5_B_{pos}.npz')['net']).ravel().copy()
    for lo, hi, nc in nspec:
        band = (h >= lo) & (h <= hi)
        for fo in range(FOLDS):
            tr = band & (fold != fo) & np.isfinite(yn); va = band & (fold == fo)
            net[va] = _fitn(Xn[tr][:, :nc], yn[tr], P).predict(Xn[va][:, :nc])
    net = net.reshape(nD, 24)
    np.save(f, net)
    return net


def seq_feats6(pos):
    f = HD / f'sf6_{pos}.npz'
    if f.exists():
        return dict(np.load(f))
    M, _ = base_models(pos)
    seq = SEQS[pos]
    hist = feats.History(M['prior'])
    Xn = []
    for i in range(len(seq)):
        hist.update(obs_from(seq, i))
        nr, _ = feats.features_v6(hist)
        Xn.append(nr)
    out = dict(Xn=np.array(Xn, np.float32))
    np.savez(f, **out)
    return out


def fc6(pos, fc, nspec, ntag, P=None):
    """Replace net forecasts of fc (from fc5) by band models where specified."""
    fc = dict(fc)
    X = seq_feats6(pos)['Xn']
    N = fc['N'].copy()
    for md in net_models6(pos, nspec, ntag, P):
        for h in range(md['lo'], md['hi'] + 1):
            N[:, h] = md['model'].predict(X[:, h, :md['nc']])
    fc['N'] = N
    return fc


def qtab6(pos, net_oof, K=15):
    g = gen6(pos)
    return honest_qtab(g, dict(net=net_oof), K)


I_HR = NN.index('hour')


def qtab_hr(pos, net_oof, K=15, min_n=150):
    """Honest residual quantiles by (horizon, hour of day of the target hour)."""
    g = gen6(pos)
    res = g['Yn'] - net_oof
    hr = g['Xn'][:, :, I_HR].astype(int); nk = g['Xn'][:, :, I_NK] > 0.5; day = g['Xn'][:, :, I_CS] > 50
    qs = [(i + 0.5) / K for i in range(K)]
    tab = {}
    for h in range(24):
        for t in range(24):
            m = (hr[:, h] == t) & ~nk[:, h] & np.isfinite(res[:, h])
            r = res[:, h][m]
            if len(r) < min_n:
                dflag = np.median(day[:, h][hr[:, h] == t]) > 0.5 if (hr[:, h] == t).any() else False
                m = (day[:, h] == dflag) & ~nk[:, h] & np.isfinite(res[:, h]); r = res[:, h][m]
            tab[(h, t)] = [float(x) for x in np.quantile(r, qs)]
    return tab


def with_hours(fc):
    fc = dict(fc)
    hrs = pd.Series(fc['ts']).dt.hour.to_numpy()
    fc['DAY'] = (hrs[:, None] + np.arange(24)[None, :]) % 24
    return fc
