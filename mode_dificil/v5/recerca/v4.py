"""v4 pipeline: honest out-of-fold forecasts (folds = calendar weeks), honest residual quantiles,
wait-and-see value-function correction (E[perfect-foresight marginal value of stored energy])."""
import os
from lab import *  # noqa

G = 24
GW = CAP / G
GRID = (np.arange(G) + 0.5) * GW
NJ = int(os.environ.get('NJ', '4'))
PN = dict(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=NJ)
PM = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=NJ)
NN = feats.NET_NAMES
I_CS, I_NK, I_LL1, I_HIB = NN.index('cs'), NN.index('night_known'), NN.index('load_l1'), NN.index('hours_in_block')
FOLDS = 5


def train_of(pos):
    return (D.copy() if pos == 'ALL' else D[~D.Fecha.isin(SEQS[pos].Fecha)]).reset_index(drop=True)


def base_models(pos, mtag='m3'):
    if pos == 'ALL':
        return train_models(train_of(pos), f'{feats.VERSION}_ALL'), train_mkt3('ALL', None, mtag)
    return models_for(pos), train_mkt3(pos, None, mtag)


def gen4(pos, featfn=None, tag='g4'):
    f = HD / f'{tag}_{pos}.npz'
    if f.exists():
        return dict(np.load(f))
    featfn = featfn or feats.features_v3
    M, _ = base_models(pos)
    train = train_of(pos)
    Xn, Xm, Yn, Yp, Yc, meta = [], [], [], [], [], []
    for phase in (0, 84):
        for si, seg in enumerate(segments(train)):
            n = len(seg)
            net = (seg.load_kw - seg.pv_kw).to_numpy(); pr = seg.precio_eur_per_kwh.to_numpy(); co = seg.co2_kg_per_kwh.to_numpy()
            hist = feats.History(M['prior'])
            for i in range(n):
                if (i - phase) % 168 == 0 or i == 0:
                    hist.reset()
                hist.update(obs_from(seg, i))
                nr, mr = featfn(hist)
                Xn.append(nr); Xm.append(mr)
                for Y, a in ((Yn, net), (Yp, pr), (Yc, co)):
                    x = a[i:i + 24]
                    Y.append(np.concatenate([x, np.full(24 - len(x), np.nan)]))
                meta.append((phase, si, i, hist.ts.toordinal(), hist.ts.hour, len(hist.known), n))
    out = dict(Xn=np.array(Xn, np.float32), Xm=np.array(Xm, np.float32), Yn=np.array(Yn), Yp=np.array(Yp), Yc=np.array(Yc),
               meta=np.array(meta))
    np.savez(f, **out)
    return out


def oof4(pos, g=None, tag='oof4', gtag='g4', mb=None):
    f = HD / f'{tag}_{pos}.npz'
    if f.exists():
        return dict(np.load(f))
    g = g or gen4(pos, tag=gtag)
    mb = mb or (IB3, IC3)
    nD = len(g['meta'])
    fold = np.repeat((g['meta'][:, 3] // 7) % FOLDS, 24)
    Xn = g['Xn'].reshape(nD * 24, -1); Xm = g['Xm'].reshape(nD * 24, -1)
    yn, yp, yc = g['Yn'].ravel(), g['Yp'].ravel(), g['Yc'].ravel()
    net = np.zeros(nD * 24); p = np.zeros(nD * 24); c = np.zeros(nD * 24)
    hm = Xm[:, 0] >= 3
    for fo in range(FOLDS):
        tr, va = fold != fo, fold == fo
        ok = tr & np.isfinite(yn)
        net[va] = lgb.LGBMRegressor(**PN).fit(Xn[ok], yn[ok]).predict(Xn[va])
        ok = tr & np.isfinite(yp) & hm
        p[va] = Xm[va, mb[0]] + lgb.LGBMRegressor(**PM).fit(Xm[ok], yp[ok] - Xm[ok, mb[0]]).predict(Xm[va])
        c[va] = Xm[va, mb[1]] + lgb.LGBMRegressor(**PM).fit(Xm[ok], yc[ok] - Xm[ok, mb[1]]).predict(Xm[va])
        print(f'  [{tag} {pos}] fold {fo}', flush=True)
    out = dict(net=net.reshape(nD, 24), p=p.reshape(nD, 24), c=c.reshape(nD, 24))
    np.savez(f, **out)
    return out


def honest_qtab(g, o, K=15):
    res = g['Yn'] - o['net']
    day = g['Xn'][:, :, I_CS] > 50
    nk = g['Xn'][:, :, I_NK] > 0.5
    qs = [(i + 0.5) / K for i in range(K)]
    tab = {}
    for h in range(24):
        for dflag in (0, 1):
            r = res[:, h][(day[:, h] == bool(dflag)) & ~nk[:, h]]
            r = r[np.isfinite(r)]
            tab[(h, dflag)] = [float(x) for x in np.quantile(r, qs)] if len(r) > 50 else [0.0] * K
    return tab


# ------------------------------------------------------------------ value-function correction
def pf_all(seg, rc, rk):
    n = len(seg)
    net = (seg.load_kw - seg.pv_kw).to_numpy()
    w = 0.7 * seg.co2_kg_per_kwh.to_numpy() / rc + 0.3 * seg.precio_eur_per_kwh.to_numpy() / rk
    out = np.zeros((n + 1, G))
    V = [(0.0, CAP)]
    for k in range(n - 1, -1, -1):
        V = dp.back_step(V, float(w[k]), [float(net[k])], CAP, PMAX)
        out[k] = dp.grid_slopes(V, GRID)
    return out


def scen(npred, day, nkflags, ll1, qtab, rs):
    out = []
    for h in range(len(npred)):
        if nkflags[h] and np.isfinite(ll1):
            out.append([float(ll1)])
        else:
            out.append([npred[h] + rs * q for q in qtab[(h, int(day[h]))]])
    return out


def vf_features(lam, ws, hour, hib, npred):
    w3 = np.sort(ws[3:]) if len(ws) > 3 else np.array([ws[-1]])
    ctx = [ws[0], ws[1] if len(ws) > 1 else ws[0], ws[2] if len(ws) > 2 else ws[-1], w3[0], w3[len(w3) // 4], w3[len(w3) // 2],
           w3[(3 * len(w3)) // 4], w3[-1], float(np.mean(ws[1:])), lam[0], lam[6], lam[12], lam[18], lam[23],
           float(np.mean(npred)), float(np.min(npred)), hour, hib]
    return [[GRID[j], lam[j]] + ctx for j in range(G)]


def vf_data(pos, rs=1.0, tag='vf', gtag='g4', otag='oof4', K=15):
    f = HD / f'{tag}_{pos}.npz'
    if f.exists():
        return dict(np.load(f))
    M, _ = base_models(pos)
    rc, rk = M['rc'], M['rk']
    g, o = gen4(pos, tag=gtag), oof4(pos, tag=otag, gtag=gtag)
    qtab = honest_qtab(g, o, K)
    segs = segments(train_of(pos))
    PF = [pf_all(s, rc, rk) for s in segs]
    X, y, grp = [], [], []
    meta = g['meta']
    for d in range(len(meta)):
        ph, si, i, ordd, hour, nk, n = (int(x) for x in meta[d])
        if i + 1 > n - 36:
            continue
        P, C = o['p'][d].copy(), o['c'][d].copy()
        P[:nk] = g['Yp'][d, :nk]; C[:nk] = g['Yc'][d, :nk]
        ws = 0.7 * C / rc + 0.3 * P / rk
        xn = g['Xn'][d]
        netss = scen(o['net'][d], xn[:, I_CS] > 50, xn[:, I_NK] > 0.5, float(xn[0, I_LL1]), qtab, rs)
        V1 = dp.backward([float(x) for x in ws[1:]], netss[1:], CAP, PMAX)
        lam = -np.array(dp.grid_slopes(V1, GRID))
        lam_pf = -PF[si][i + 1]
        X.extend(vf_features(lam, ws, hour, float(xn[0, I_HIB]), o['net'][d]))
        y.extend(lam_pf - lam)
        grp.extend([(ordd // 7) % FOLDS] * G)
    out = dict(X=np.array(X, np.float32), y=np.array(y), grp=np.array(grp))
    np.savez(f, **out)
    return out


PV = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=NJ)


def vf_model(pos, tag='vf', params=None, **kw):
    f = HD / f'{tag}m_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    v = vf_data(pos, tag=tag, **kw)
    P = dict(PV)
    if params:
        P.update(params)
    m = lgb.LGBMRegressor(**P).fit(v['X'], v['y'])
    pickle.dump(m, open(f, 'wb'))
    return m


def pav_noninc(x):
    """Isotonic regression (non-increasing) by pool-adjacent-violators."""
    vals, wts = [], []
    for v in x:
        vals.append(float(v)); wts.append(1.0)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w
            vals.pop(); wts.pop()
    out = []
    for v, w in zip(vals, wts):
        out.extend([v] * int(round(w)))
    return out


class VFAgent(FastAgent):
    def __init__(self, fc, M, vfm=None, alpha=1.0, **kw):
        super().__init__(fc, M, **kw)
        self.vfm, self.alpha = vfm, alpha
        ts = pd.Series(fc['ts'])
        blk = (ts.diff() != pd.Timedelta(hours=1)).cumsum()
        self.hib = np.minimum(ts.groupby(blk).cumcount().to_numpy(), 200)
        self.hours = ts.dt.hour.to_numpy()

    def act(self, obs):
        i = self.i; self.i += 1
        fc, H = self.fc, self.H
        ws = 0.7 * fc['C'][i, :H] / self.rc + 0.3 * fc['P'][i, :H] / self.rk
        netss = scen(fc['N'][i], fc['DAY'][i], fc['NK'][i] if self.night else np.zeros(H), fc['LL1'][i], self.q, self.rs)
        soc = float(obs['soc_kwh'])
        V1 = dp.backward([float(x) for x in ws[1:]], netss[1:], CAP, PMAX)
        if self.vfm is not None:
            lam = -np.array(dp.grid_slopes(V1, GRID))
            X = np.array(vf_features(lam, ws, int(self.hours[i]), float(self.hib[i]), fc['N'][i]), dtype=np.float32)
            lam2 = np.maximum(lam + self.alpha * self.vfm.predict(X), 0.0)
            V1 = [(-v, GW) for v in pav_noninc(lam2)]
        p = dp.decide(V1, float(ws[0]), netss[0], soc, CAP, PMAX)
        mc = max(0.0, min(PMAX, (CAP - soc) / dp.ETA - 1e-4)); md = max(0.0, min(PMAX, soc * dp.ETA - 1e-4))
        return float(max(-md, min(mc, p)))
