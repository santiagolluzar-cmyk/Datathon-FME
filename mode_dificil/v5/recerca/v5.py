"""v5: horizon-band market models + policy-evaluation (rollout) targets for the marginal value of stored energy."""
from v4m import *  # noqa

NC = {'v3': len(feats.MKT3_NAMES), 'v4': len(feats.MKT4_NAMES)}
IBP, IBC = feats.MKT4_NAMES.index('bp'), feats.MKT4_NAMES.index('bc')


def gen5(pos):
    return gen4(pos, featfn=feats.features_v4, tag='g5')


def _fit(X, y, P=None):
    PP = dict(PM)
    if P:
        PP.update(P)
    return lgb.LGBMRegressor(**PP).fit(X, y)


def mkt_models5(pos, spec, tag):
    f = HD / f'mk5_{tag}_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    g = gen5(pos)
    nD = len(g['meta'])
    Xm = g['Xm'].reshape(nD * 24, -1)
    h = Xm[:, 0]
    yp, yc = g['Yp'].ravel(), g['Yc'].ravel()
    out = []
    for fn, lo, hi, tlo, thi in spec:
        m = (h >= tlo) & (h <= thi) & np.isfinite(yp)
        X = Xm[m][:, :NC[fn]]
        out.append(dict(fn=fn, lo=lo, hi=hi, price=_fit(X, yp[m] - X[:, IBP]), co2=_fit(X, yc[m] - X[:, IBC])))
    pickle.dump(out, open(f, 'wb'))
    return out


def predict5(models, X24):
    """X24: (T,24,F) v4 feature rows -> P, C (T,24) for h>=3 (NaN elsewhere)."""
    T = X24.shape[0]
    X = X24.reshape(T * 24, -1)
    h = X[:, 0]
    P = np.full(T * 24, np.nan); C = np.full(T * 24, np.nan)
    for md in models:
        m = (h >= md['lo']) & (h <= md['hi'])
        Xb = X[m][:, :NC[md['fn']]]
        P[m] = Xb[:, IBP] + md['price'].predict(Xb)
        C[m] = Xb[:, IBC] + md['co2'].predict(Xb)
    return P.reshape(T, 24), C.reshape(T, 24)


def oof5(pos, spec, tag):
    f = HD / f'oof5_{tag}_{pos}.npz'
    if f.exists():
        return dict(np.load(f))
    g = gen5(pos)
    nD = len(g['meta'])
    Xm = g['Xm'].reshape(nD * 24, -1)
    h = Xm[:, 0]
    fold = np.repeat((g['meta'][:, 3] // 7) % FOLDS, 24)
    yp, yc = g['Yp'].ravel(), g['Yc'].ravel()
    P = np.full(nD * 24, np.nan); C = np.full(nD * 24, np.nan)
    for fn, lo, hi, tlo, thi in spec:
        band = (h >= lo) & (h <= hi)
        tband = (h >= tlo) & (h <= thi)
        for fo in range(FOLDS):
            tr = tband & (fold != fo) & np.isfinite(yp); va = band & (fold == fo)
            Xt, Xv = Xm[tr][:, :NC[fn]], Xm[va][:, :NC[fn]]
            P[va] = Xv[:, IBP] + _fit(Xt, yp[tr] - Xt[:, IBP]).predict(Xv)
            C[va] = Xv[:, IBC] + _fit(Xt, yc[tr] - Xt[:, IBC]).predict(Xv)
        print(f'  [oof5 {tag} {pos}] band {lo}-{hi}', flush=True)
    f4 = HD / f'oof4_{pos}.npz'
    if f4.exists():
        net = np.load(f4)['net']
    else:
        Xn = g['Xn'].reshape(nD * 24, -1)
        yn = g['Yn'].ravel()
        net = np.zeros(nD * 24)
        for fo in range(FOLDS):
            tr = (fold != fo) & np.isfinite(yn)
            net[fold == fo] = lgb.LGBMRegressor(**PN).fit(Xn[tr], yn[tr]).predict(Xn[fold == fo])
        net = net.reshape(nD, 24)
    assert net.shape[0] == nD
    out = dict(net=net, p=P.reshape(nD, 24), c=C.reshape(nD, 24))
    np.savez(f, **out)
    return out


def fc5(pos, spec, tag):
    fc = dict(build_cache(pos))
    mf = mkt_feat_cache(pos, 'v4')
    P, C = predict5(mkt_models5(pos, spec, tag), mf['X'])
    for j in range(3):
        ok = mf['NK'] > j
        P[ok, j] = mf['KP'][ok, j]; C[ok, j] = mf['KC'][ok, j]
    # steps with fewer than 3 known hours (end of data): carry last known forward
    for arr in (P, C):
        for i in np.where(~np.isfinite(arr).all(axis=1))[0]:
            row = arr[i]
            for j in range(24):
                if not np.isfinite(row[j]):
                    row[j] = row[j - 1] if j > 0 else 0.0
    fc['P'], fc['C'] = P, C
    return fc


# ------------------------------------------------------------------ CE preparation on training decisions
def ce_prep(pos, o, rc, rk, qtab, rs=1.0):
    g = gen5(pos)
    meta = g['meta']
    nD = len(meta)
    WS = np.zeros((nD, 24)); LAM = np.zeros((nD, G)); V1s = [None] * nD; N0 = [None] * nD
    for d in range(nD):
        nk = int(meta[d, 5])
        P, C = o['p'][d].copy(), o['c'][d].copy()
        P[:nk] = g['Yp'][d, :nk]; C[:nk] = g['Yc'][d, :nk]
        for arr in (P, C):
            for j in range(24):
                if not np.isfinite(arr[j]):
                    arr[j] = arr[j - 1]
        ws = 0.7 * C / rc + 0.3 * P / rk
        xn = g['Xn'][d]
        netss = scen(o['net'][d], xn[:, I_CS] > 50, xn[:, I_NK] > 0.5, float(xn[0, I_LL1]), qtab, rs)
        V1 = dp.backward([float(x) for x in ws[1:]], netss[1:], CAP, PMAX)
        WS[d] = ws; V1s[d] = V1; N0[d] = sorted(netss[0]); LAM[d] = -np.array(dp.grid_slopes(V1, GRID))
    return WS, LAM, V1s, N0


def vf_X(g, o, WS, LAM):
    meta = g['meta']
    X = []
    for d in range(len(meta)):
        X.extend(vf_features(LAM[d], WS[d], int(meta[d, 4]), float(g['Xn'][d, 0, I_HIB]), o['net'][d]))
    return np.array(X, np.float32)


def _vint(sl, soc):
    """Integral of grid slopes from 0 to soc."""
    full = int(soc // GW)
    full = min(full, G)
    v = float(np.sum(sl[:full])) * GW
    if full < G:
        v += sl[full] * (soc - full * GW)
    return v


def rollout_lam(pos, WS, V1s, N0, rc, rk, T=30, step=2):
    """Marginal value of stored energy under the given policy (V1s per decision), by paired rollouts on true data."""
    g = gen5(pos)
    meta = g['meta']
    nD = len(meta)
    segs = segments(train_of(pos))
    PF = [pf_all(s, rc, rk) for s in segs]
    NT = [(s.load_kw - s.pv_kw).to_numpy() for s in segs]
    WT = [0.7 * s.co2_kg_per_kwh.to_numpy() / rc + 0.3 * s.precio_eur_per_kwh.to_numpy() / rk for s in segs]
    idx = {(int(m[0]), int(m[1]), int(m[2])): d for d, m in enumerate(meta)}
    SG = np.arange(0, G + 1, step) * GW
    out = np.full((nD, G), np.nan)
    eta = dp.ETA
    for d in range(nD):
        ph, si, i, n = int(meta[d, 0]), int(meta[d, 1]), int(meta[d, 2]), int(meta[d, 6])
        end = i + 1 + T
        if end > n - 24:
            continue
        net, wt, pfe = NT[si], WT[si], PF[si][end]
        memo = {}
        J = []
        for s0 in SG:
            soc = float(s0)
            path = []
            tau = i + 1
            tail = None
            while tau < end:
                key = (tau, round(soc, 4))
                if key in memo:
                    tail = memo[key]
                    break
                dd = idx[(ph, si, tau)]
                p = dp.decide(V1s[dd], float(WS[dd, 0]), N0[dd], soc, CAP, PMAX)
                mc = max(0.0, min(PMAX, (CAP - soc) / eta - 1e-4)); md = max(0.0, min(PMAX, soc * eta - 1e-4))
                p = max(-md, min(mc, p))
                c = wt[tau] * max(net[tau] + p, 0.0)
                path.append((key, c))
                soc += p * eta if p > 0 else p / eta
                tau += 1
            if tail is None:
                tail = _vint(pfe, soc)
            for key, c in reversed(path):
                tail += c
                memo[key] = tail
            J.append(tail)
        lam = -np.diff(J) / (GW * step)
        out[d] = np.repeat(lam, step)
    return out


def vf_fit(X, y, grp, P=None):
    """y: (nD*G,) with NaN for invalid. Returns (oof predictions for all rows, final model)."""
    PP = dict(PV)
    if P:
        PP.update(P)
    ok = np.isfinite(y)
    oof = np.zeros(len(y))
    for fo in range(FOLDS):
        tr = ok & (grp != fo)
        oof[grp == fo] = lgb.LGBMRegressor(**PP).fit(X[tr], y[tr]).predict(X[grp == fo])
    return oof, lgb.LGBMRegressor(**PP).fit(X[ok], y[ok])


def corrected_V1s(LAM, corr, alpha):
    lam2 = np.maximum(LAM + alpha * corr.reshape(LAM.shape), 0.0)
    return [[(-v, GW) for v in pav_noninc(row)] for row in lam2]


def build_vf(pos, spec, tag, iters=2, alpha=1.0, T=30, rs=1.0, K=15):
    """Policy iteration: iteration 0 evaluates the CE policy, later ones the corrected policy. Returns list of models."""
    f = HD / f'vf5_{tag}_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    M, _ = base_models(pos)
    rc, rk = M['rc'], M['rk']
    g = gen5(pos)
    o = oof5(pos, spec, tag)
    qtab = honest_qtab(g, o, K)
    t0 = time.time()
    WS, LAM, V1s, N0 = ce_prep(pos, o, rc, rk, qtab, rs)
    X = vf_X(g, o, WS, LAM)
    grp = np.repeat((g['meta'][:, 3] // 7) % FOLDS, G)
    models = []
    pol = V1s
    for it in range(iters):
        lam_t = rollout_lam(pos, WS, pol, N0, rc, rk, T=T)
        y = (lam_t - LAM).ravel()
        oof, m = vf_fit(X, y, grp)
        models.append(m)
        ok = np.isfinite(y)
        print(f'  [vf5 {tag} {pos}] iter {it}: target mean {np.nanmean(y):+.4f}, |target| {np.nanmean(np.abs(y)):.4f}, '
              f'oof resid {np.mean(np.abs(y[ok] - oof[ok])):.4f} ({time.time() - t0:.0f}s)', flush=True)
        pol = corrected_V1s(LAM, oof, alpha)
    out = dict(models=models, qtab=qtab)
    pickle.dump(out, open(f, 'wb'))
    return out


def build_vf_pf(pos, spec, tag, rs=1.0, K=15, P=None, ptag='pf'):
    """Wait-and-see targets: perfect-foresight marginal value minus CE marginal value."""
    f = HD / f'vf5{ptag}_{tag}_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    M, _ = base_models(pos)
    rc, rk = M['rc'], M['rk']
    g = gen5(pos)
    o = oof5(pos, spec, tag)
    qtab = honest_qtab(g, o, K)
    WS, LAM, V1s, N0 = ce_prep(pos, o, rc, rk, qtab, rs)
    X = vf_X(g, o, WS, LAM)
    grp = np.repeat((g['meta'][:, 3] // 7) % FOLDS, G)
    segs = segments(train_of(pos))
    PF = [pf_all(s, rc, rk) for s in segs]
    meta = g['meta']
    y = np.full((len(meta), G), np.nan)
    for d in range(len(meta)):
        si, i, n = int(meta[d, 1]), int(meta[d, 2]), int(meta[d, 6])
        if i + 1 <= n - 36:
            y[d] = -PF[si][i + 1] - LAM[d]
    oof, m = vf_fit(X, y.ravel(), grp, P)
    out = dict(models=[m], qtab=qtab, oof=oof.astype(np.float32), y=y.astype(np.float32), LAM=LAM.astype(np.float32))
    pickle.dump(out, open(f, 'wb'))
    return out
