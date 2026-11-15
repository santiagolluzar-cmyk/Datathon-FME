"""Analog market forecast (teammate's idea) inside the honest cross-fit framework; blends with band models."""
from pkg6 import *  # noqa


def mfeat(x):
    month = x[:, 4]
    return np.column_stack([x[:, 8] / .04, x[:, 9] / .04, x[:, 10] / .04, x[:, 11] / .04, x[:, 12] / .04, x[:, 13] / .04,
                            (x[:, 10] - x[:, 8]) / .015, (x[:, 13] - x[:, 11]) / .015, x[:, 20] / .04, x[:, 21] / .04,
                            x[:, 18] / .06, x[:, 19] / .06, np.sin(month * 2 * np.pi / 12) * .7, np.cos(month * 2 * np.pi / 12) * .7,
                            (x[:, 2] >= 5) * .5])


def analog_fc(pos, k=16):
    g = gen5(pos)
    ok = np.isfinite(g['Yp']).all(axis=1) & (g['meta'][:, 5] >= 3)
    L = g['Xm'][ok][:, 0, :44].astype(float)
    LX = mfeat(L); med = np.nanmedian(LX, axis=0); LX = np.where(np.isfinite(LX), LX, med)
    LP, LC, LH = g['Yp'][ok], g['Yc'][ok], L[:, 5].astype(int)
    mf = mkt_feat_cache(pos, 'v4')
    X0 = mf['X'][:, 0, :44].astype(float)
    QX = mfeat(X0); QX = np.where(np.isfinite(QX), QX, med)
    T = len(X0)
    P = np.full((T, 24), np.nan); C = np.full((T, 24), np.nan)
    pools = {h: np.flatnonzero(LH == h) for h in range(24)}
    for i in range(T):
        if mf['NK'][i] < 3:
            continue
        pool = pools[int(X0[i, 5])]
        d = ((LX[pool] - QX[i]) ** 2).sum(axis=1)
        pick = np.argpartition(d, k - 1)[:k]
        w = 1 / (.2 + d[pick]); w /= w.sum()
        nb = pool[pick]
        P[i] = mf['KP'][i, 2] + w @ (LP[nb] - LP[nb][:, 2:3])
        C[i] = np.maximum(mf['KC'][i, 2] + w @ (LC[nb] - LC[nb][:, 2:3]), 0.0)
    return P, C


MM = {p: models_for(p) for p in POS}
F = {p: research_fc(p) for p in POS}
Q = {p: qtab_hr(p, oofnet6(p, NS, NTAG, PS)) for p in POS}
ref = compare('v3', lambda p: VFAgent(F[p], MM[p], qtab=Q[p], K=15, res_scale=1.0))
print('   MAE w v3:', {k: round(np.mean([mkt_mae(p, F[p])[k] for p in POS]), 4) for k in ('h3-5', 'h6-11', 'h12-23')}, flush=True)
AN = {p: analog_fc(p) for p in POS}


def blend(p, wa, lo=3, hi=23):
    fc = dict(F[p]); P = fc['P'].copy(); C = fc['C'].copy()
    Pa, Ca = AN[p]
    ok = np.isfinite(Pa).all(axis=1)
    P[ok, lo:hi + 1] = (1 - wa) * P[ok, lo:hi + 1] + wa * Pa[ok, lo:hi + 1]
    C[ok, lo:hi + 1] = (1 - wa) * C[ok, lo:hi + 1] + wa * Ca[ok, lo:hi + 1]
    fc['P'], fc['C'] = P, C
    return fc


for name, wa, lo, hi in (('analeg pur', 1.0, 3, 23), ('50% analeg tot', 0.5, 3, 23), ('30% analeg tot', 0.3, 3, 23),
                         ('50% analeg h>=9', 0.5, 9, 23), ('50% analeg h>=6', 0.5, 6, 23), ('30% analeg h3-8 + 60% h>=9', None, 0, 0)):
    if wa is None:
        FB_ = {}
        for p in POS:
            a = blend(p, 0.3, 3, 8); F0 = F[p]; F[p] = a; FB_[p] = blend(p, 0.6, 9, 23); F[p] = F0
    else:
        FB_ = {p: blend(p, wa, lo, hi) for p in POS}
    compare(name, lambda p, FB_=FB_: VFAgent(FB_[p], MM[p], qtab=Q[p], K=15, res_scale=1.0), ref)
    print('   MAE w:', {k: round(np.mean([mkt_mae(p, FB_[p])[k] for p in POS]), 4) for k in ('h3-5', 'h6-11', 'h12-23')}, flush=True)
