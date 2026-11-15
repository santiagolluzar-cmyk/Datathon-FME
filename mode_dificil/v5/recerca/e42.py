from v6 import *  # noqa
B = [('v4', 3, 3, 3, 3), ('v4', 4, 4, 4, 4), ('v4', 5, 5, 5, 5), ('v4', 6, 8, 6, 8), ('v3', 9, 23, 3, 23)]
NS = [(0, 0, N6), (1, 1, N6), (2, 2, N6), (3, 5, N6)]
I_HR = NN.index('hour')


def qtab_hr(pos, net_oof, K=15, min_n=150):
    g = gen6(pos)
    res = g['Yn'] - net_oof
    hr = g['Xn'][:, :, I_HR].astype(int); nk = g['Xn'][:, :, I_NK] > 0.5; day = g['Xn'][:, :, I_CS] > 50
    qs = [(i + 0.5) / K for i in range(K)]
    tab = {}
    for h in range(24):
        for t in range(24):
            m = (hr[:, h] == t) & ~nk[:, h] & np.isfinite(res[:, h])
            r = res[:, h][m]
            if len(r) < min_n:   # fall back to day/night pool
                dflag = np.median(day[:, h][hr[:, h] == t]) > 0.5 if (hr[:, h] == t).any() else False
                m = (day[:, h] == dflag) & ~nk[:, h] & np.isfinite(res[:, h]); r = res[:, h][m]
            tab[(h, t)] = [float(x) for x in np.quantile(r, qs)]
    return tab


def with_hours(fc):
    fc = dict(fc)
    hrs = pd.Series(fc['ts']).dt.hour.to_numpy()
    fc['DAY'] = (hrs[:, None] + np.arange(24)[None, :]) % 24
    return fc


MM = {p: models_for(p) for p in POS}
F = {p: fc6(p, fc5(p, B, 'B'), NS, 'n6') for p in POS}
O = {p: oofnet6(p, NS, 'n6') for p in POS}
Q = {p: qtab6(p, O[p]) for p in POS}
ref = compare('n6 quantils (h, dia/nit) K=15', lambda p: VFAgent(F[p], MM[p], qtab=Q[p], K=15, res_scale=1.0))
FH = {p: with_hours(F[p]) for p in POS}
for K in (15, 25):
    QH = {p: qtab_hr(p, O[p], K) for p in POS}
    for rs in (1.0, 1.2):
        compare(f'quantils (h, hora del dia) K={K} rs={rs}', lambda p, QH=QH, K=K, rs=rs: VFAgent(FH[p], MM[p], qtab=QH[p], K=K, res_scale=rs), ref)
Q25 = {p: qtab6(p, O[p], 25) for p in POS}
compare('quantils (h, dia/nit) K=25', lambda p: VFAgent(F[p], MM[p], qtab=Q25[p], K=25, res_scale=1.0), ref)
