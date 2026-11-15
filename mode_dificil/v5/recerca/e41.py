import sys
from v6 import *  # noqa
B = [('v4', 3, 3, 3, 3), ('v4', 4, 4, 4, 4), ('v4', 5, 5, 5, 5), ('v4', 6, 8, 6, 8), ('v3', 9, 23, 3, 23)]
NS = {'n28': [(0, 0, N28), (1, 1, N28), (2, 2, N28), (3, 5, N28)], 'n6': [(0, 0, N6), (1, 1, N6), (2, 2, N6), (3, 5, N6)]}
if len(sys.argv) > 1:
    p = sys.argv[1]
    g = gen6(p); seq_feats6(p)
    for t, ns in NS.items():
        o = oofnet6(p, ns, t); net_models6(p, ns, t)
        res = np.abs(g['Yn'] - o); nk = g['Xn'][:, :, I_NK] > 0.5; res[nk] = np.nan
        print(p, t, 'MAE honest h0 %.1f h1 %.1f h2 %.1f h3-5 %.1f' % (np.nanmean(res[:, 0]), np.nanmean(res[:, 1]), np.nanmean(res[:, 2]), np.nanmean(res[:, 3:6])), flush=True)
    sys.exit()
MM = {p: models_for(p) for p in POS}
FB = {p: fc5(p, B, 'B') for p in POS}
QB = {p: honest_qtab(gen5(p), oof5(p, B, 'B'), 15) for p in POS}
ref = compare('v2 (mercat B, demanda unica)', lambda p: VFAgent(FB[p], MM[p], qtab=QB[p], K=15, res_scale=1.0))
for t, ns in NS.items():
    F = {p: fc6(p, FB[p], ns, t) for p in POS}; Q = {p: qtab6(p, oofnet6(p, ns, t)) for p in POS}
    for rs in (1.0, 1.25):
        compare(f'demanda per horitzo [{t}] rs={rs}', lambda p, F=F, Q=Q, rs=rs: VFAgent(F[p], MM[p], qtab=Q[p], K=15, res_scale=rs), ref)
