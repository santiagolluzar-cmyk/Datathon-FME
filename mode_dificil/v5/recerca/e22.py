import sys
from v4m import *  # noqa
BANDS = {'v4h3': ('v4', 3, 3), 'v4h4': ('v4', 4, 4), 'v4h5': ('v4', 5, 5), 'v4b35': ('v4', 3, 5), 'v4b68': ('v4', 6, 8), 'v4b611': ('v4', 6, 11),
         'v4b914': ('v4', 9, 14), 'v4b1223': ('v4', 12, 23), 'v4b1523': ('v4', 15, 23), 'v3s': ('v3', 3, 8), 'v3b914': ('v3', 9, 14), 'v3b1523': ('v3', 15, 23),
         'v4s': ('v4', 3, 8), 'v4l': ('v4', 9, 23)}
if len(sys.argv) > 1:
    p = sys.argv[1]
    gen_mkt(p, 'v3'); mkt_feat_cache(p, 'v3')
    for tag, (fn, lo, hi) in BANDS.items():
        train_mkt(p, fn, tag, None, (lo, hi))
    print('ok', p, flush=True); sys.exit()
MM = {p: models_for(p) for p in POS}
QT = {p: honest_qtab(gen4(p), oof4(p)) for p in POS}
F3 = {p: fc_for(p) for p in POS}
MF = {(p, fn): mkt_feat_cache(p, fn) for p in POS for fn in ('v3', 'v4')}


def stitch(p, parts):
    fc = dict(F3[p]); P = fc['P'].copy(); C = fc['C'].copy()
    for tag in parts:
        fn, lo, hi = BANDS[tag]
        Pm, Cm = predict_mkt([(train_mkt(p, fn, tag, None, (lo, hi)), lo, hi)], MF[(p, fn)])
        P[:, lo:hi + 1] = Pm[:, lo:hi + 1]; C[:, lo:hi + 1] = Cm[:, lo:hi + 1]
    fc['P'], fc['C'] = P, C
    return fc


ref = compare('v4s+v4l (ref 19.996)', lambda p: VFAgent(stitch(p, ['v4s', 'v4l']), MM[p], qtab=QT[p], K=15, res_scale=1.0))
for name, parts in (('A v4[3-8] + m3[9-23]', ['v4s']),
                    ('B v4 h3,h4,h5 + v4[6-8] + m3', ['v4h3', 'v4h4', 'v4h5', 'v4b68']),
                    ('C v4[3-5] + v4[6-11] + m3[12-23]', ['v4b35', 'v4b611']),
                    ('D v3[3-8] + m3', ['v3s']),
                    ('E v4[3-5]+v4[6-8]+v4[9-14]+v4[15-23]', ['v4b35', 'v4b68', 'v4b914', 'v4b1523']),
                    ('F v4[3-5]+v4[6-8]+v3[9-14]+v3[15-23]', ['v4b35', 'v4b68', 'v3b914', 'v3b1523']),
                    ('G v4 h3,h4,h5+v4[6-8]+v3[9-14]+v3[15-23]', ['v4h3', 'v4h4', 'v4h5', 'v4b68', 'v3b914', 'v3b1523'])):
    F = {p: stitch(p, parts) for p in POS}
    compare(name, lambda p, F=F: VFAgent(F[p], MM[p], qtab=QT[p], K=15, res_scale=1.0), ref)
    print('   MAE w:', {k: round(np.mean([mkt_mae(p, F[p])[k] for p in POS]), 4) for k in ('h3-5', 'h6-11', 'h12-23')}, flush=True)
