from e60 import *  # noqa

MM = {p: models_for(p) for p in POS}
F0 = {p: research_fc(p) for p in POS}
LIB = {p: analog_lib(p) for p in POS}
AN = {(p, k): analog_fc(p, LIB[p], k) for p in POS for k in (8, 16)}
Q = {p: qtab_hr(p, oofnet6(p, NS, NTAG, PS)) for p in POS}
X6 = {p: seq_feats6(p)['Xn'] for p in POS}
QM = {p: q0_models(p) for p in POS}


def mix(p, wts, k=16):
    """wts: list of (lo, hi, w_analog)."""
    fc = dict(F0[p]); P = fc['P'].copy(); C = fc['C'].copy()
    Pa, Ca = AN[(p, k)]
    ok = np.isfinite(Pa).all(axis=1)
    for lo, hi, wa in wts:
        P[ok, lo:hi + 1] = (1 - wa) * P[ok, lo:hi + 1] + wa * Pa[ok, lo:hi + 1]
        C[ok, lo:hi + 1] = (1 - wa) * C[ok, lo:hi + 1] + wa * Ca[ok, lo:hi + 1]
    fc['P'], fc['C'] = P, C
    return fc


def ev(name, wts, k=16, ref=None):
    F = {p: mix(p, wts, k) for p in POS}
    return compare(name, lambda p: QAgent(F[p], MM[p], QM[p], X6[p], 1.25, qtab=Q[p], K=15, res_scale=1.0), ref)


ref = ev('v5 (60% analegs a tot arreu)', [(3, 23, 0.6)])
ev('40% h3-5, 60% h6-8, 80% h>=9', [(3, 5, 0.4), (6, 8, 0.6), (9, 23, 0.8)], ref=ref)
ev('50% h3-8, 100% h>=9', [(3, 8, 0.5), (9, 23, 1.0)], ref=ref)
ev('60% h3-8, 80% h>=9', [(3, 8, 0.6), (9, 23, 0.8)], ref=ref)
ev('70% h3-8, 50% h>=9', [(3, 8, 0.7), (9, 23, 0.5)], ref=ref)
ev('60% tot, k=8', [(3, 23, 0.6)], k=8, ref=ref)
