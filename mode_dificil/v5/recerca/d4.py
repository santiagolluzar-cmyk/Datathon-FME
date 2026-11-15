"""Robustness on holiday-rich blocks (the hidden test is full of them): scenario widths."""
from e60 import *  # noqa

HOL = {'2026-01-01', '2026-04-01', '2026-08-05', '2026-08-07', '2026-08-09', '2025-12-01', '2026-05-01', '2025-12-18', '2025-09-09'}
MM = {p: models_for(p) for p in POS}
F = {p: research_fc7(p) for p in POS}
Q = {p: qtab_hr(p, oofnet6(p, NS, NTAG, PS)) for p in POS}
X6 = {p: seq_feats6(p)['Xn'] for p in POS}
QM = {p: q0_models(p) for p in POS}


def ev(name, rs, qs, ref=None):
    out = {p: run_detail(QAgent(F[p], MM[p], QM[p], X6[p], qs, qtab=Q[p], K=15, res_scale=rs), SEQS[p]) for p in POS}
    tot = np.mean([out[p].attrs['score'] for p in POS])
    msg = f'{name:34s} MITJANA={tot:.3f}'
    if ref is not None:
        dh, dn = 0.0, 0.0
        for p in POS:
            a = out[p].groupby('block').agg(g=('gain', 'sum'), ini=('ts', 'first'))
            b = ref[p].groupby('block').gain.sum()
            d = a.g - b
            hol = a.ini.astype(str).str[:10].isin(HOL)
            dh += d[hol].sum() / 3; dn += d[~hol].sum() / 3
        msg += f' | dif en blocs festius: {dh:+.3f} | en blocs normals: {dn:+.3f}'
    print(msg, flush=True)
    return out


ref = ev('v5 (rs=1, q0=1.25)', 1.0, 1.25)
ev('rs=1.5, q0=1.25', 1.5, 1.25, ref)
ev('rs=1.5, q0=2.0', 1.5, 2.0, ref)
ev('rs=2.0, q0=2.0', 2.0, 2.0, ref)
ev('rs=1.0, q0=2.0', 1.0, 2.0, ref)
