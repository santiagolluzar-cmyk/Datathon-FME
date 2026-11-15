"""Conditional quantile scenarios for the current hour (h=0) on top of v7."""
import sys
from pkg7 import *  # noqa

QS = [(i + 0.5) / 9 for i in range(9)]
PQ = dict(n_estimators=250, learning_rate=0.05, num_leaves=15, min_child_samples=60, subsample=0.8, subsample_freq=1,
          colsample_bytree=0.8, verbose=-1, n_jobs=NJ)


def q0_models(pos, tag='q0a', hs=(0,)):
    f = HD / f'{tag}_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    g = gen6(pos)
    o = oofnet6(pos, NS, NTAG, PS)
    out = {}
    for h in hs:
        X = g['Xn'][:, h, :]; r = g['Yn'][:, h] - o[:, h]
        ok = np.isfinite(r) & ~(g['Xn'][:, h, I_NK] > 0.5)
        out[h] = [lgb.LGBMRegressor(objective='quantile', alpha=q, **PQ).fit(X[ok], r[ok]) for q in QS]
    pickle.dump(out, open(f, 'wb'))
    return out


class QAgent(VFAgent):
    def __init__(self, fc, M, qm, X6, scale=1.0, **kw):
        super().__init__(fc, M, **kw)
        self.Q = {h: scale * np.sort(np.column_stack([m.predict(X6[:, h, :]) for m in ms]), axis=1) for h, ms in qm.items()}

    def act(self, obs):
        i = self.i; self.i += 1
        fc, H = self.fc, self.H
        ws = 0.7 * fc['C'][i, :H] / self.rc + 0.3 * fc['P'][i, :H] / self.rk
        netss = scen(fc['N'][i], fc['DAY'][i], fc['NK'][i], fc['LL1'][i], self.q, self.rs)
        for h, Qh in self.Q.items():
            if not (fc['NK'][i, h] and np.isfinite(fc['LL1'][i])):
                netss[h] = [float(fc['N'][i, h] + x) for x in Qh[i]]
        soc = float(obs['soc_kwh'])
        V1 = dp.backward([float(x) for x in ws[1:]], netss[1:], CAP, PMAX)
        p = dp.decide(V1, float(ws[0]), netss[0], soc, CAP, PMAX)
        mc = max(0.0, min(PMAX, (CAP - soc) / dp.ETA - 1e-4)); md = max(0.0, min(PMAX, soc * dp.ETA - 1e-4))
        return float(max(-md, min(mc, p)))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        q0_models(sys.argv[1]); q0_models(sys.argv[1], 'q012', (0, 1, 2)); print('ok', flush=True); sys.exit()
    MM = {p: models_for(p) for p in POS}
    F = {p: research_fc7(p) for p in POS}
    Q = {p: qtab_hr(p, oofnet6(p, NS, NTAG, PS)) for p in POS}
    X6 = {p: seq_feats6(p)['Xn'] for p in POS}
    ref = compare('v4 (v3 + analegs)', lambda p: VFAgent(F[p], MM[p], qtab=Q[p], K=15, res_scale=1.0))
    for tag in ('q0a', 'q012'):
        QM = {p: q0_models(p, tag) for p in POS}
        for sc in (1.0, 1.25):
            compare(f'quantils condicionals [{tag}] x{sc}', lambda p, QM=QM, sc=sc: QAgent(F[p], MM[p], QM[p], X6[p], sc, qtab=Q[p], K=15, res_scale=1.0), ref)
