"""Precompute forecasts per evaluation sequence (forecasts do not depend on battery actions), then tune decisions fast."""
from exp3 import *  # noqa

NET_CS = feats.NET_NAMES.index('cs')


def build_cache(pos, mkt='v2', net='net', tag=''):
    f = HD / f'fc_{feats.VERSION}_{pos}_{mkt}_{net}{tag}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    M = models_for(pos)
    M2 = train_mkt2(pos) if mkt == 'v2' else None
    seq = SEQS[pos]
    hist = feats.History(M['prior'])
    T = len(seq)
    P = np.zeros((T, 24)); C = np.zeros((T, 24)); N = np.zeros((T, 24)); DAY = np.zeros((T, 24), int)
    NK = np.zeros((T, 24), int); LL1 = np.full(T, np.nan)
    for i in range(T):
        r = seq.iloc[i]
        prev = seq.iloc[i - 1] if i > 0 else r
        fut = seq.iloc[i:i + 3]
        obs = {'timestamp': r.Fecha, 'preu_actual': r.precio_eur_per_kwh, 'co2_actual': r.co2_kg_per_kwh,
               'preus_properes_24h': fut.precio_eur_per_kwh.tolist(), 'co2_properes_24h': fut.co2_kg_per_kwh.tolist(),
               'noves_dades_5min': {'ultim_consum_kw': prev.load_kw, 'ultima_solar_kw': prev.pv_kw,
                                    'irradiancia_wm2': prev.irradiancia_wm2, 'temp_ambient_c': prev.temp_ambient_c,
                                    'temp_panel_c': prev.temp_panel_c, 'vent_ms': prev.vent_ms}}
        hist.update(obs)
        nr, mr = feats.features_v2(hist) if mkt == 'v2' else feats.features(hist)
        Xn, Xm = np.array(nr, float), np.array(mr, float)
        if mkt == 'v2':
            pp = Xm[:, IB] + M2['price2'].predict(Xm); pc = Xm[:, IC] + M2['co22'].predict(Xm)
        else:
            pp, pc = M['price'].predict(Xm), M['co2'].predict(Xm)
        kn = hist.known
        for j, (p, c) in enumerate(kn):
            pp[j], pc[j] = p, c
        P[i], C[i] = pp, pc
        if net == 'net':
            N[i] = M['net'].predict(Xn)
        else:
            Ms = net  # dict with load/pv models
            N[i] = Ms['load'].predict(Xn) - np.clip(Ms['pv'].predict(Xn), 0, None)
        DAY[i] = (Xn[:, NET_CS] > 50).astype(int)
        if 'night_known' in feats.NET_NAMES:
            NK[i] = Xn[:, feats.NET_NAMES.index('night_known')]
            LL1[i] = Xn[0, feats.NET_NAMES.index('load_l1')] - 0.0
    out = dict(P=P, C=C, N=N, DAY=DAY, NK=NK, LL1=LL1, ts=list(seq.Fecha))
    pickle.dump(out, open(f, 'wb'))
    return out


class FastAgent:
    def __init__(self, fc, M, K=9, res_scale=1.0, v_term=0.0, H=24, ratio=None, qtab=None, nets_fn=None, night=True):
        self.fc, self.M, self.K, self.rs, self.v, self.H = fc, M, K, res_scale, v_term, H
        self.night = night and 'NK' in fc
        self.rk = M['rk']; self.rc = ratio * self.rk if ratio else M['rc']
        self.q = qtab if qtab is not None else (residual_quantiles(M, K) if K > 1 else None)
        self.i = 0
        self.nets_fn = nets_fn

    def act(self, obs):
        i = self.i; self.i += 1
        fc, H = self.fc, self.H
        ws = list(0.7 * fc['C'][i, :H] / self.rc + 0.3 * fc['P'][i, :H] / self.rk)
        if self.nets_fn:
            netss = self.nets_fn(self, i)
        elif self.q is None:
            netss = [[fc['N'][i, h]] for h in range(H)]
        else:
            netss = [[fc['N'][i, h] + self.rs * q for q in self.q[(h, int(fc['DAY'][i, h]))]] for h in range(H)]
        if self.night:
            for h in range(H):
                if fc['NK'][i, h] == 1 and np.isfinite(fc['LL1'][i]):
                    netss[h] = [float(fc['LL1'][i])]
        soc = float(obs['soc_kwh'])
        cap = float(obs.get('cap_max_kwh', CAP)); pmax = float(obs.get('power_max_kw', PMAX))
        p = dp.plan(ws, netss, soc, cap, pmax, self.v)
        mc = max(0.0, min(pmax, (cap - soc) / dp.ETA - 1e-4)); md = max(0.0, min(pmax, soc * dp.ETA - 1e-4))
        return float(max(-md, min(mc, p)))


def fast_eval(name, make, poss=('last', 'first', 'mid'), quiet=False):
    out = {}
    for pos in poss:
        out[pos] = run(make(pos), SEQS[pos])
    mean = np.mean([r['score'] for r in out.values()])
    if not quiet:
        print(f"{name:46s} " + ' '.join(f"{p}={r['score']:.2f}" for p, r in out.items())
              + f" | MITJANA={mean:.3f} export={np.mean([r['export_mwh'] for r in out.values()]):.0f}MWh "
              f"inf={sum(r['inf'] for r in out.values())} t={max(r['t'] for r in out.values()):.1f}s/seq", flush=True)
    return mean
