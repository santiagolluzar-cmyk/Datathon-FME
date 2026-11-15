"""Market model v2: LightGBM on the residual of the blend heuristic + level-shift features."""
from exp import *  # noqa

IB = feats.MKT2_NAMES.index('bp')
IC = feats.MKT2_NAMES.index('bc')


def gen_mkt2(train, prior, block_len=168, phases=(0, 84)):
    X, yp, yc, meta = [], [], [], []
    for phase in phases:
        bid = 0
        for seg in segments(train):
            n = len(seg)
            pr, co = seg.precio_eur_per_kwh.to_numpy(), seg.co2_kg_per_kwh.to_numpy()
            hist = feats.History(prior)
            for i in range(n):
                if (i - phase) % block_len == 0 or i == 0:
                    hist.reset(); bid += 1
                hist.update(obs_from(seg, i))
                _, mr = feats.features_v2(hist)
                for h in range(3, feats.H):
                    if i + h >= n:
                        break
                    X.append(mr[h]); yp.append(pr[i + h]); yc.append(co[i + h]); meta.append((bid, h))
    return np.array(X, float), np.array(yp), np.array(yc), np.array(meta)


def train_mkt2(pos, params=None, tag='v2'):
    f = HD / f'mkt{tag}_{feats.VERSION}_{pos}.pkl'
    if f.exists():
        return pickle.load(open(f, 'rb'))
    M = models_for(pos)
    seq = SEQS[pos]
    train = D[~D.Fecha.isin(seq.Fecha)].reset_index(drop=True)
    t0 = time.time()
    X, yp, yc, meta = gen_mkt2(train, M['prior'])
    P = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=100, subsample=0.8,
             subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)
    if params:
        P.update(params)
    rp, rcz = yp - X[:, IB], yc - X[:, IC]
    out = dict(price2=lgb.LGBMRegressor(**P).fit(X, rp), co22=lgb.LGBMRegressor(**P).fit(X, rcz))
    pickle.dump(out, open(f, 'wb'))
    print(f'  [mkt{tag} {pos}] {X.shape} ({time.time()-t0:.0f}s)', flush=True)
    return out


class AgentV2(Agent):
    def __init__(self, M, M2, **kw):
        super().__init__(M, **kw)
        self.M2 = M2

        def mkt(obs, me=self):
            _, mr = feats.features_v2(me.hist)
            X = np.array(mr, float)
            return X[:, IB] + me.M2['price2'].predict(X), X[:, IC] + me.M2['co22'].predict(X)
        self.use_mkt = mkt


def evaluate_v2(name, make, poss=('last', 'first', 'mid'), tag='v2', params=None):
    out = {}
    for pos in poss:
        M = models_for(pos)
        M2 = train_mkt2(pos, params, tag)
        out[pos] = run(make(M, M2, SEQS[pos]), SEQS[pos])
    mean = np.mean([r['score'] for r in out.values()])
    print(f"{name:46s} " + ' '.join(f"{p}={r['score']:.2f}" for p, r in out.items())
          + f" | MITJANA={mean:.3f} export={np.mean([r['export_mwh'] for r in out.values()]):.0f}MWh "
          f"inf={sum(r['inf'] for r in out.values())} t={max(r['t'] for r in out.values()):.0f}s/seq", flush=True)
    return mean


if __name__ == '__main__':
    from e2 import truth_hooks
    print('refs: ML/ML K9 = 19.199 ; mercat ML + net REAL = 20.127 ; oracle 21.88')

    def mk_true_net(M, M2, seq):
        _, tn = truth_hooks(seq)
        return AgentV2(M, M2, K=1, use_net=tn)
    evaluate_v2('mercat v2 + net REAL', mk_true_net)
    evaluate_v2('mercat v2 + net ML (K=9)', lambda M, M2, s: AgentV2(M, M2, K=9))
