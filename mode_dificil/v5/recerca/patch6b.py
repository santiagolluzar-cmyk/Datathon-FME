"""Runtime optimisations for agent_v6.py: truncated long-horizon models, pooled net only for uncovered rows."""
from pathlib import Path

HD = Path(__file__).parent
s = (HD / 'agent_v6.py').read_text(encoding='utf-8')


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("""    NCOLS = {'v3': 44, 'v4': 68}""",
    """    NCOLS = {'v3': 44, 'v4': 68}
    N_ITER_NET = 300          # arbres usats del model general de demanda (horitzons llargs): mes rapid, mateix resultat
    N_ITER_LONG = 250         # idem per al model de mercat d'horitzo llarg""")
rep("""    def _predict(self, name, X, fallback):
        m = self.models.get(name)
        if m is None:
            return fallback
        try:
            y = np.asarray(m.predict(X, num_threads=1), dtype=float)""",
    """    def _predict(self, name, X, fallback, n_iter=None):
        m = self.models.get(name)
        if m is None:
            return fallback
        try:
            X = np.ascontiguousarray(X)
            if n_iter:
                y = np.asarray(m.predict(X, num_threads=1, num_iteration=n_iter), dtype=float)
            else:
                y = np.asarray(m.predict(X, num_threads=1), dtype=float)""")
rep("""            prices[lo:hi + 1] += self._predict(f'p{i}', Xb, z)
            co2s[lo:hi + 1] += self._predict(f'c{i}', Xb, z)""",
    """            n_it = self.N_ITER_LONG if fn == 'v3' else None
            prices[lo:hi + 1] += self._predict(f'p{i}', Xb, z, n_it)
            co2s[lo:hi + 1] += self._predict(f'c{i}', Xb, z, n_it)""")
rep("""        npred = np.array(self._predict('net', Xn[:, :28], fallback_net), dtype=float)
        for i, (lo, hi, nc) in enumerate(self.prior['nbands']):""",
    """        nb = self.prior['nbands']
        lo0 = (max(b[1] for b in nb) + 1) if nb and all(f'n{i}' in self.models for i in range(len(nb))) else 0
        npred = np.array(fallback_net, dtype=float)
        npred[lo0:] = self._predict('net', Xn[lo0:, :28], fallback_net[lo0:], self.N_ITER_NET)
        for i, (lo, hi, nc) in enumerate(nb):""")
(HD / 'agent_v6.py').write_text(s, encoding='utf-8')

p = (HD / 'pkg6.py').read_text(encoding='utf-8')
p = p.replace("NTAG = 'n6'\n", "NTAG = 'n6s'\nPS = dict(num_leaves=31, n_estimators=400, min_child_samples=30)\nN_ITER_NET, N_ITER_LONG = 300, 250\n")
p = p.replace("q = qtab_hr(pos, oofnet6(pos, NS, NTAG), 15)", "q = qtab_hr(pos, oofnet6(pos, NS, NTAG, PS), 15)")
p = p.replace("enumerate(net_models6(pos, NS, NTAG)):", "enumerate(net_models6(pos, NS, NTAG, PS)):")
p = p.replace("""def load_agent(d, name='ag6'):""", """def research_fc(pos):
    M, _ = base_models(pos)
    fc = with_hours(fc6(pos, fc5(pos, SPEC, TAG), NS, NTAG, PS))
    X = seq_feats6(pos)['Xn']
    T = X.shape[0]
    N = fc['N'].copy(); P = fc['P'].copy(); C = fc['C'].copy()
    N[:, 6:] = M['net'].predict(X[:, 6:, :N28].reshape(-1, N28), num_iteration=N_ITER_NET).reshape(T, 18)
    md = mkt_models5(pos, SPEC, TAG)[-1]
    mf = mkt_feat_cache(pos, 'v4')
    Xm = mf['X'][:, 9:, :NC['v3']].reshape(-1, NC['v3'])
    ok = mf['NK'] >= 3
    P[ok, 9:] = (Xm[:, IBP] + md['price'].predict(Xm, num_iteration=N_ITER_LONG)).reshape(T, 15)[ok]
    C[ok, 9:] = (Xm[:, IBC] + md['co2'].predict(Xm, num_iteration=N_ITER_LONG)).reshape(T, 15)[ok]
    fc['N'], fc['P'], fc['C'] = N, P, C
    return fc


def load_agent(d, name='ag6'):""")
p = p.replace("    fc = with_hours(fc6(pos, fc5(pos, SPEC, TAG), NS, NTAG))\n", "    fc = research_fc(pos)\n")
(HD / 'pkg6.py').write_text(p, encoding='utf-8')
print('patch ok')
