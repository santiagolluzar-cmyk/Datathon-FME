"""agent_v6.py -> agent_v7.py: blend band market models with analog trajectories (teammate's idea)."""
from pathlib import Path

HD = Path(__file__).parent
s = (HD / 'agent_v6.py').read_text(encoding='utf-8')


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("""    N_ITER_NET = 300""", """    ANALOG_W = 0.6            # pes de la previsio de mercat per analegs (trajectories historiques semblants)
    ANALOG_K = 16
    N_ITER_NET = 300""")
rep("""        self.hist = History(self.prior)
        self.n_calls = 0""", """        self.hist = History(self.prior)
        self.n_calls = 0
        self.analog = None
        try:
            with np.load(DIR / 'analog.npz', allow_pickle=False) as z:
                self.analog = {key: z[key] for key in z.files}
            self.an_pools = {h: np.flatnonzero(self.analog['hour'] == h) for h in range(24)}
        except Exception:
            self.analog = None""")
rep("""    def _plan(self, obs, soc, cap, pmax):""", """    def _analog(self, row):
        \"\"\"Previsio de preu i CO2 a 24 h: mitjana ponderada de les trajectories de train mes semblants a les
        3 hores conegudes (mateixa hora d'origen), enganxada a la darrera hora coneguda.\"\"\"
        lib = self.analog
        mo = row[4]
        x = np.array([row[8] / .04, row[9] / .04, row[10] / .04, row[11] / .04, row[12] / .04, row[13] / .04,
                      (row[10] - row[8]) / .015, (row[13] - row[11]) / .015, row[20] / .04, row[21] / .04,
                      row[18] / .06, row[19] / .06, math.sin(mo * 2 * math.pi / 12) * .7, math.cos(mo * 2 * math.pi / 12) * .7,
                      (row[2] >= 5) * .5], dtype=float)
        x = np.where(np.isfinite(x), x, lib['median'])
        pool = self.an_pools[int(row[5])]
        d = ((lib['x'][pool] - x) ** 2).sum(axis=1)
        k = min(self.ANALOG_K, len(pool))
        pick = np.argpartition(d, k - 1)[:k]
        w = 1.0 / (.2 + d[pick])
        w /= w.sum()
        nb = pool[pick]
        pa = row[10] + w @ (lib['dp'][nb] - lib['dp'][nb][:, 2:3])
        ca = np.maximum(row[13] + w @ (lib['dc'][nb] - lib['dc'][nb][:, 2:3]), 0.0)
        return pa, ca

    def _plan(self, obs, soc, cap, pmax):""")
rep("""        known = self.hist.known
        for j, (p, c) in enumerate(known[:H]):
            prices[j], co2s[j] = p, c""", """        known = self.hist.known
        if self.analog is not None and len(known) >= 3:
            try:
                pa, ca = self._analog(Xm[0])
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(ca)):
                    aw = self.ANALOG_W
                    prices[3:] = (1 - aw) * prices[3:] + aw * pa[3:]
                    co2s[3:] = (1 - aw) * co2s[3:] + aw * ca[3:]
            except Exception:
                pass
        for j, (p, c) in enumerate(known[:H]):
            prices[j], co2s[j] = p, c""")
(HD / 'agent_v7.py').write_text(s, encoding='utf-8')
print('patch7 ok')
