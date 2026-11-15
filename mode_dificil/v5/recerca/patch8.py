"""agent_v7.py -> agent_v8.py: conditional quantile scenarios for the current hour."""
from pathlib import Path

HD = Path(__file__).parent
s = (HD / 'agent_v7.py').read_text(encoding='utf-8')


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("""    ANALOG_W = 0.6""", """    Q0_SCALE = 1.25           # amplada dels quantils condicionals de l'hora actual
    N_Q0 = 9
    ANALOG_W = 0.6""")
rep("""                     + [f'n{i}' for i in range(len(self.prior['nbands']))])""",
    """                     + [f'n{i}' for i in range(len(self.prior['nbands']))] + [f'q{j}' for j in range(self.N_Q0)])""")
rep("""        # final de la simulacio: no te sentit guardar energia per a hores que no existiran""",
    """        # hora actual: quantils de l'error condicionats a la situacio (nuvols, rampes...), si no es nit coneguda
        if not (night[0] and not math.isnan(load_l1)) and all(f'q{j}' in self.models for j in range(self.N_Q0)):
            try:
                x0 = np.ascontiguousarray(Xn[0:1])
                qv = sorted(self.Q0_SCALE * float(self.models[f'q{j}'].predict(x0, num_threads=1)[0]) for j in range(self.N_Q0))
                if all(math.isfinite(v) for v in qv):
                    netss[0] = [float(npred[0]) + v for v in qv]
            except Exception:
                pass
        # final de la simulacio: no te sentit guardar energia per a hores que no existiran""")
(HD / 'agent_v8.py').write_text(s, encoding='utf-8')
print('patch8 ok')
