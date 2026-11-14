"""Compara agents amb l'ORACLE: l'optim amb informacio perfecta de tot el periode (cota superior).

Executar des de l'arrel:  python -B -m experiments.comparar_oracle
- L'oracle coneix preus, CO2 i demanda futurs de tot el bloc i usa els pesos exactes del score
  (0.7/CO2_base, 0.3/Cost_base del mateix bloc). No es lliurable: nomes serveix de sostre.
- Sequencies "tipus test": una setmana continua (168 h) per mes, concatenades, amb SOC i agent
  persistents (com el test ocult: ultima setmana de cada mes). Tambe primera i setmana del mig.
- Tot s'avalua amb starter_kit/environment.py (CampusBatteryEnv), preu d'exportacio 0.
- A "setmanes_independents" l'oracle optimitza cada setmana amb els pesos d'aquella setmana, pero
  el score agregat usa les bases de totes les setmanes juntes: per aixo l'agent hi pot superar el 100%.
"""
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from eval_local import construir_dataset_horari, seleccionar_blocs
from starter_kit.environment import CampusBatteryEnv

ROOT = Path(__file__).resolve().parents[1]
CAP, PMAX = 2000.0, 500.0
ETA = math.sqrt(0.9)


def carregar(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BatteryAgent


class Oracle:
    """DP exacta enrere sobre tot el bloc (funcio de valor convexa lineal a trossos)."""

    def __init__(self, block):
        net = (block.load_kw - block.pv_kw).clip(lower=0).to_numpy()
        co2, price = block.co2_kg_per_kwh.to_numpy(), block.precio_eur_per_kwh.to_numpy()
        w = 0.7 * co2 / float(net @ co2) + 0.3 * price / float(net @ price)
        self.w = w
        self.dmax = np.minimum(PMAX, net) / ETA
        up = PMAX * ETA
        trams = [(0.0, CAP)]
        self.V = [None] * (len(w) + 1)
        self.V[-1] = trams
        for k in range(len(w) - 1, -1, -1):
            nous = sorted(trams + [(-w[k] / ETA, up), (-w[k] * ETA, self.dmax[k])])
            retall, res = up, []
            for s, l in nous:
                if retall > 0:
                    if l <= retall:
                        retall -= l
                        continue
                    l -= retall
                    retall = 0.0
                res.append([s, l])
            retall = self.dmax[k]
            while retall > 1e-12 and res:
                if res[-1][1] <= retall:
                    retall -= res[-1][1]
                    res.pop()
                else:
                    res[-1][1] -= retall
                    retall = 0.0
            trams = [tuple(t) for t in res]
            self.V[k] = trams
        self.k = 0

    def act(self, obs):
        k, soc = self.k, float(obs['soc_kwh'])
        self.k += 1
        w0 = self.w[k]
        b1 = sum(l for s, l in self.V[k + 1] if s < -w0 / ETA)
        b2 = sum(l for s, l in self.V[k + 1] if s < -w0 * ETA)
        if soc < b1:
            p = (min(b1, soc + PMAX * ETA) - soc) / ETA
            return float(max(0.0, min(p, PMAX, (CAP - soc) / ETA - 1e-4)))
        if soc > b2:
            p = -(soc - max(b2, soc - self.dmax[k])) * ETA
            return float(min(0.0, max(p, -PMAX, -(soc * ETA - 1e-4))))
        return 0.0


def simular(make, block):
    agent = make(block)
    env = CampusBatteryEnv(block, export_price=0.0)
    obs = env.reset()
    start = time.perf_counter()
    while obs is not None:
        obs, _ = env.step(agent.act(obs))
    elapsed = time.perf_counter() - start
    d = env.df
    price, co2 = d.precio_eur_per_kwh.to_numpy(), d.co2_kg_per_kwh.to_numpy()
    base = np.maximum((d.load_kw - d.pv_kw).to_numpy(), 0.0)
    grid = np.asarray(env.grid)
    return {'cost_base': float(base @ price), 'cost_agent': float(grid @ price), 'co2_base': float(base @ co2),
            'co2_agent': float(grid @ co2), 'infraccions': env.clipped + env.violations, 'temps_s': elapsed}


def score(raws):
    t = {k: sum(r[k] for r in raws) for k in raws[0]}
    co2 = 100 * (t['co2_base'] - t['co2_agent']) / t['co2_base']
    cost = 100 * (t['cost_base'] - t['cost_agent']) / t['cost_base']
    return max(0.0, 0.7 * co2 + 0.3 * cost - 0.2 * t['infraccions']), co2, cost, int(t['infraccions']), t['temps_s']


def sequencia(data, pos):
    blocs = []
    for _, g in data.groupby(data.Fecha.dt.to_period('M')):
        g = g.reset_index(drop=True)
        f = g.Fecha
        c = [i for i in range(len(g) - 167) if f[i].hour == 0 and f[i + 167] == f[i] + pd.Timedelta(hours=167)]
        if c:
            i = c[-1] if pos == 'ultima' else c[0] if pos == 'primera' else c[len(c) // 2]
            blocs.append(g.iloc[i:i + 168])
    return pd.concat(blocs, ignore_index=True)


def main():
    data = construir_dataset_horari()
    suites = {'setmanes_independents': seleccionar_blocs(data, mode='setmanes'),
              **{f'seq_{p}_setmana_mes': [sequencia(data, p)] for p in ('ultima', 'primera', 'mig')},
              'train_continu': [data]}
    agents = {'oracle (cota superior)': Oracle,
              'agent.py': lambda b, c=carregar(ROOT / 'agent.py'): c(),
              'legacy v5 (linprog, p70)': lambda b, c=carregar(ROOT / 'legacy' / 'v5_mpc_linprog_p70.py'): c()}
    rows = []
    for suite, blocks in suites.items():
        print(f'\n{suite} ({sum(map(len, blocks))} h)', flush=True)
        for name, make in agents.items():
            s, co2, cost, inf, el = score([simular(make, b) for b in blocks])
            rows.append({'suite': suite, 'agent': name, 'score': s, 'co2_pct': co2, 'cost_pct': cost,
                         'infraccions': inf, 'temps_s': el})
            print(f'  {name:26s} score={s:.4f} co2={co2:.3f}% cost={cost:.3f}% infraccions={inf} temps={el:.1f}s', flush=True)
    df = pd.DataFrame(rows)
    piv = df.pivot(index='suite', columns='agent', values='score')
    piv['% de l oracle (agent.py)'] = 100 * piv['agent.py'] / piv['oracle (cota superior)']
    print('\n' + piv.round(4).to_string())
    df.to_csv(ROOT / 'experiments' / 'resultats_oracle.csv', index=False)
    print('\nResultats: experiments/resultats_oracle.csv')


if __name__ == '__main__':
    main()
