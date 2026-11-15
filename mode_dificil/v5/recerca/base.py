"""Hard-mode development base: data as the server builds it, official (new) env, calendar, evaluation."""
import math, sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

SP = Path(r'C:\Users\Núria\AppData\Local\Temp\claude\3ab9b5b62eda92f3\73fe9ce0-e95f-4694-8210-72e4d2985bdb\scratchpad')
REPO = Path(r'C:\Users\Núria\Documents\GitHub\Datathon-FME')
sys.path.insert(0, str(REPO))
from eval_local import construir_dataset_horari  # noqa: E402

_spec = importlib.util.spec_from_file_location('env_hard_official', SP / 'env_hard_official.py')
_env = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_env)
CampusBatteryEnv = _env.CampusBatteryEnv

CAP, PMAX, ETA = 6000.0, 1500.0, math.sqrt(0.9)
SPEC = {"capacity_kwh": CAP, "power_kw": PMAX, "efficiency": 0.90, "soc_init": 0.50, "soc_min": 0.0,
        "soc_max": 1.0, "dt_hours": 1.0, "forecast_horizon": 3}
METEO = {'Atersa - IRRADIANCIA (W/m2)': 'irradiancia_wm2', 'Atersa - TEMPERATURA AMBIENTE (SENSOR 1) (ºC)': 'temp_ambient_c',
         'Atersa - TEMPERATURA PANEL (SENSOR 1) (ºC)': 'temp_panel_c', 'Atersa - VELOCIDAD VIENTO (SENSOR 1) (m/s)': 'vent_ms'}
CACHE = SP / 'hd' / 'hard_hourly.pkl'


def dataset():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    d = construir_dataset_horari().set_index('Fecha')
    m = pd.read_csv(REPO / 'data' / 'train' / 'meteo.csv', usecols=['Fecha', *METEO], parse_dates=['Fecha'])
    m = m.rename(columns=METEO).set_index('Fecha').sort_index()
    for c in METEO.values():
        m[c] = pd.to_numeric(m[c], errors='coerce')
    d = d.join(m.resample('1h').mean(), how='left')
    d['pv_kw'] = d['pv_kw'] * 12.0
    d = d.reset_index()
    d.to_pickle(CACHE)
    return d


# ------------------------------------------------------------- calendar (Catalonia / Barcelona / UPC 2025-26)
FESTIUS = {'2025-09-11', '2025-09-24', '2025-10-12', '2025-11-01', '2025-12-06', '2025-12-08', '2025-12-25', '2025-12-26',
           '2026-01-01', '2026-01-06', '2026-04-03', '2026-04-06', '2026-05-01', '2026-05-25', '2026-06-24',
           '2026-08-15', '2026-09-11'}
VACANCES = [('2025-12-24', '2026-01-06'), ('2026-03-30', '2026-04-06'), ('2026-08-01', '2026-08-31')]


def tipus_dia(ts):
    """0 laborable, 1 laborable de vacances, 2 dissabte, 3 diumenge o festiu."""
    ts = pd.Timestamp(ts)
    s = ts.strftime('%Y-%m-%d')
    if s in FESTIUS or ts.dayofweek == 6:
        return 3
    if ts.dayofweek == 5:
        return 2
    for a, b in VACANCES:
        if a <= s <= b:
            return 1
    return 0


# ------------------------------------------------------------- evaluation sequences
def month_seq(data, pos='last'):
    out = []
    for _, g in data.groupby(data.Fecha.dt.to_period('M')):
        g = g.reset_index(drop=True)
        f = g.Fecha
        c = [i for i in range(len(g) - 167) if f[i].hour == 0 and f[i + 167] == f[i] + pd.Timedelta(hours=167)]
        if c:
            i = c[-1] if pos == 'last' else c[0] if pos == 'first' else c[len(c) // 2]
            out.append(g.iloc[i:i + 168])
    return pd.concat(out, ignore_index=True)


def run(agent, block, xp=0.0):
    cols = ['Fecha', 'load_kw', 'pv_kw', 'precio_eur_per_kwh', 'co2_kg_per_kwh', *METEO.values()]
    env = CampusBatteryEnv(block[cols], spec=SPEC, export_price=xp)
    o = env.reset()
    t0 = time.perf_counter()
    while o is not None:
        o, _ = env.step(agent.act(o))
    el = time.perf_counter() - t0
    d = env.df
    pr, co2 = d.precio_eur_per_kwh.to_numpy(), d.co2_kg_per_kwh.to_numpy()
    base = np.maximum((d.load_kw - d.pv_kw).to_numpy(), 0.0)
    g, e = np.asarray(env.grid), np.asarray(env.exports)
    cb, kb, ca, ka = base @ co2, base @ pr, g @ co2, g @ pr - xp * e.sum()
    c, k = 100 * (cb - ca) / cb, 100 * (kb - ka) / kb
    inf = env.clipped + env.violations
    return dict(score=max(0, 0.7 * c + 0.3 * k - 0.2 * inf), co2=c, cost=k, inf=inf, t=el, export_mwh=e.sum() / 1000)


def true_weights(block):
    net = (block.load_kw - block.pv_kw).clip(lower=0).to_numpy()
    co2, pr = block.co2_kg_per_kwh.to_numpy(), block.precio_eur_per_kwh.to_numpy()
    return 100 * (0.7 * co2 / float(net @ co2) + 0.3 * pr / float(net @ pr))
