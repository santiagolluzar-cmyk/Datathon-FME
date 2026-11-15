"""Detailed runs: per-hour frames, linear score contributions, paired comparisons across the 33 block-weeks."""
import sys
sys.path.insert(0, r'C:\Users\Núria\AppData\Local\Temp\claude\3ab9b5b62eda92f3\73fe9ce0-e95f-4694-8210-72e4d2985bdb\scratchpad')
from exp6 import *  # noqa
from base import CampusBatteryEnv, SPEC, METEO

POS = ('last', 'first', 'mid')


def run_detail(agent, seq):
    cols = ['Fecha', 'load_kw', 'pv_kw', 'precio_eur_per_kwh', 'co2_kg_per_kwh', *METEO.values()]
    env = CampusBatteryEnv(seq[cols], spec=SPEC, export_price=0.0)
    o = env.reset()
    socs = []
    t0 = time.perf_counter()
    while o is not None:
        socs.append(o['soc_kwh'])
        o, _ = env.step(agent.act(o))
    el = time.perf_counter() - t0
    d = env.df
    pr, co2 = d.precio_eur_per_kwh.to_numpy(), d.co2_kg_per_kwh.to_numpy()
    net = (d.load_kw - d.pv_kw).to_numpy()
    base = np.maximum(net, 0.0)
    cb, kb = base @ co2, base @ pr
    w = 100 * (0.7 * co2 / cb + 0.3 * pr / kb)
    wc, wk = 100 * 0.7 * co2 / cb, 100 * 0.3 * pr / kb
    g, e = np.asarray(env.grid), np.asarray(env.exports)
    f = pd.DataFrame({'ts': d.Fecha, 'w': w, 'wc': wc, 'wk': wk, 'net': net, 'base': base, 'grid': g, 'exp': e,
                      'soc': socs, 'act': np.asarray(env.actions)})
    f['gain'] = f.w * (f.base - f.grid)
    f['block'] = (f.ts.diff() != pd.Timedelta(hours=1)).cumsum()
    f['hour'] = f.ts.dt.hour
    f['dib'] = f.groupby('block').cumcount() // 24
    f.attrs.update(score=f.gain.sum(), co2=100 * (cb - g @ co2) / cb, cost=100 * (kb - g @ pr) / kb,
                   inf=env.clipped + env.violations, t=el)
    return f


def fc_for(pos, tag='m3'):
    mc = market_cache(pos, train_mkt3(pos, None, tag), tag)
    fc = dict(build_cache(pos))
    fc['P'], fc['C'] = mc['P'], mc['C']
    return fc


def compare(name, make, ref=None, poss=POS, verbose=True):
    """make(pos) -> agent. Returns dict pos -> frame. Prints mean score and paired diff vs ref (dict of frames)."""
    out = {p: run_detail(make(p), SEQS[p]) for p in poss}
    sc = [out[p].attrs['score'] for p in poss]
    msg = f"{name:44s} " + ' '.join(f"{p}={s:.3f}" for p, s in zip(poss, sc)) + f" | MITJANA={np.mean(sc):.3f}"
    msg += f" co2={np.mean([out[p].attrs['co2'] for p in poss]):.2f} cost={np.mean([out[p].attrs['cost'] for p in poss]):.2f}"
    if ref is not None:
        d = np.concatenate([(out[p].groupby('block').gain.sum() - ref[p].groupby('block').gain.sum()).to_numpy() for p in poss])
        se = d.std(ddof=1) / np.sqrt(len(d))
        msg += f" | dif/bloc vs ref: {d.sum()/len(poss):+.3f} (t={d.mean()/se:+.1f}, millor en {int((d>0).sum())}/{len(d)})"
    msg += f" inf={sum(out[p].attrs['inf'] for p in poss)} t={max(out[p].attrs['t'] for p in poss):.0f}s"
    if verbose:
        print(msg, flush=True)
    return out
