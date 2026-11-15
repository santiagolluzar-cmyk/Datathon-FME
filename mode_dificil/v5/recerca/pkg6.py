"""Package writer + equivalence test for the v6 agent."""
import json, shutil, importlib.util, zipfile
from pathlib import Path
from v6 import *  # noqa

SPEC = [('v4', 3, 3, 3, 3), ('v4', 4, 4, 4, 4), ('v4', 5, 5, 5, 5), ('v4', 6, 8, 6, 8), ('v3', 9, 23, 3, 23)]
TAG = 'B'
NS = [(0, 0, N6), (1, 1, N6), (2, 2, N6), (3, 5, N6)]
NTAG = 'n6s'
PS = dict(num_leaves=31, n_estimators=400, min_child_samples=30)
N_ITER_NET, N_ITER_LONG = 300, 250
FILES = (['agent.py', 'prior.json', 'model_net.txt'] + [f'model_{t}{i}.txt' for i in range(len(SPEC)) for t in ('p', 'c')]
         + [f'model_n{i}.txt' for i in range(len(NS))])


def write6(out_dir, pos, agent_src=HD / 'agent_v6.py'):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    M, _ = base_models(pos)
    q = qtab_hr(pos, oofnet6(pos, NS, NTAG, PS), 15)
    prior = {k: v for k, v in M['prior'].items() if k != '_cache'}
    prior['qtab'] = {f'{h}_{t}': v for (h, t), v in q.items()}
    prior['rc_train'], prior['rk_train'] = M['rc'], M['rk']
    prior['bands'] = [[fn, lo, hi] for fn, lo, hi, _, _ in SPEC]
    prior['nbands'] = [[lo, hi, nc] for lo, hi, nc in NS]
    (out_dir / 'prior.json').write_text(json.dumps(prior), encoding='utf-8')
    (out_dir / 'model_net.txt').write_text(M['net'].booster_.model_to_string(), encoding='utf-8')
    for i, md in enumerate(mkt_models5(pos, SPEC, TAG)):
        (out_dir / f'model_p{i}.txt').write_text(md['price'].booster_.model_to_string(), encoding='utf-8')
        (out_dir / f'model_c{i}.txt').write_text(md['co2'].booster_.model_to_string(), encoding='utf-8')
    for i, md in enumerate(net_models6(pos, NS, NTAG, PS)):
        (out_dir / f'model_n{i}.txt').write_text(md['model'].booster_.model_to_string(), encoding='utf-8')
    shutil.copy(agent_src, out_dir / 'agent.py')
    return out_dir, q


def research_fc(pos):
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


def load_agent(d, name='ag6'):
    spec = importlib.util.spec_from_file_location(name, Path(d) / 'agent.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


if __name__ == '__main__':
    pos = 'last'
    M = models_for(pos)
    OUT, q = write6(HD / 'equiv6', pos)
    mod = load_agent(OUT)
    seq = SEQS[pos]
    A = type('A', (mod.BatteryAgent,), {'RATIO_CO2_COST': M['rc'] / M['rk'], 'N_TOTAL': 10 ** 9})
    Bc = type('B', (mod.BatteryAgent,), {'RATIO_CO2_COST': M['rc'] / M['rk'], 'N_TOTAL': len(seq)})
    fc = research_fc(pos)
    r1 = run_detail(VFAgent(fc, M, qtab=q, K=15, res_scale=1.0), seq)
    a = A(); print('models carregats:', len(a.models))
    r2 = run_detail(a, seq)
    r3 = run_detail(Bc(), seq)
    d = (r1.act - r2.act).abs().to_numpy()
    print('recerca:', round(r1.attrs['score'], 4), '| agent.py autonom:', round(r2.attrs['score'], 4), '| passos amb accio diferent:',
          int((d > 1e-6).sum()), 'primer:', (np.where(d > 1e-6)[0][:3]).tolist(), '| temps:', round(r2.attrs['t'], 1), 's per', len(seq), 'h')
    print('amb final de simulacio conegut:', round(r3.attrs['score'], 4), 'inf', r3.attrs['inf'])
