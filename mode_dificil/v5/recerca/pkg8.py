"""v8 = v7 + conditional quantiles for the current hour. Equivalence test + final export."""
import sys
from e60 import *  # noqa

FILES8 = FILES7 + [f'model_q{j}.txt' for j in range(9)]
QSC = 1.25


def write8(out_dir, pos):
    out_dir, q = write6(out_dir, pos, agent_src=HD / 'agent_v8.py')
    np.savez_compressed(out_dir / 'analog.npz', **analog_lib(pos))
    for j, m in enumerate(q0_models(pos)[0]):
        (out_dir / f'model_q{j}.txt').write_text(m.booster_.model_to_string(), encoding='utf-8')
    return out_dir, q


if __name__ == '__main__':
    if len(sys.argv) > 1:
        OUT = Path(sys.argv[1])
        write8(OUT, 'ALL')
        zp = OUT / 'submission.zip'
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
            for n in FILES8:
                z.write(OUT / n, n)
        print('paquet:', round(zp.stat().st_size / 1e6, 2), 'MB', len(FILES8), 'fitxers')
        mod = load_agent(OUT, 'final8')
        seq = pd.concat([SEQS['last'], SEQS['first'].iloc[:168]], ignore_index=True)
        a = mod.BatteryAgent()
        r = run_detail(a, seq)
        print(f"prova funcional {len(seq)} h: models={len(a.models)} analegs={'si' if a.analog is not None else 'NO'} inf={r.attrs['inf']} temps={r.attrs['t']:.1f}s")
        sys.exit()
    pos = 'last'
    M = models_for(pos)
    OUT, q = write8(HD / 'equiv8', pos)
    mod = load_agent(OUT, 'ag8')
    seq = SEQS[pos]
    A = type('A', (mod.BatteryAgent,), {'RATIO_CO2_COST': M['rc'] / M['rk'], 'N_TOTAL': 10 ** 9})
    r1 = run_detail(QAgent(research_fc7(pos), M, q0_models(pos), seq_feats6(pos)['Xn'], QSC, qtab=q, K=15, res_scale=1.0), seq)
    a = A()
    r2 = run_detail(a, seq)
    d = (r1.act - r2.act).abs().to_numpy()
    print('recerca:', round(r1.attrs['score'], 4), '| agent.py autonom:', round(r2.attrs['score'], 4), '| passos amb accio diferent:',
          int((d > 1e-6).sum()), '| models:', len(a.models), '| temps:', round(r2.attrs['t'], 1), 's per', len(seq), 'h')
