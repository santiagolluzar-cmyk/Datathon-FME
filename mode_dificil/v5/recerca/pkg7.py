"""v7 = v6 + analog market blend. Package writer + equivalence test + final export."""
import sys
from pkg6 import *  # noqa

AW, AK = 0.6, 16
FILES7 = FILES + ['analog.npz']


def mfeat(x):
    month = x[:, 4]
    return np.column_stack([x[:, 8] / .04, x[:, 9] / .04, x[:, 10] / .04, x[:, 11] / .04, x[:, 12] / .04, x[:, 13] / .04,
                            (x[:, 10] - x[:, 8]) / .015, (x[:, 13] - x[:, 11]) / .015, x[:, 20] / .04, x[:, 21] / .04,
                            x[:, 18] / .06, x[:, 19] / .06, np.sin(month * 2 * np.pi / 12) * .7, np.cos(month * 2 * np.pi / 12) * .7,
                            (x[:, 2] >= 5) * .5])


def analog_lib(pos):
    g = gen5(pos)
    ok = np.isfinite(g['Yp']).all(axis=1) & (g['meta'][:, 5] >= 3)
    L = g['Xm'][ok][:, 0, :44].astype(float)
    LX = mfeat(L)
    med = np.nanmedian(LX, axis=0)
    LX = np.where(np.isfinite(LX), LX, med)
    return dict(x=LX.astype(np.float32), hour=L[:, 5].astype(np.int16), dp=g['Yp'][ok].astype(np.float32),
                dc=g['Yc'][ok].astype(np.float32), median=med.astype(np.float32))


def analog_fc(pos, lib, k=AK):
    mf = mkt_feat_cache(pos, 'v4')
    X0 = mf['X'][:, 0, :44].astype(float)
    QX = mfeat(X0)
    QX = np.where(np.isfinite(QX), QX, lib['median'])
    T = len(X0)
    P = np.full((T, 24), np.nan); C = np.full((T, 24), np.nan)
    pools = {h: np.flatnonzero(lib['hour'] == h) for h in range(24)}
    for i in range(T):
        if mf['NK'][i] < 3:
            continue
        pool = pools[int(X0[i, 5])]
        d = ((lib['x'][pool] - QX[i]) ** 2).sum(axis=1)
        pick = np.argpartition(d, k - 1)[:k]
        w = 1 / (.2 + d[pick]); w /= w.sum()
        nb = pool[pick]
        P[i] = mf['KP'][i, 2] + w @ (lib['dp'][nb] - lib['dp'][nb][:, 2:3])
        C[i] = np.maximum(mf['KC'][i, 2] + w @ (lib['dc'][nb] - lib['dc'][nb][:, 2:3]), 0.0)
    return P, C


def research_fc7(pos, aw=AW):
    fc = research_fc(pos)
    Pa, Ca = analog_fc(pos, analog_lib(pos))
    ok = np.isfinite(Pa).all(axis=1)
    P, C = fc['P'].copy(), fc['C'].copy()
    P[ok, 3:] = (1 - aw) * P[ok, 3:] + aw * Pa[ok, 3:]
    C[ok, 3:] = (1 - aw) * C[ok, 3:] + aw * Ca[ok, 3:]
    fc['P'], fc['C'] = P, C
    return fc


def write7(out_dir, pos):
    out_dir, q = write6(out_dir, pos, agent_src=HD / 'agent_v7.py')
    np.savez_compressed(out_dir / 'analog.npz', **analog_lib(pos))
    return out_dir, q


if __name__ == '__main__':
    if len(sys.argv) > 1:   # final export
        OUT = Path(sys.argv[1])
        write7(OUT, 'ALL')
        zp = OUT / 'submission.zip'
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
            for n in FILES7:
                z.write(OUT / n, n)
        print('paquet:', round(zp.stat().st_size / 1e6, 2), 'MB', len(FILES7), 'fitxers')
        mod = load_agent(OUT, 'final7')
        seq = pd.concat([SEQS['last'], SEQS['first'].iloc[:168]], ignore_index=True)
        a = mod.BatteryAgent()
        r = run_detail(a, seq)
        print(f"prova funcional {len(seq)} h: models={len(a.models)} analegs={'si' if a.analog is not None else 'NO'} inf={r.attrs['inf']} "
              f"temps={r.attrs['t']:.1f}s")
        sys.exit()
    pos = 'last'
    M = models_for(pos)
    OUT, q = write7(HD / 'equiv7', pos)
    mod = load_agent(OUT, 'ag7')
    seq = SEQS[pos]
    A = type('A', (mod.BatteryAgent,), {'RATIO_CO2_COST': M['rc'] / M['rk'], 'N_TOTAL': 10 ** 9})
    r1 = run_detail(VFAgent(research_fc7(pos), M, qtab=q, K=15, res_scale=1.0), seq)
    a = A()
    r2 = run_detail(a, seq)
    d = (r1.act - r2.act).abs().to_numpy()
    print('recerca:', round(r1.attrs['score'], 4), '| agent.py autonom:', round(r2.attrs['score'], 4), '| passos amb accio diferent:',
          int((d > 1e-6).sum()), '| analegs carregats:', a.analog is not None, '| temps:', round(r2.attrs['t'], 1), 's per', len(seq), 'h')
