from e60 import *  # noqa
import hard

MM = {p: models_for(p) for p in POS}
rows = []
for p in POS:
    seq = SEQS[p]
    FCv1 = fc_for(p)
    fr = {'oracle': run_detail(hard.Oracle(seq), seq),
          'v1': run_detail(FastAgent(FCv1, MM[p], K=15, res_scale=2.0), seq),
          'v5': run_detail(QAgent(research_fc7(p), MM[p], q0_models(p), seq_feats6(p)['Xn'], 1.25, qtab=qtab_hr(p, oofnet6(p, NS, NTAG, PS)), K=15, res_scale=1.0), seq)}
    o = fr['oracle']
    for b, gb in o.groupby('block'):
        idx = gb.index
        wb = float((gb.w * gb.base).sum())
        r = dict(pos=p, ini=str(gb.ts.iloc[0])[:10], net=round(gb.net.clip(lower=0).mean()), exc_dia=round((-gb.net.clip(upper=0)).sum() / 7))
        for k, f in fr.items():
            r[k] = round(100 * f.gain.loc[idx].sum() / wb, 1)
            r['exp_' + k] = round(f.exp.loc[idx].sum() / 7)
        rows.append(r)
df = pd.DataFrame(rows)
df['v5/or'] = (df.v5 / df.oracle).round(2); df['v1/or'] = (df.v1 / df.oracle).round(2)
print(df.sort_values('exc_dia').to_string(index=False))
print('correlacio excedent vs (oracle - v5):', round(np.corrcoef(df.exc_dia, df.oracle - df.v5)[0, 1], 2))
