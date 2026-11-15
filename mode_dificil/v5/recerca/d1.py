from lab import *  # noqa
import hard

MM = {p: models_for(p) for p in POS}
FC = {p: fc_for(p) for p in POS}
cur = compare('ACTUAL (m3, K15, rs2)', lambda p: FastAgent(FC[p], MM[p], K=15, res_scale=2.0))
orc = compare('ORACLE', lambda p: hard.Oracle(SEQS[p]))
A = pd.concat([cur[p].assign(pos=p) for p in POS]); O = pd.concat([orc[p].assign(pos=p) for p in POS])
A['reg'] = (O.gain - A.gain) / 3.0
A['regc'] = (A.wc * (A.grid - O.grid)) / 3.0; A['regk'] = (A.wk * (A.grid - O.grid)) / 3.0
A['waste'] = A.w * A.exp / 3.0
A['ograd'] = O.act.to_numpy()
print('regret total (punts, mitjana 3 seq):', round(A.reg.sum(), 3), '| part CO2:', round(A.regc.sum(), 3), '| part cost:', round(A.regk.sum(), 3),
      '| valor energia abocada:', round(A.waste.sum(), 3))
print('oracle mix cost/co2:', round(np.mean([orc[p].attrs['cost'] / orc[p].attrs['co2'] for p in POS]), 3),
      '| actual:', round(np.mean([cur[p].attrs['cost'] / cur[p].attrs['co2'] for p in POS]), 3))
h = A.groupby('hour').agg(regret=('reg', 'sum'), waste=('waste', 'sum'), act=('act', 'mean'), oact=('ograd', 'mean'), exp=('exp', 'sum'))
print(h.round(3).T.to_string())
print('per dia dins del bloc:'); print(A.groupby('dib').agg(regret=('reg', 'sum'), waste=('waste', 'sum')).round(3).T.to_string())
A['mes'] = A.ts.dt.month
print('per mes:'); print(A.groupby('mes').agg(regret=('reg', 'sum'), waste=('waste', 'sum')).round(3).T.to_string())
A['wd'] = A.ts.dt.dayofweek
print('per dia setmana:'); print(A.groupby('wd').agg(regret=('reg', 'sum'), waste=('waste', 'sum')).round(3).T.to_string())
