"""One-off patch: append qtab_hr/with_hours to v6.py and create agent_v6.py from agent_v5.py."""
from pathlib import Path

HD = Path(__file__).parent
v6 = (HD / 'v6.py').read_text(encoding='utf-8')
if 'def qtab_hr' not in v6:
    v6 += '''

I_HR = NN.index('hour')


def qtab_hr(pos, net_oof, K=15, min_n=150):
    """Honest residual quantiles by (horizon, hour of day of the target hour)."""
    g = gen6(pos)
    res = g['Yn'] - net_oof
    hr = g['Xn'][:, :, I_HR].astype(int); nk = g['Xn'][:, :, I_NK] > 0.5; day = g['Xn'][:, :, I_CS] > 50
    qs = [(i + 0.5) / K for i in range(K)]
    tab = {}
    for h in range(24):
        for t in range(24):
            m = (hr[:, h] == t) & ~nk[:, h] & np.isfinite(res[:, h])
            r = res[:, h][m]
            if len(r) < min_n:
                dflag = np.median(day[:, h][hr[:, h] == t]) > 0.5 if (hr[:, h] == t).any() else False
                m = (day[:, h] == dflag) & ~nk[:, h] & np.isfinite(res[:, h]); r = res[:, h][m]
            tab[(h, t)] = [float(x) for x in np.quantile(r, qs)]
    return tab


def with_hours(fc):
    fc = dict(fc)
    hrs = pd.Series(fc['ts']).dt.hour.to_numpy()
    fc['DAY'] = (hrs[:, None] + np.arange(24)[None, :]) % 24
    return fc
'''
    (HD / 'v6.py').write_text(v6, encoding='utf-8')

s = (HD / 'agent_v5.py').read_text(encoding='utf-8')


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("""    # anomalies (valor - perfil del seu propi tipus de dia) de les darreres 24 hores conegudes""",
    """    # dinamica a curt termini de la demanda (per als models especialitzats de l'hora actual i seguents)
    t2 = ts - timedelta(hours=2)
    cs2 = clear_sky(pr, t2)
    l3 = g(k - 3, 'load')
    v2, v3 = g(k - 2, 'pv'), g(k - 3, 'pv')
    i2, i3 = g(k - 2, 'irr'), g(k - 3, 'irr')
    kk1 = irr_l1 / cs_prev if cs_prev > 50 else NAN
    kk2 = i2 / cs2 if cs2 > 50 else NAN
    pvk1 = pv_l1 / cs_prev if cs_prev > 50 else NAN
    pvk2 = v2 / cs2 if cs2 > 50 else NAN
    la1 = load_l1 - pl1
    la2 = load_l2 - prof(pr, 'load', t2)
    nbase = [l3, v2, v3, i2, i3, kk2, kk1 - kk2, load_l1 - load_l2, load_l2 - l3, la1, la2, pvk1, pvk2,
             g(k - 1, 'tamb') - g(k - 2, 'tamb')]
    # anomalies (valor - perfil del seu propi tipus de dia) de les darreres 24 hores conegudes""")
rep("""                         k_today, tamb_l1, lratio_l1, lratio_today, min(k, 200),
                         pv_persist, pv_per_irr, load_ramp, night_known, tpan_l1])""",
    """                         k_today, tamb_l1, lratio_l1, lratio_today, min(k, 200),
                         pv_persist, pv_per_irr, load_ramp, night_known, tpan_l1]
                        + nbase + _net_extras(hist, h, th, cs, cs_prev, pvk1, la1))""")
rep("""def _v4_extras(hist, h, th, known, jl, ap, ac, am, anom):""",
    """def _net_extras(hist, h, th, cs, cs_prev, pvk1, la1):
    k, g, pr = hist.k, hist.get, hist.prior
    ty = th - timedelta(hours=24)
    csy = clear_sky(pr, ty)
    pv_pk = pvk1 * cs
    load_pa = prof(pr, 'load', th) + la1
    vy = g(k + h - 24, 'pv')
    return [cs / cs_prev if cs_prev > 20 else NAN, pv_pk, g(k + h - 24, 'load') - prof(pr, 'load', ty), load_pa,
            vy / csy if csy > 50 else NAN, load_pa - pv_pk]


def _v4_extras(hist, h, th, known, jl, ap, ac, am, anom):""")
rep("""            names = ['net'] + [f'{t}{i}' for i in range(len(self.prior['bands'])) for t in ('p', 'c')]""",
    """            names = (['net'] + [f'{t}{i}' for i in range(len(self.prior['bands'])) for t in ('p', 'c')]
                     + [f'n{i}' for i in range(len(self.prior['nbands']))])""")
rep("""        npred = self._predict('net', Xn, fallback_net)""",
    """        npred = np.array(self._predict('net', Xn[:, :28], fallback_net), dtype=float)
        for i, (lo, hi, nc) in enumerate(self.prior['nbands']):   # models de demanda especialitzats per horitzo
            npred[lo:hi + 1] = self._predict(f'n{i}', Xn[lo:hi + 1, :nc], npred[lo:hi + 1])""")
rep("""                q = self.qtab[(h, 1 if net_rows[h][8] > 50 else 0)]""",
    """                q = self.qtab[(h, int(net_rows[h][1]))]   # error tipic per horitzo i hora del dia""")
(HD / 'agent_v6.py').write_text(s, encoding='utf-8')
print('patch ok')
