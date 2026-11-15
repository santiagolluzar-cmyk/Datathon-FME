"""Runtime feature builder (shared by training-data generation and the agent -> no train/serve skew).

At decision time t the agent knows:
  - price/co2 for hours t, t+1, t+2
  - load, pv, meteo of hour t-1 (only valid if the previous obs was exactly 1h earlier)
  - its own history inside the current continuous block
"""
import math
from datetime import timedelta

H = 24
FESTIUS = {'2025-09-11', '2025-09-24', '2025-10-12', '2025-11-01', '2025-12-06', '2025-12-08', '2025-12-25', '2025-12-26',
           '2026-01-01', '2026-01-06', '2026-04-03', '2026-04-06', '2026-05-01', '2026-05-25', '2026-06-24',
           '2026-08-15', '2026-09-11'}
VACANCES = [('2025-12-24', '2026-01-06'), ('2026-03-30', '2026-04-06'), ('2026-08-01', '2026-08-31')]
NAN = float('nan')


VERSION = 'v3'
# Festius nacionals no substituibles: 'vall' tot el dia a la tarifa 2.0TD (afecten preu i demanda peninsular)
FESTIUS_NAC = {'2025-10-12', '2025-11-01', '2025-12-06', '2025-12-08', '2025-12-25', '2026-01-01', '2026-01-06',
               '2026-05-01', '2026-08-15'}
_TM_CACHE = {}


def tipus_mercat(ts):
    """0 laborable, 1 dissabte, 2 diumenge o festiu nacional (per a preu i CO2)."""
    key = (ts.year, ts.month, ts.day)
    v = _TM_CACHE.get(key)
    if v is None:
        s = ts.strftime('%Y-%m-%d')
        v = 2 if (ts.weekday() == 6 or s in FESTIUS_NAC) else (1 if ts.weekday() == 5 else 0)
        _TM_CACHE[key] = v
    return v


_TD_CACHE = {}


def tipus_dia(ts):
    key = (ts.year, ts.month, ts.day)
    v = _TD_CACHE.get(key)
    if v is None:
        v = _TD_CACHE[key] = _tipus_dia(ts)
    return v


def _tipus_dia(ts):
    s = ts.strftime('%Y-%m-%d')
    if s in FESTIUS or ts.weekday() == 6:
        return 3
    if ts.weekday() == 5:
        return 2
    for a, b in VACANCES:
        if a <= s <= b:
            return 1
    return 0


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return NAN
    return x if math.isfinite(x) else NAN


class History:
    """Per-block memory keyed by hour index relative to block start."""

    def __init__(self, prior):
        self.prior = prior  # dict of profile tables (see build_prior)
        self.reset()

    def reset(self):
        self.t0 = None
        self.k = -1          # hours since block start
        self.vals = {}       # k -> dict(load, pv, irr, tamb, tpan, wind, price, co2)
        self.last_ts = None

    def update(self, obs):
        ts = obs['timestamp']
        ts = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
        valid_prev = self.last_ts is not None and ts - self.last_ts == timedelta(hours=1)
        if not valid_prev:
            self.reset()
            self.t0 = ts
        self.k = int(round((ts - self.t0).total_seconds() / 3600))
        self.last_ts = ts
        self.ts = ts
        nd = obs.get('noves_dades_5min') or {}
        if valid_prev:
            d = self.vals.setdefault(self.k - 1, {})
            d.update(load=_f(nd.get('ultim_consum_kw')), pv=_f(nd.get('ultima_solar_kw')),
                     irr=_f(nd.get('irradiancia_wm2')), tamb=_f(nd.get('temp_ambient_c')),
                     tpan=_f(nd.get('temp_panel_c')), wind=_f(nd.get('vent_ms')))
        P = list(obs.get('preus_properes_24h') if obs.get('preus_properes_24h') is not None else [])
        C = list(obs.get('co2_properes_24h') if obs.get('co2_properes_24h') is not None else [])
        p0, c0 = _f(obs.get('preu_actual')), _f(obs.get('co2_actual'))
        if not P or abs(_f(P[0]) - p0) > 1e-9:
            P, C = [p0] + P, [c0] + C
        self.known = []
        for j, (p, c) in enumerate(zip(P, C)):
            p, c = _f(p), _f(c)
            if math.isnan(p) or math.isnan(c):
                break
            self.vals.setdefault(self.k + j, {}).update(price=p, co2=c)
            self.known.append((p, c))
        return valid_prev

    def get(self, k, key):
        d = self.vals.get(k)
        return NAN if d is None else d.get(key, NAN)


def build_prior(df):
    """Profile tables from training data (pandas DataFrame with Fecha, load_kw, pv_kw, price, co2, irr...)."""
    import pandas as pd
    d = df.copy()
    d['h'] = d.Fecha.dt.hour
    d['m'] = d.Fecha.dt.month
    tds = [tipus_dia(t) for t in d.Fecha.dt.to_pydatetime()]
    tms = [tipus_mercat(t) for t in d.Fecha.dt.to_pydatetime()]
    d['doy'] = d.Fecha.dt.dayofyear
    prior = {}
    for col, key in (('load_kw', 'load'), ('pv_kw', 'pv'), ('precio_eur_per_kwh', 'price'), ('co2_kg_per_kwh', 'co2'), ('irradiancia_wm2', 'irr')):
        d['td'] = tms if key in ('price', 'co2') else tds
        g1 = d.groupby(['m', 'td', 'h'])[col].mean()
        g2 = d.groupby(['td', 'h'])[col].mean()
        g3 = d.groupby(['m', 'h'])[col].mean()
        prior[key] = {'mth': {f'{a}_{b}_{c}': float(v) for (a, b, c), v in g1.items()},
                      'th': {f'{b}_{c}': float(v) for (b, c), v in g2.items()},
                      'mh': {f'{a}_{c}': float(v) for (a, c), v in g3.items()}}
    # clear-sky envelope of irradiance by day-of-year (window) and hour
    cs = {}
    for h in range(24):
        sub = d[d.h == h][['doy', 'irradiancia_wm2']].dropna()
        for doy in range(1, 367):
            dist = ((sub.doy - doy + 183) % 366 - 183).abs()
            vals = sub.irradiancia_wm2[dist <= 20]
            if len(vals) < 5:
                vals = sub.irradiancia_wm2[dist <= 45]
            cs[f'{doy}_{h}'] = float(vals.quantile(0.95)) if len(vals) else 0.0
    prior['cs'] = cs
    return prior


def prof(prior, key, ts):
    cache = prior.setdefault('_cache', {})
    ck = (key, ts.year, ts.month, ts.day, ts.hour)
    v = cache.get(ck)
    if v is None:
        v = cache[ck] = _prof(prior, key, ts)
    return v


def _prof(prior, key, ts):
    td = tipus_mercat(ts) if key in ('price', 'co2') else tipus_dia(ts)
    m, h = ts.month, ts.hour
    t = prior[key]
    v = t['mth'].get(f'{m}_{td}_{h}')
    if v is None:  # e.g. November or a day type unseen in that month: neighbour months
        vs = [t['mth'].get(f'{mm}_{td}_{h}') for mm in ((m - 2) % 12 + 1, m % 12 + 1)]
        vs = [x for x in vs if x is not None]
        v = sum(vs) / len(vs) if vs else t['th'].get(f'{td}_{h}', NAN)
    return v


def clear_sky(prior, ts):
    return prior['cs'].get(f'{ts.timetuple().tm_yday}_{ts.hour}', 0.0)


NET_NAMES = ['h', 'hour', 'dow', 'td', 'month', 'cur_hour', 'prof_load', 'prof_pv', 'cs', 'cs_prev',
             'load_l1', 'load_l2', 'load_y', 'pv_l1', 'pv_y', 'irr_l1', 'k_l1', 'k_mean3', 'k_today', 'tamb_l1',
             'lratio_l1', 'lratio_today', 'hours_in_block',
             'pv_persist', 'pv_per_irr', 'load_ramp', 'night_known', 'tpan_l1']
MKT_NAMES = ['h', 'hour', 'dow', 'td', 'month', 'cur_hour', 'prof_price', 'prof_co2', 'p0', 'p1', 'p2', 'c0', 'c1', 'c2',
             'p_y', 'c_y', 'p_y2', 'c_y2', 'p_mean24', 'c_mean24', 'p_dev', 'c_dev', 'cs', 'k_today', 'irr_l1', 'wind_l1',
             'hours_in_block']


def features(hist):
    """Returns (net_rows, mkt_rows), each a list of H feature lists for target hours t..t+H-1."""
    ts, k, pr = hist.ts, hist.k, hist.prior
    g = hist.get
    load_l1, load_l2 = g(k - 1, 'load'), g(k - 2, 'load')
    pv_l1, irr_l1, tamb_l1, wind_l1 = g(k - 1, 'pv'), g(k - 1, 'irr'), g(k - 1, 'tamb'), g(k - 1, 'wind')
    t1 = ts - timedelta(hours=1)
    cs_prev = clear_sky(pr, t1)
    k_l1 = irr_l1 / cs_prev if cs_prev > 50 and not math.isnan(irr_l1) else NAN
    ks = []
    for j in (1, 2, 3):
        csj = clear_sky(pr, ts - timedelta(hours=j))
        ij = g(k - j, 'irr')
        if csj > 50 and not math.isnan(ij):
            ks.append(ij / csj)
    k_mean3 = sum(ks) / len(ks) if ks else NAN
    # today's clearness and load ratio so far
    kt, lr = [], []
    for j in range(1, ts.hour + 1):
        tj = ts - timedelta(hours=j)
        csj = clear_sky(pr, tj)
        ij, lj = g(k - j, 'irr'), g(k - j, 'load')
        if csj > 50 and not math.isnan(ij):
            kt.append(ij / csj)
        pl = prof(pr, 'load', tj)
        if not math.isnan(lj) and pl and pl > 1:
            lr.append(lj / pl)
    k_today = sum(kt) / len(kt) if kt else NAN
    lratio_today = sum(lr) / len(lr) if lr else NAN
    pl1 = prof(pr, 'load', t1)
    lratio_l1 = load_l1 / pl1 if pl1 and pl1 > 1 and not math.isnan(load_l1) else NAN
    known = hist.known
    p = [x[0] for x in known] + [NAN] * 3
    c = [x[1] for x in known] + [NAN] * 3
    p24 = [g(k - j, 'price') for j in range(1, 25)]
    c24 = [g(k - j, 'co2') for j in range(1, 25)]
    p24v = [x for x in p24 if not math.isnan(x)]
    c24v = [x for x in c24 if not math.isnan(x)]
    p_mean24 = sum(p24v) / len(p24v) if p24v else NAN
    c_mean24 = sum(c24v) / len(c24v) if c24v else NAN
    pp0, pc0 = prof(pr, 'price', ts), prof(pr, 'co2', ts)
    p_dev = p[0] - pp0 if not math.isnan(p[0]) else NAN
    c_dev = c[0] - pc0 if not math.isnan(c[0]) else NAN
    # physical persistence features (multiplicative interactions are hard for trees)
    pv_per_irr = pv_l1 / irr_l1 if (not math.isnan(irr_l1) and irr_l1 > 30 and not math.isnan(pv_l1)) else NAN
    kk = k_l1 if not math.isnan(k_l1) else k_mean3
    tpan_l1 = g(k - 1, 'tpan')
    net_rows, mkt_rows = [], []
    for h in range(H):
        th = ts + timedelta(hours=h)
        td, hr, dow, mo = tipus_dia(th), th.hour, th.weekday(), th.month
        cs = clear_sky(pr, th)
        if not math.isnan(kk) and not math.isnan(pv_per_irr):
            pv_persist = kk * cs * pv_per_irr
        elif cs_prev > 20 and not math.isnan(pv_l1):
            pv_persist = pv_l1 * cs / cs_prev
        else:
            pv_persist = NAN
        load_ramp = (load_l1 + prof(pr, 'load', th) - pl1) if (not math.isnan(load_l1) and pl1 == pl1) else NAN
        # frozen-meter night: hours 1..5 repeat the reading of hour 0 (which repeats hour 23)
        night_known = 1 if (ts.hour <= 5 and hr <= 5 and h <= 5 - ts.hour and not math.isnan(load_l1)) else 0
        net_rows.append([h, hr, dow, td, mo, ts.hour, prof(pr, 'load', th), prof(pr, 'pv', th), cs, cs_prev,
                         load_l1, load_l2, g(k + h - 24, 'load'), pv_l1, g(k + h - 24, 'pv'), irr_l1, k_l1, k_mean3,
                         k_today, tamb_l1, lratio_l1, lratio_today, min(k, 200),
                         pv_persist, pv_per_irr, load_ramp, night_known, tpan_l1])
        tm = tipus_mercat(th)
        mkt_rows.append([h, hr, dow, tm, mo, ts.hour, prof(pr, 'price', th), prof(pr, 'co2', th), p[0], p[1], p[2],
                         c[0], c[1], c[2], g(k + h - 24, 'price'), g(k + h - 24, 'co2'), g(k + h - 48, 'price'),
                         g(k + h - 48, 'co2'), p_mean24, c_mean24, p_dev, c_dev, cs, k_today, irr_l1, wind_l1,
                         min(k, 200)])
    return net_rows, mkt_rows


MKT2_NAMES = MKT_NAMES + ['bp', 'bc', 'y_ok', 'p_lvl', 'c_lvl', 'p_lvl2', 'c_lvl2']
MKT3_NAMES = MKT2_NAMES + ['p_shy', 'c_shy', 'p_shp', 'c_shp', 'p_m1', 'c_m1', 'p_m3', 'c_m3', 'p_slope', 'c_slope']


def features_v3(hist):
    """v2 + short-term dynamics: yesterday's / profile's shape anchored on the last known hour, recent trajectory."""
    net_rows, rows = features_v2(hist)
    k, g, ts, pr = hist.k, hist.get, hist.ts, hist.prior
    known = hist.known
    jl = len(known) - 1  # last known offset (normally 2)
    if jl < 0:
        return net_rows, [r + [NAN] * 10 for r in rows]
    pl, cl = known[jl]
    tl = ts + timedelta(hours=jl)
    ypl, ycl = g(k + jl - 24, 'price'), g(k + jl - 24, 'co2')
    ppl, pcl = prof(pr, 'price', tl), prof(pr, 'co2', tl)
    p_m1, c_m1 = g(k - 1, 'price'), g(k - 1, 'co2')
    p_m3, c_m3 = g(k - 3, 'price'), g(k - 3, 'co2')
    p_slope = (known[jl][0] - known[0][0]) / jl if jl > 0 else NAN
    c_slope = (known[jl][1] - known[0][1]) / jl if jl > 0 else NAN
    out = []
    for h, r in enumerate(rows):
        th = ts + timedelta(hours=h)
        yp, yc = g(k + h - 24, 'price'), g(k + h - 24, 'co2')
        p_shy = pl + (yp - ypl) if not (math.isnan(yp) or math.isnan(ypl)) else NAN
        c_shy = cl + (yc - ycl) if not (math.isnan(yc) or math.isnan(ycl)) else NAN
        p_shp = pl + (prof(pr, 'price', th) - ppl)
        c_shp = cl + (prof(pr, 'co2', th) - pcl)
        out.append(r + [p_shy, c_shy, p_shp, c_shp, p_m1, c_m1, p_m3, c_m3, p_slope, c_slope])
    return net_rows, out


def blend(hist, th, h):
    """Heuristic baseline for target hour th: 0.5*yesterday + 0.5*profile (profile if no yesterday)."""
    yp, yc = hist.get(hist.k + h - 24, 'price'), hist.get(hist.k + h - 24, 'co2')
    ap, ac = prof(hist.prior, 'price', th), prof(hist.prior, 'co2', th)
    if math.isnan(yp) or math.isnan(yc):
        return ap, ac, 0
    return 0.5 * yp + 0.5 * ap, 0.5 * yc + 0.5 * ac, 1


def features_v2(hist):
    net_rows, mkt_rows = features(hist)
    k, g = hist.k, hist.get
    known = hist.known
    # level shift of the known hours versus the same hours yesterday
    dp_, dc_ = [], []
    for j, (p, c) in enumerate(known):
        yp, yc = g(k + j - 24, 'price'), g(k + j - 24, 'co2')
        if not math.isnan(yp):
            dp_.append(p - yp); dc_.append(c - yc)
    p_lvl = sum(dp_) / len(dp_) if dp_ else NAN
    c_lvl = sum(dc_) / len(dc_) if dc_ else NAN
    # level of the known hours versus profile
    dp2, dc2 = [], []
    for j, (p, c) in enumerate(known):
        tj = hist.ts + timedelta(hours=j)
        dp2.append(p - prof(hist.prior, 'price', tj)); dc2.append(c - prof(hist.prior, 'co2', tj))
    p_lvl2 = sum(dp2) / len(dp2) if dp2 else NAN
    c_lvl2 = sum(dc2) / len(dc2) if dc2 else NAN
    out = []
    for h, row in enumerate(mkt_rows):
        th = hist.ts + timedelta(hours=h)
        bp, bc, ok = blend(hist, th, h)
        out.append(row + [bp, bc, ok, p_lvl, c_lvl, p_lvl2, c_lvl2])
    return net_rows, out


MKT4_NAMES = MKT3_NAMES + ['pam6', 'cam6', 'pam12', 'cam12', 'pam24', 'cam24', 'p_dd', 'c_dd', 'p_d1', 'c_d1',
                           'p_ay', 'c_ay', 'p_ay2', 'c_ay2', 'p_adm', 'c_adm', 'n_days', 'p_shm', 'c_shm', 'tm_y',
                           'p_pla', 'c_pla', 'p_pay', 'c_pay']


def features_v4(hist):
    """v3 + anomaly features (value - profile of its own day type), multi-day analog shapes, level-adjusted profiles."""
    net_rows, rows = features_v3(hist)
    k, g, ts, pr = hist.k, hist.get, hist.ts, hist.prior
    kn = hist.known
    nk = len(kn)
    jl = nk - 1

    def anom(kk, key):
        v = g(kk, key)
        if math.isnan(v):
            return NAN
        return v - prof(pr, key, ts + timedelta(hours=kk - k))

    A = {key: [anom(k + 2 - j, key) for j in range(24)] for key in ('price', 'co2')}

    def mfirst(key, n):
        xs = [x for x in A[key][:n] if not math.isnan(x)]
        return sum(xs) / len(xs) if xs else NAN

    am = [mfirst('price', 6), mfirst('co2', 6), mfirst('price', 12), mfirst('co2', 12), mfirst('price', 24), mfirst('co2', 24)]
    if nk >= 3:
        dd_p, dd_c = kn[2][0] - 2 * kn[1][0] + kn[0][0], kn[2][1] - 2 * kn[1][1] + kn[0][1]
    else:
        dd_p = dd_c = NAN
    if nk >= 2:
        d1_p, d1_c = kn[-1][0] - kn[-2][0], kn[-1][1] - kn[-2][1]
    else:
        d1_p = d1_c = NAN
    out = []
    for h, r in enumerate(rows):
        th = ts + timedelta(hours=h)
        ay_p, ay_c = anom(k + h - 24, 'price'), anom(k + h - 24, 'co2')
        ay2_p, ay2_c = anom(k + h - 48, 'price'), anom(k + h - 48, 'co2')
        sp = sc = ip = ic = 0.0
        n = ni = 0
        for d in range(1, 7):
            vp, vc = g(k + h - 24 * d, 'price'), g(k + h - 24 * d, 'co2')
            if not math.isnan(vp):
                td_ = th - timedelta(days=d)
                n += 1
                sp += vp - prof(pr, 'price', td_)
                sc += vc - prof(pr, 'co2', td_)
                if jl >= 0:
                    b_p, b_c = g(k + jl - 24 * d, 'price'), g(k + jl - 24 * d, 'co2')
                    if not math.isnan(b_p):
                        ni += 1
                        ip += vp - b_p
                        ic += vc - b_c
        adm_p, adm_c = (sp / n, sc / n) if n else (NAN, NAN)
        shm_p, shm_c = (kn[jl][0] + ip / ni, kn[jl][1] + ic / ni) if ni else (NAN, NAN)
        pp, pc = prof(pr, 'price', th), prof(pr, 'co2', th)
        out.append(r + am + [dd_p, dd_c, d1_p, d1_c, ay_p, ay_c, ay2_p, ay2_c, adm_p, adm_c, n, shm_p, shm_c,
                             tipus_mercat(th - timedelta(hours=24)), pp + am[0], pc + am[1], pp + ay_p, pc + ay_c])
    return net_rows, out


NET6_EXTRA = ['load_l3', 'pv_l2', 'pv_l3', 'irr_l2', 'irr_l3', 'k_l2', 'dk', 'dload1', 'dload2', 'la1', 'la2', 'pvk1', 'pvk2',
              'dtamb', 'cs_ratio', 'pv_pk', 'la_y', 'load_pa', 'pvk_y', 'net_naive']
NET6_NAMES = NET_NAMES + NET6_EXTRA


def features_v6(hist):
    """v4 market rows + net rows extended with short-term dynamics (for horizon-specialised net models)."""
    net_rows, mkt_rows = features_v4(hist)
    k, g, ts, pr = hist.k, hist.get, hist.ts, hist.prior
    t1, t2 = ts - timedelta(hours=1), ts - timedelta(hours=2)
    cs1, cs2 = clear_sky(pr, t1), clear_sky(pr, t2)
    l1, l2, l3 = g(k - 1, 'load'), g(k - 2, 'load'), g(k - 3, 'load')
    v1, v2, v3 = g(k - 1, 'pv'), g(k - 2, 'pv'), g(k - 3, 'pv')
    i1, i2, i3 = g(k - 1, 'irr'), g(k - 2, 'irr'), g(k - 3, 'irr')
    k1 = i1 / cs1 if cs1 > 50 else NAN
    k2 = i2 / cs2 if cs2 > 50 else NAN
    pvk1 = v1 / cs1 if cs1 > 50 else NAN
    pvk2 = v2 / cs2 if cs2 > 50 else NAN
    la1 = l1 - prof(pr, 'load', t1)
    la2 = l2 - prof(pr, 'load', t2)
    base = [l3, v2, v3, i2, i3, k2, k1 - k2, l1 - l2, l2 - l3, la1, la2, pvk1, pvk2, g(k - 1, 'tamb') - g(k - 2, 'tamb')]
    out = []
    for h, r in enumerate(net_rows):
        th = ts + timedelta(hours=h)
        cs = clear_sky(pr, th)
        ty = th - timedelta(hours=24)
        csy = clear_sky(pr, ty)
        pl = prof(pr, 'load', th)
        pv_pk = pvk1 * cs
        load_pa = pl + la1
        vy = g(k + h - 24, 'pv')
        out.append(r + base + [cs / cs1 if cs1 > 20 else NAN, pv_pk, g(k + h - 24, 'load') - prof(pr, 'load', ty), load_pa,
                               vy / csy if csy > 50 else NAN, load_pa - pv_pk])
    return out, mkt_rows
