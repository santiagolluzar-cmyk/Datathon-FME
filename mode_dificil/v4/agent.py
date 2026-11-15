"""agent.py — Repte UPC Sostenible, MODE DIFICIL (bateria 6.000 kWh / ±1.500 kW, solar x12, mercat a 3 h).

Estrategia (resum):
1. PREVISIO del mercat (+3..+24 h): heuristica robusta (0,5*ahir + 0,5*perfil mes/tipus de dia/hora)
   corregida per un LightGBM entrenat sobre el seu residu (nivell d'avui vs ahir, hores conegudes, sol...).
2. PREVISIO de la demanda neta (consum - solar x12) per a les properes 24 h amb un LightGBM
   (darrera hora mesurada, irradiancia/index de claredat, perfil, calendari de festius i vacances).
   Nit (0-5 h): el comptador del campus queda congelat -> la demanda d'aquestes hores es coneguda exactament.
3. DECISIO: optimitzador exacte per horitzo lliscant de 24 h (programacio dinamica sobre funcions
   convexes lineals a trossos). La demanda neta entra com a ESCENARIS (quantils de l'error de previsio):
   el cost esperat penalitza naturalment descarregar mes del que el campus consumira (energia abocada)
   i premia carregar amb l'excedent solar gratuit.
4. Pesos del score: w = 0,7*CO2/REF_CO2 + 0,3*preu/REF_COST (ratio de les linies base del mode dificil).
5. Seguretat: clamp fisic amb cap_max_kwh/power_max_kw (0 infraccions); si falta qualsevol model o dada,
   l'agent degrada a previsions heuristiques en lloc de fallar.
"""
import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - fallback if lightgbm is unavailable
    lgb = None

DIR = Path(__file__).parent
NAN = float('nan')
H = 24
ETA = math.sqrt(0.9)

# ----------------------------------------------------------------------------------------- calendari
FESTIUS = {'2025-09-11', '2025-09-24', '2025-10-12', '2025-11-01', '2025-12-06', '2025-12-08', '2025-12-25', '2025-12-26',
           '2026-01-01', '2026-01-06', '2026-04-03', '2026-04-06', '2026-05-01', '2026-05-25', '2026-06-24',
           '2026-08-15', '2026-09-11'}
VACANCES = [('2025-12-24', '2026-01-06'), ('2026-03-30', '2026-04-06'), ('2026-08-01', '2026-08-31')]
# Festius nacionals no substituibles: periode 'vall' tot el dia a la tarifa 2.0TD (preu) i menys demanda peninsular
FESTIUS_NAC = {'2025-10-12', '2025-11-01', '2025-12-06', '2025-12-08', '2025-12-25', '2026-01-01', '2026-01-06',
               '2026-05-01', '2026-08-15'}
_TM, _TD = {}, {}


def tipus_mercat(ts):
    key = (ts.year, ts.month, ts.day)
    v = _TM.get(key)
    if v is None:
        s = ts.strftime('%Y-%m-%d')
        v = _TM[key] = 2 if (ts.weekday() == 6 or s in FESTIUS_NAC) else (1 if ts.weekday() == 5 else 0)
    return v


def tipus_dia(ts):
    """0 laborable, 1 laborable de vacances, 2 dissabte, 3 diumenge o festiu (per al consum del campus)."""
    key = (ts.year, ts.month, ts.day)
    v = _TD.get(key)
    if v is None:
        s = ts.strftime('%Y-%m-%d')
        if s in FESTIUS or ts.weekday() == 6:
            v = 3
        elif ts.weekday() == 5:
            v = 2
        else:
            v = 1 if any(a <= s <= b for a, b in VACANCES) else 0
        _TD[key] = v
    return v


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return NAN
    return x if math.isfinite(x) else NAN


def _to_datetime(ts):
    """Accepta pd.Timestamp, datetime, numpy.datetime64 o text ISO."""
    from datetime import datetime
    if hasattr(ts, 'to_pydatetime'):
        return ts.to_pydatetime().replace(tzinfo=None)
    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)
    if isinstance(ts, np.datetime64):
        return datetime(1970, 1, 1) + timedelta(seconds=int(ts.astype('datetime64[s]').astype('int64')))
    return datetime.fromisoformat(str(ts).replace('T', ' ')[:19])


# ----------------------------------------------------------------------------------------- perfils
def prof(prior, key, ts):
    cache = prior['_cache']
    ck = (key, ts.year, ts.month, ts.day, ts.hour)
    v = cache.get(ck)
    if v is None:
        td = tipus_mercat(ts) if key in ('price', 'co2') else tipus_dia(ts)
        m, h = ts.month, ts.hour
        t = prior[key]
        v = t['mth'].get(f'{m}_{td}_{h}')
        if v is None:  # p.ex. novembre (sense dades de train): mitjana dels mesos veins
            vs = [t['mth'].get(f'{mm}_{td}_{h}') for mm in ((m - 2) % 12 + 1, m % 12 + 1)]
            vs = [x for x in vs if x is not None]
            v = sum(vs) / len(vs) if vs else t['th'].get(f'{td}_{h}', NAN)
        cache[ck] = v
    return v


def clear_sky(prior, ts):
    return prior['cs'].get(f'{ts.timetuple().tm_yday}_{ts.hour}', 0.0)


# ----------------------------------------------------------------------------------------- memoria
class History:
    """Memoria del bloc continu actual (es reinicia als salts temporals del test)."""

    def __init__(self, prior):
        self.prior = prior
        self.reset()

    def reset(self):
        self.t0, self.k, self.vals, self.last_ts = None, -1, {}, None

    def update(self, obs):
        ts = _to_datetime(obs.get('timestamp'))
        gap = None if self.last_ts is None else (ts - self.last_ts).total_seconds() / 3600.0
        cont = gap is not None and 0.0 <= gap <= 6.0   # hores soltes perdudes: no es perd la memoria del bloc
        if not cont:
            self.reset()
            self.t0 = ts
        prev_k = self.k
        self.k = int(round((ts - self.t0).total_seconds() / 3600))
        self.last_ts = self.ts = ts
        nd = obs.get('noves_dades_5min')
        nd = nd if isinstance(nd, dict) else {}
        if cont and gap >= 1.0:  # la telemetria correspon a la fila anterior del simulador
            d = self.vals.setdefault(prev_k, {})
            d.update(load=_f(nd.get('ultim_consum_kw')), pv=_f(nd.get('ultima_solar_kw')),
                     irr=_f(nd.get('irradiancia_wm2')), tamb=_f(nd.get('temp_ambient_c')),
                     tpan=_f(nd.get('temp_panel_c')), wind=_f(nd.get('vent_ms')))
        P = obs.get('preus_properes_24h')
        C = obs.get('co2_properes_24h')
        P = [] if P is None else list(P)
        C = [] if C is None else list(C)
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

    def get(self, k, key):
        d = self.vals.get(k)
        return NAN if d is None else d.get(key, NAN)


# ----------------------------------------------------------------------------------------- variables
def features(hist):
    ts, k, pr, g = hist.ts, hist.k, hist.prior, hist.get
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
    p24 = [x for x in (g(k - j, 'price') for j in range(1, 25)) if not math.isnan(x)]
    c24 = [x for x in (g(k - j, 'co2') for j in range(1, 25)) if not math.isnan(x)]
    p_mean24 = sum(p24) / len(p24) if p24 else NAN
    c_mean24 = sum(c24) / len(c24) if c24 else NAN
    pp0, pc0 = prof(pr, 'price', ts), prof(pr, 'co2', ts)
    p_dev = p[0] - pp0 if not math.isnan(p[0]) else NAN
    c_dev = c[0] - pc0 if not math.isnan(c[0]) else NAN
    pv_per_irr = pv_l1 / irr_l1 if (not math.isnan(irr_l1) and irr_l1 > 30 and not math.isnan(pv_l1)) else NAN
    kk = k_l1 if not math.isnan(k_l1) else k_mean3
    tpan_l1 = g(k - 1, 'tpan')
    # nivell de les hores de mercat conegudes respecte d'ahir i respecte del perfil
    dp_, dc_, dp2, dc2 = [], [], [], []
    for j, (pj, cj) in enumerate(known):
        yp, yc = g(k + j - 24, 'price'), g(k + j - 24, 'co2')
        if not math.isnan(yp):
            dp_.append(pj - yp); dc_.append(cj - yc)
        tj = ts + timedelta(hours=j)
        dp2.append(pj - prof(pr, 'price', tj)); dc2.append(cj - prof(pr, 'co2', tj))
    p_lvl = sum(dp_) / len(dp_) if dp_ else NAN
    c_lvl = sum(dc_) / len(dc_) if dc_ else NAN
    p_lvl2 = sum(dp2) / len(dp2) if dp2 else NAN
    c_lvl2 = sum(dc2) / len(dc2) if dc2 else NAN
    jl = len(known) - 1
    if jl >= 0:
        pl_, cl_ = known[jl]
        tl = ts + timedelta(hours=jl)
        ypl, ycl = g(k + jl - 24, 'price'), g(k + jl - 24, 'co2')
        ppl, pcl = prof(pr, 'price', tl), prof(pr, 'co2', tl)
    else:
        pl_ = cl_ = ypl = ycl = ppl = pcl = NAN
    p_m1, c_m1 = g(k - 1, 'price'), g(k - 1, 'co2')
    p_m3, c_m3 = g(k - 3, 'price'), g(k - 3, 'co2')
    p_slope = (known[jl][0] - known[0][0]) / jl if jl > 0 else NAN
    c_slope = (known[jl][1] - known[0][1]) / jl if jl > 0 else NAN
    # dinamica a curt termini de la demanda (per als models especialitzats de l'hora actual i seguents)
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
    # anomalies (valor - perfil del seu propi tipus de dia) de les darreres 24 hores conegudes
    def anom(kk, key):
        v = g(kk, key)
        if math.isnan(v):
            return NAN
        return v - prof(pr, key, ts + timedelta(hours=kk - k))

    def mfirst(xs, n):
        xs = [x for x in xs[:n] if not math.isnan(x)]
        return sum(xs) / len(xs) if xs else NAN

    AP = [anom(k + 2 - j, 'price') for j in range(24)]
    AC = [anom(k + 2 - j, 'co2') for j in range(24)]
    am = [mfirst(AP, 6), mfirst(AC, 6), mfirst(AP, 12), mfirst(AC, 12), mfirst(AP, 24), mfirst(AC, 24)]
    nkn = len(known)
    if nkn >= 3:
        dd_p, dd_c = known[2][0] - 2 * known[1][0] + known[0][0], known[2][1] - 2 * known[1][1] + known[0][1]
    else:
        dd_p = dd_c = NAN
    if nkn >= 2:
        d1_p, d1_c = known[-1][0] - known[-2][0], known[-1][1] - known[-2][1]
    else:
        d1_p = d1_c = NAN
    net_rows, mkt_rows, night = [], [], []
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
        night_known = 1 if (ts.hour <= 5 and hr <= 5 and h <= 5 - ts.hour and not math.isnan(load_l1)) else 0
        night.append(night_known)
        net_rows.append([h, hr, dow, td, mo, ts.hour, prof(pr, 'load', th), prof(pr, 'pv', th), cs, cs_prev,
                         load_l1, load_l2, g(k + h - 24, 'load'), pv_l1, g(k + h - 24, 'pv'), irr_l1, k_l1, k_mean3,
                         k_today, tamb_l1, lratio_l1, lratio_today, min(k, 200),
                         pv_persist, pv_per_irr, load_ramp, night_known, tpan_l1]
                        + nbase + _net_extras(hist, h, th, cs, cs_prev, pvk1, la1))
        tm = tipus_mercat(th)
        yp, yc = g(k + h - 24, 'price'), g(k + h - 24, 'co2')
        ap, ac = prof(pr, 'price', th), prof(pr, 'co2', th)
        if math.isnan(yp) or math.isnan(yc):
            bp, bc, ok = ap, ac, 0
        else:
            bp, bc, ok = 0.5 * yp + 0.5 * ap, 0.5 * yc + 0.5 * ac, 1
        # dinamica a curt termini: forma d'ahir / del perfil enganxada a la darrera hora coneguda
        p_shy = pl_ + (yp - ypl) if not (math.isnan(yp) or math.isnan(ypl)) else NAN
        c_shy = cl_ + (yc - ycl) if not (math.isnan(yc) or math.isnan(ycl)) else NAN
        p_shp = pl_ + (ap - ppl) if jl >= 0 else NAN
        c_shp = cl_ + (ac - pcl) if jl >= 0 else NAN
        mkt_rows.append([h, hr, dow, tm, mo, ts.hour, ap, ac, p[0], p[1], p[2], c[0], c[1], c[2],
                         yp, yc, g(k + h - 48, 'price'), g(k + h - 48, 'co2'), p_mean24, c_mean24, p_dev, c_dev,
                         cs, k_today, irr_l1, wind_l1, min(k, 200), bp, bc, ok, p_lvl, c_lvl, p_lvl2, c_lvl2,
                         p_shy, c_shy, p_shp, c_shp, p_m1, c_m1, p_m3, c_m3, p_slope, c_slope]
                        + am + [dd_p, dd_c, d1_p, d1_c] + _v4_extras(hist, h, th, known, jl, ap, ac, am, anom))
    return net_rows, mkt_rows, night, load_l1


def _net_extras(hist, h, th, cs, cs_prev, pvk1, la1):
    k, g, pr = hist.k, hist.get, hist.prior
    ty = th - timedelta(hours=24)
    csy = clear_sky(pr, ty)
    pv_pk = pvk1 * cs
    load_pa = prof(pr, 'load', th) + la1
    vy = g(k + h - 24, 'pv')
    return [cs / cs_prev if cs_prev > 20 else NAN, pv_pk, g(k + h - 24, 'load') - prof(pr, 'load', ty), load_pa,
            vy / csy if csy > 50 else NAN, load_pa - pv_pk]


def _v4_extras(hist, h, th, known, jl, ap, ac, am, anom):
    k, g, pr = hist.k, hist.get, hist.prior
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
    shm_p, shm_c = (known[jl][0] + ip / ni, known[jl][1] + ic / ni) if ni else (NAN, NAN)
    return [ay_p, ay_c, ay2_p, ay2_c, adm_p, adm_c, n, shm_p, shm_c, tipus_mercat(th - timedelta(hours=24)),
            ap + am[0], ac + am[1], ap + ay_p, ac + ay_c]


# ----------------------------------------------------------------------------------------- optimitzador
def _count_gt(sorted_vals, x):
    """Nombre d'elements > x en una llista ordenada (cerca binaria)."""
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return len(sorted_vals) - lo


def stage_segments(w, nets, U, D, pmax):
    """Cost esperat d'una hora en funcio de -canvi de SOC, com a trams (pendent, longitud) creixents.
    nets: escenaris de demanda neta ORDENATS."""
    K = len(nets)
    bps = {-D, 0.0, U}
    for n in nets:
        p = -n
        if 0.0 < p < pmax:
            bps.add(p * ETA)
        elif -pmax < p < 0.0:
            bps.add(p / ETA)
    bps = sorted(b for b in bps if -D <= b <= U)
    segs = []
    for a, b in zip(bps[:-1], bps[1:]):
        if b - a <= 1e-9:
            continue
        mid = 0.5 * (a + b)
        if mid > 0:
            p, dpd = mid / ETA, 1.0 / ETA
        else:
            p, dpd = mid * ETA, ETA
        frac = _count_gt(nets, -p) / K   # escenaris amb n + p > 0 (importacio)
        segs.append((-(w * frac * dpd), b - a))
    segs.reverse()
    return segs


def backward(ws, netss, cap, pmax):
    """Funcio de cost futur V(SOC) (convexa, trams ordenats) despres d'aplicar totes les hores."""
    U, D = pmax * ETA, pmax / ETA
    V = [(0.0, cap)]  # valor terminal 0
    for w, nets in zip(reversed(ws), reversed(netss)):
        segs = list(V)
        for s in stage_segments(w, nets, U, D, pmax):
            lo, hi = 0, len(segs)
            while lo < hi:
                mid = (lo + hi) // 2
                if segs[mid] < s:
                    lo = mid + 1
                else:
                    hi = mid
            segs.insert(lo, s)
        cut, out = U, []
        for sl, ln in segs:
            if cut > 0.0:
                if ln <= cut:
                    cut -= ln
                    continue
                ln -= cut
                cut = 0.0
            out.append([sl, ln])
        cut = D
        while cut > 1e-9 and out:
            if out[-1][1] <= cut:
                cut -= out[-1][1]
                out.pop()
            else:
                out[-1][1] -= cut
                cut = 0.0
        merged = []
        for sl, ln in out:
            if merged and abs(merged[-1][0] - sl) < 1e-15:
                merged[-1][1] += ln
            else:
                merged.append([sl, ln])
        V = [tuple(x) for x in merged]
    return V


def _slope_at(V, y):
    acc = 0.0
    for sl, ln in V:
        acc += ln
        if y < acc:
            return sl
    return V[-1][0] if V else 0.0


def decide(V1, w0, nets0, soc, cap, pmax):
    U, D = pmax * ETA, pmax / ETA
    lo, hi = max(-D, -soc), min(U, cap - soc)
    if hi - lo <= 1e-9:
        return 0.0
    K = len(nets0)
    pts = {lo, hi, 0.0}
    for n in nets0:
        p = -n
        d = p * ETA if p > 0 else p / ETA
        if lo < d < hi:
            pts.add(d)
    acc = 0.0
    for sl, ln in V1:
        acc += ln
        d = acc - soc
        if lo < d < hi:
            pts.add(d)
    pts = sorted(x for x in pts if lo <= x <= hi)
    best = lo
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a <= 1e-9:
            continue
        mid = 0.5 * (a + b)
        if mid > 0:
            p, dpd = mid / ETA, 1.0 / ETA
        else:
            p, dpd = mid * ETA, ETA
        frac = _count_gt(nets0, -p) / K
        if w0 * frac * dpd + _slope_at(V1, soc + mid) >= 0.0:
            best = a
            break
        best = b
    return best / ETA if best > 0 else best * ETA


# ----------------------------------------------------------------------------------------- agent
class BatteryAgent:
    CAP_DEFECTE = 6000.0
    PMAX_DEFECTE = 1500.0
    RATIO_CO2_COST = 0.6345   # CO2_base/Cost_base del mode dificil (kg/EUR), deduida del leaderboard public
    RES_SCALE = 1.0           # escenaris = quantils de l'error de previsio fora de mostra (plecs per setmanes)
    N_TOTAL = 2016            # hores del periode d'avaluacio (12 setmanes): permet planificar el final
    NCOLS = {'v3': 44, 'v4': 68}
    ANALOG_W = 0.6            # pes de la previsio de mercat per analegs (trajectories historiques semblants)
    ANALOG_K = 16
    N_ITER_NET = 300          # arbres usats del model general de demanda (horitzons llargs): mes rapid, mateix resultat
    N_ITER_LONG = 250         # idem per al model de mercat d'horitzo llarg

    def __init__(self):
        self.prior = json.loads((DIR / 'prior.json').read_text(encoding='utf-8'))
        self.prior['_cache'] = {}
        self.qtab = {tuple(int(x) for x in k.split('_')): v for k, v in self.prior['qtab'].items()}
        self.rk = float(self.prior['rk_train'])
        self.rc = self.RATIO_CO2_COST * self.rk
        self.models = {}
        if lgb is not None:
            names = (['net'] + [f'{t}{i}' for i in range(len(self.prior['bands'])) for t in ('p', 'c')]
                     + [f'n{i}' for i in range(len(self.prior['nbands']))])
            for name in names:
                try:
                    self.models[name] = lgb.Booster(model_str=(DIR / f'model_{name}.txt').read_text(encoding='utf-8'))
                except Exception:
                    pass
        self.hist = History(self.prior)
        self.n_calls = 0
        self.analog = None
        try:
            with np.load(DIR / 'analog.npz', allow_pickle=False) as z:
                self.analog = {key: z[key] for key in z.files}
            self.an_pools = {h: np.flatnonzero(self.analog['hour'] == h) for h in range(24)}
        except Exception:
            self.analog = None

    def _predict(self, name, X, fallback, n_iter=None):
        m = self.models.get(name)
        if m is None:
            return fallback
        try:
            X = np.ascontiguousarray(X)
            if n_iter:
                y = np.asarray(m.predict(X, num_threads=1, num_iteration=n_iter), dtype=float)
            else:
                y = np.asarray(m.predict(X, num_threads=1), dtype=float)
            return np.where(np.isfinite(y), y, fallback)
        except Exception:
            return fallback

    def _analog(self, row):
        """Previsio de preu i CO2 a 24 h: mitjana ponderada de les trajectories de train mes semblants a les
        3 hores conegudes (mateixa hora d'origen), enganxada a la darrera hora coneguda."""
        lib = self.analog
        mo = row[4]
        x = np.array([row[8] / .04, row[9] / .04, row[10] / .04, row[11] / .04, row[12] / .04, row[13] / .04,
                      (row[10] - row[8]) / .015, (row[13] - row[11]) / .015, row[20] / .04, row[21] / .04,
                      row[18] / .06, row[19] / .06, math.sin(mo * 2 * math.pi / 12) * .7, math.cos(mo * 2 * math.pi / 12) * .7,
                      (row[2] >= 5) * .5], dtype=float)
        x = np.where(np.isfinite(x), x, lib['median'])
        pool = self.an_pools[int(row[5])]
        d = ((lib['x'][pool] - x) ** 2).sum(axis=1)
        k = min(self.ANALOG_K, len(pool))
        pick = np.argpartition(d, k - 1)[:k]
        w = 1.0 / (.2 + d[pick])
        w /= w.sum()
        nb = pool[pick]
        pa = row[10] + w @ (lib['dp'][nb] - lib['dp'][nb][:, 2:3])
        ca = np.maximum(row[13] + w @ (lib['dc'][nb] - lib['dc'][nb][:, 2:3]), 0.0)
        return pa, ca

    def _plan(self, obs, soc, cap, pmax):
        self.hist.update(obs)
        net_rows, mkt_rows, night, load_l1 = features(self.hist)
        Xn = np.array(net_rows, dtype=float)
        Xm = np.array(mkt_rows, dtype=float)
        prices, co2s = Xm[:, 27].copy(), Xm[:, 28].copy()   # base heuristica (0,5*ahir + 0,5*perfil)
        for i, (fn, lo, hi) in enumerate(self.prior['bands']):  # models especialitzats per horitzo
            Xb = Xm[lo:hi + 1, :self.NCOLS[fn]]
            z = np.zeros(hi - lo + 1)
            n_it = self.N_ITER_LONG if fn == 'v3' else None
            prices[lo:hi + 1] += self._predict(f'p{i}', Xb, z, n_it)
            co2s[lo:hi + 1] += self._predict(f'c{i}', Xb, z, n_it)
        known = self.hist.known
        if self.analog is not None and len(known) >= 3:
            try:
                pa, ca = self._analog(Xm[0])
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(ca)):
                    aw = self.ANALOG_W
                    prices[3:] = (1 - aw) * prices[3:] + aw * pa[3:]
                    co2s[3:] = (1 - aw) * co2s[3:] + aw * ca[3:]
            except Exception:
                pass
        for j, (p, c) in enumerate(known[:H]):
            prices[j], co2s[j] = p, c
        ws = [0.7 * c / self.rc + 0.3 * p / self.rk for p, c in zip(prices, co2s)]
        fallback_net = np.array([r[6] - r[7] for r in net_rows])  # perfil consum - perfil solar
        nb = self.prior['nbands']
        lo0 = (max(b[1] for b in nb) + 1) if nb and all(f'n{i}' in self.models for i in range(len(nb))) else 0
        npred = np.array(fallback_net, dtype=float)
        npred[lo0:] = self._predict('net', Xn[lo0:, :28], fallback_net[lo0:], self.N_ITER_NET)
        for i, (lo, hi, nc) in enumerate(nb):   # models de demanda especialitzats per horitzo
            npred[lo:hi + 1] = self._predict(f'n{i}', Xn[lo:hi + 1, :nc], npred[lo:hi + 1])
        netss = []
        for h in range(H):
            if night[h] and not math.isnan(load_l1):
                netss.append([load_l1])  # comptador congelat a la nit: demanda coneguda
            else:
                q = self.qtab[(h, int(net_rows[h][1]))]   # error tipic per horitzo i hora del dia
                netss.append(sorted(npred[h] + self.RES_SCALE * x for x in q))
        # final de la simulacio: no te sentit guardar energia per a hores que no existiran
        hh = H
        rem = self.N_TOTAL - self.n_calls + 1   # hores que queden, incloent-hi aquesta
        if 0 < rem < H and self.hist.k + rem == 168:   # el final coincideix amb el final d'un bloc de 7 dies
            hh = rem
        if len(known) < 3:          # el simulador nomes escurca el vector de mercat a les 2 darreres hores
            hh = min(hh, max(1, len(known)))
        ws, netss = ws[:hh], netss[:hh]
        V1 = backward(ws[1:], netss[1:], cap, pmax)
        return decide(V1, ws[0], netss[0], soc, cap, pmax)

    def act(self, obs: dict) -> float:
        cap = _f(obs.get('cap_max_kwh'))
        cap = self.CAP_DEFECTE if math.isnan(cap) or cap <= 0 else cap
        pmax = _f(obs.get('power_max_kw'))
        pmax = self.PMAX_DEFECTE if math.isnan(pmax) or pmax <= 0 else pmax
        soc = _f(obs.get('soc_kwh'))
        soc = 0.0 if math.isnan(soc) else min(cap, max(0.0, soc))
        self.n_calls += 1
        try:
            p = float(self._plan(obs, soc, cap, pmax))
            if not math.isfinite(p):
                p = 0.0
        except Exception:
            p = 0.0
        eta = math.sqrt(0.9)
        max_c = max(0.0, min(pmax, (cap - soc) / eta - 1e-4))
        max_d = max(0.0, min(pmax, soc * eta - 1e-4))
        return float(max(-max_d, min(max_c, p)))
