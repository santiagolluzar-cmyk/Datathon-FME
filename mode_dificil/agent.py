"""Control del mode dificil: mercat per trajectories historiques semblants,
demanda residual i quantils condicionals, amb MPC convex exacte de 24 hores.
Nomes utilitza les dades publicades de train i les observacions rebudes.
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
        valid_prev = self.last_ts is not None and ts - self.last_ts == timedelta(hours=1)
        if not valid_prev:
            self.reset()
            self.t0 = ts
        self.k = int(round((ts - self.t0).total_seconds() / 3600))
        self.last_ts = self.ts = ts
        nd = obs.get('noves_dades_5min')
        nd = nd if isinstance(nd, dict) else {}
        if valid_prev:  # la telemetria nomes es valida si l'obs anterior era l'hora anterior
            d = self.vals.setdefault(self.k - 1, {})
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
                         pv_persist, pv_per_irr, load_ramp, night_known, tpan_l1])
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
                         p_shy, c_shy, p_shp, c_shp, p_m1, c_m1, p_m3, c_m3, p_slope, c_slope])
    return net_rows, mkt_rows, night, load_l1


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

def market_features(xm):
    x=np.asarray(xm,dtype=float)
    if x.ndim==3:
        x=x[:,0]
    elif x.ndim==2 and x.shape[0]==24:
        x=x[0:1]
    month=x[:,4]
    return np.column_stack([x[:,8]/.04,x[:,9]/.04,x[:,10]/.04,
                            x[:,11]/.04,x[:,12]/.04,x[:,13]/.04,
                            (x[:,10]-x[:,8])/.015,(x[:,13]-x[:,11])/.015,
                            x[:,20]/.04,x[:,21]/.04,x[:,18]/.06,x[:,19]/.06,
                            np.sin(month*2*np.pi/12)*.7,np.cos(month*2*np.pi/12)*.7,
                            (x[:,2]>=5)*.5])


class MarketAnalog:
    def __init__(self,path,k=4):
        with np.load(path,allow_pickle=False) as z:
            self.library={key:z[key] for key in z.files}
        self.k=int(k)
        self.groups={hour:np.flatnonzero(self.library['hour']==hour) for hour in range(24)}

    def predict(self,xm):
        """Input market feature24x44 matrix; output future24 price/CO2 arrays."""
        xm=np.asarray(xm,dtype=float)
        row=xm[0]
        x=market_features(xm)[0]
        lib=self.library
        x=np.where(np.isfinite(x),x,lib['median'])
        pool=self.groups[int(row[5])]
        if not len(pool):
            raise ValueError('No market analogs for origin hour')
        d=np.sum((lib['x'][pool]-x)**2,axis=1)
        k=min(self.k,len(pool))
        picked=np.argpartition(d,k-1)[:k]
        neighbors=pool[picked]
        weights=1/(.2+d[picked])
        weights/=weights.sum()
        forecasts=[]
        for key,start in [('delta_price',8),('delta_co2',11)]:
            known=row[start:start+3]
            finite=np.flatnonzero(np.isfinite(known))
            if not len(finite):
                raise ValueError('Current market observation is absent')
            j=int(finite[-1])
            delta=lib[key][neighbors]
            forecast=known[j]+np.sum(weights[:,None]*(delta-delta[:,j,None]),axis=0)
            forecast[finite]=known[finite]
            if key=='delta_co2':
                forecast=np.maximum(forecast,0.)
            forecasts.append(forecast)
        return forecasts


class BatteryAgent:
    CAP_DEFECTE = 6000.0
    PMAX_DEFECTE = 1500.0
    RATIO_CO2_COST = 0.6345
    RES_SCALE = 2.0

    def __init__(self):
        self.prior = json.loads((DIR / 'prior.json').read_text(encoding='utf-8'))
        self.prior['_cache'] = {}
        self.qtab = {tuple(int(x) for x in k.split('_')): v for k, v in self.prior['qtab'].items()}
        self.models = {}
        if lgb is not None:
            for name in ['net'] + [f'q{j}' for j in range(7)]:
                filename = 'model_net.txt' if name == 'net' else f'net_{name}.txt'
                try:
                    self.models[name] = lgb.Booster(model_str=(DIR / filename).read_text(encoding='utf-8'))
                except Exception:
                    pass
        try:
            self.market = MarketAnalog(DIR / 'analog_library.npz', k=4)
        except Exception:
            self.market = None
        self.hist = History(self.prior)

    def _predict(self, name, X, fallback):
        model = self.models.get(name)
        if model is None:
            return fallback
        try:
            y = np.asarray(model.predict(X, num_threads=1), dtype=float)
            return np.where(np.isfinite(y), y, fallback)
        except Exception:
            return fallback

    def _plan(self, obs, soc, cap, pmax):
        self.hist.update(obs)
        net_rows, mkt_rows, night, load_l1 = features(self.hist)
        Xn = np.asarray(net_rows, dtype=float)
        Xm = np.asarray(mkt_rows, dtype=float)
        prices, co2s = Xm[:, 27].copy(), Xm[:, 28].copy()
        if self.market is not None:
            try:
                prices, co2s = self.market.predict(Xm)
            except Exception:
                pass
        for j, (p, c) in enumerate(self.hist.known[:H]):
            prices[j], co2s[j] = p, c
        # Nonnegative marginal costs preserve the convex optimizer's assumptions.
        ws = np.maximum(0.7 * co2s / self.RATIO_CO2_COST + 0.3 * prices, 0.0).tolist()
        base = np.where(np.isfinite(Xn[:, 25]), Xn[:, 25], Xn[:, 6])
        base -= np.where(np.isfinite(Xn[:, 23]), Xn[:, 23], Xn[:, 7])
        npred = base + self._predict('net', Xn, np.zeros(H))
        netss = []
        for h in range(H):
            if night[h] and math.isfinite(load_l1):
                netss.append([load_l1])
            else:
                q = self.qtab[(h, int(Xn[h, 8] > 50))]
                netss.append(sorted(npred[h] + self.RES_SCALE * x for x in q))
        # Conditional current-hour quantiles replace a fixed error-width assumption.
        if not (night[0] and math.isfinite(load_l1)) and all(f'q{j}' in self.models for j in range(7)):
            scenarios = [float(base[0] + self._predict(f'q{j}', Xn[:1], np.zeros(1))[0]) for j in range(7)]
            if all(math.isfinite(x) for x in scenarios):
                netss[0] = sorted(scenarios)
        V1 = backward(ws[1:], netss[1:], cap, pmax)
        return decide(V1, ws[0], netss[0], soc, cap, pmax)

    def act(self, obs: dict) -> float:
        cap = _f(obs.get('cap_max_kwh'))
        cap = self.CAP_DEFECTE if math.isnan(cap) or cap <= 0 else cap
        pmax = _f(obs.get('power_max_kw'))
        pmax = self.PMAX_DEFECTE if math.isnan(pmax) or pmax <= 0 else pmax
        soc = _f(obs.get('soc_kwh'))
        soc = 0.0 if math.isnan(soc) else min(cap, max(0.0, soc))
        try:
            p = float(self._plan(obs, soc, cap, pmax))
            if not math.isfinite(p):
                p = 0.0
        except Exception:
            p = 0.0
        max_c = max(0.0, min(pmax, (cap - soc) / ETA - 1e-4))
        max_d = max(0.0, min(pmax, soc * ETA - 1e-4))
        return float(max(-max_d, min(max_c, p)))
