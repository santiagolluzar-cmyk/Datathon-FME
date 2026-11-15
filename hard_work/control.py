"""Replay control policies against cached causal forecasts, without re-running models."""
import math
import time
import json
import numpy as np

ETA = math.sqrt(0.9)


def backward_terminal(module, ws, netss, cap, pmax, terminal=0.0):
    """Same exact convex optimizer, with an optional remaining-energy value."""
    if abs(terminal) < 1e-15:
        return module.backward(ws, netss, cap, pmax)
    U, D = pmax * ETA, pmax / ETA
    V = [(-float(terminal), cap)]
    for w, nets in zip(reversed(ws), reversed(netss)):
        segs = sorted(V + module.stage_segments(w, nets, U, D, pmax))
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


def _array(cache, *names, default=None):
    for name in names:
        if name in cache:
            return cache[name]
    if default is not None:
        return default
    raise KeyError(f"Cache is missing one of {names}")


def prepare(cache, module, config=None):
    """Prepare SOC-independent continuation values for one forecast policy.

    Config: scale (all horizons), future_scale, current_scale, market_strength,
    future_weight_shrink, terminal_frac, terminal_abs, horizon, net_bias,
    daylight_bias, net_profile_blend, ratio, reset_soc (default False).
    oracle=True is an undeployable 24-hour perfect-forecast benchmark.
    Model outputs otherwise remain fully causal.
    Current-scale can also be overridden in replay() without recomputing futures.
    """
    cfg = dict(config or {})
    prices = np.array(_array(cache, 'prices', 'price_pred'), dtype=float, copy=True)
    co2 = np.array(_array(cache, 'co2', 'co2_pred'), dtype=float, copy=True)
    net = np.array(_array(cache, 'net', 'net_pred'), dtype=float, copy=True)
    N, H = prices.shape
    oracle = bool(cfg.get('oracle', False))
    if oracle:
        ids = _array(cache, 'block_id', 'blocks')
        rp = _array(cache, 'price', 'real_price')
        rc = _array(cache, 'real_co2', 'actual_co2', 'co2_real', 'carbon')
        rn = _array(cache, 'load', 'load_kw')-_array(cache, 'pv', 'pv_kw')
        for i in range(N):
            for h in range(min(H, N-i)):
                if ids[i+h] != ids[i]:
                    break
                prices[i,h], co2[i,h], net[i,h] = rp[i+h], rc[i+h], rn[i+h]
    strength = float(cfg.get('market_strength', 1.0))
    if strength != 1.0:
        bp, bc = _array(cache, 'bp'), _array(cache, 'bc')
        prices[:, 3:] = bp[:, 3:] + strength * (prices[:, 3:] - bp[:, 3:])
        co2[:, 3:] = bc[:, 3:] + strength * (co2[:, 3:] - bc[:, 3:])
    rk = float(module.BatteryAgent.RATIO_CO2_COST)
    ratio = float(cfg.get('ratio', rk))
    # Overall normalization cancels from the minimizing policy.
    weights = 0.7 * co2 / ratio + 0.3 * prices
    if cfg.get('future_weight_shrink', 0.0):
        shrink = float(cfg['future_weight_shrink'])
        weights[:, 3:] = (1-shrink)*weights[:, 3:] + shrink*np.mean(weights[:, 3:], axis=1, keepdims=True)
    # Convex slope merging requires nonnegative hourly marginal costs.
    weights = np.maximum(weights, 0.0)
    night = _array(cache, 'night', default=np.zeros_like(net, dtype=bool))
    lastload = _array(cache, 'lastload', 'last_load', default=np.full(N, np.nan))
    if 'solar' in cache:
        solar = cache['solar']
    elif 'daylight' in cache:
        solar = cache['daylight']
    elif 'Xn' in cache:
        solar = cache['Xn'][:, :, 8] > 50
    else:
        raise KeyError('Cache needs solar (Nx24 bool) or Xn features')
    prior = json.loads((module.DIR/'prior.json').read_text(encoding='utf-8'))
    table = cfg.get('qtab', prior['qtab'])
    qtab = {tuple(int(x) for x in key.split('_')): np.asarray(value) for key, value in table.items()}
    net += float(cfg.get('net_bias', 0.0))
    net += np.asarray(solar) * float(cfg.get('daylight_bias', 0.0))
    if cfg.get('net_profile_blend', 0.0):
        Xn = cache['Xn']
        profile = Xn[:, :, 6] - Xn[:, :, 7]
        blend = float(cfg['net_profile_blend'])
        net = (1-blend)*net + blend*profile
    cap = float(cfg.get('cap', 6000.0))
    power = float(cfg.get('power', 1500.0))
    scale = float(cfg.get('scale', 2.0))
    fs = float(cfg.get('future_scale', scale))
    cs = float(cfg.get('current_scale', scale))
    horizon = int(cfg.get('horizon', H))
    values, current, current_q, current_known = [], [], [], []
    for i in range(N):
        nets = []
        for h in range(horizon):
            j = h % H
            if oracle:
                nets.append([float(net[i,j])])
            elif night[i, j] and np.isfinite(lastload[i]):
                nets.append([float(lastload[i])])
            else:
                qs = qtab[(j, int(solar[i, j]))]
                ns = cs if h == 0 else fs
                nets.append(sorted(net[i, j] + ns*qs))
        ws = [float(weights[i, h % H]) for h in range(horizon)]
        terminal = float(cfg.get('terminal_abs', 0.0)) + float(cfg.get('terminal_frac', 0.0))*float(np.median(ws))*ETA
        V = backward_terminal(module, ws[1:], nets[1:], cap, power, terminal=terminal)
        values.append(V)
        current.append(nets[0])
        current_q.append(qtab[(0, int(solar[i, 0]))])
        current_known.append(bool(oracle or (night[i, 0] and np.isfinite(lastload[i]))))
    return dict(values=values, current=current, weights=weights[:, 0], cfg=cfg,
                net=net[:, 0], current_q=current_q, current_known=current_known,
                cap=cap, power=power)


def replay(cache, module, prepared, current_scale=None, return_series=False):
    cap, power = prepared['cap'], prepared['power']
    block_id = _array(cache, 'block_id', 'blocks')
    load = _array(cache, 'load', 'load_kw')
    pv = _array(cache, 'pv', 'pv_kw')
    price = _array(cache, 'price', 'real_price')
    co2 = _array(cache, 'real_co2', 'actual_co2', 'co2_real', 'carbon')
    totals = np.zeros(4)
    per_block = {}
    series = []
    soc = cap/2
    old_block = None
    start = time.perf_counter()
    for i, bid in enumerate(block_id):
        bid = int(bid)
        if bid != old_block and prepared['cfg'].get('reset_soc', False):
            soc = cap/2
        old_block = bid
        nets0 = prepared['current'][i]
        if current_scale is not None and not prepared['current_known'][i]:
            nets0 = sorted(prepared['net'][i] + current_scale*prepared['current_q'][i])
        action = module.decide(prepared['values'][i], float(prepared['weights'][i]), nets0, soc, cap, power)
        max_c = max(0.0, min(power, (cap-soc)/ETA - 1e-4))
        max_d = max(0.0, min(power, soc*ETA - 1e-4))
        action = float(max(-max_d, min(max_c, action)))
        soc += action*ETA if action >= 0 else action/ETA
        base = max(0.0, load[i]-pv[i])
        grid = max(0.0, load[i]-pv[i]+action)
        raw = np.asarray([base*price[i], grid*price[i], base*co2[i], grid*co2[i]])
        totals += raw
        per_block.setdefault(bid, np.zeros(4))
        per_block[bid] += raw
        if return_series:
            series.append((action, soc, grid))
    def metrics(raw):
        bcost, cost, bco2, emissions = raw
        cp = 100*(1-cost/bcost) if bcost>0 else 0.0
        ep = 100*(1-emissions/bco2) if bco2>0 else 0.0
        return dict(score=max(0., .7*ep+.3*cp), cost_pct=cp, co2_pct=ep,
                    cost_base=bcost, cost_agent=cost, co2_base=bco2, co2_agent=emissions)
    result = dict(**metrics(totals), infraccions=0, soc_final_kwh=soc,
                  blocks={bid: metrics(raw) for bid, raw in per_block.items()},
                  replay_s=time.perf_counter()-start)
    if return_series:
        result['series'] = np.asarray(series)
    return result


def run(cache, module, config=None, return_series=False):
    start = time.perf_counter()
    p = prepare(cache, module, config)
    result = replay(cache, module, p, return_series=return_series)
    result['total_s'] = time.perf_counter()-start
    return result


def oracle_global(cache, module, reset_soc=False, ratio=None):
    """Exact full-data perfect-information optimum; never a deployable policy.

    Unlike oracle=True's finite window, this is an upper bound for the supplied
    series and objective ratio, including SOC carried across temporal gaps.
    """
    ratio = float(module.BatteryAgent.RATIO_CO2_COST if ratio is None else ratio)
    load = _array(cache, 'load', 'load_kw')
    pv = _array(cache, 'pv', 'pv_kw')
    price = _array(cache, 'price', 'real_price')
    carbon = _array(cache, 'real_co2', 'actual_co2', 'co2_real', 'carbon')
    ids = _array(cache, 'block_id', 'blocks')
    ws = .7*carbon/ratio+.3*price
    if np.any(ws < 0):
        raise ValueError('Convex oracle requires nonnegative weighted costs')
    cap, power = 6000., 1500.
    U, D = power*ETA, power/ETA
    values = [None]*len(load)
    current = [[float(n)] for n in load-pv]
    V = [(0.,cap)]
    for i in range(len(load)-1,-1,-1):
        if reset_soc and i<len(load)-1 and ids[i] != ids[i+1]:
            V = [(0.,cap)]
        values[i] = V
        segs = sorted(V + module.stage_segments(float(ws[i]), current[i], U, D, power))
        cut, out = U, []
        for sl, ln in segs:
            if cut > 0:
                trim = min(cut, ln)
                cut -= trim
                ln -= trim
            if ln > 1e-9:
                out.append([sl,ln])
        cut = D
        while cut > 1e-9 and out:
            trim = min(cut,out[-1][1])
            cut -= trim
            out[-1][1] -= trim
            if out[-1][1] <= 1e-9:
                out.pop()
        merged = []
        for sl,ln in out:
            if merged and abs(sl-merged[-1][0])<1e-15:
                merged[-1][1] += ln
            else:
                merged.append([sl,ln])
        V = [tuple(x) for x in merged]
    p = dict(values=values,current=current,weights=ws,cfg={'reset_soc':reset_soc},cap=cap,power=power)
    return replay(cache,module,p,return_series=True)
