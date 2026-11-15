"""Exact receding-horizon optimiser for the hard mode, pure Python.

Expected stage cost with K equiprobable net-demand scenarios n_i (kW) and weight w:
    f(p) = w/K * sum_i max(n_i + p, 0)          (p = grid-side battery power, + charge / - discharge)
Convex piecewise linear in p -> convex PWL in the stored-energy change D (p = D/eta if D>0 else D*eta).
Backward recursion on V(SOC) (convex PWL, list of (slope, length) sorted by slope) by slope merging.
"""
import bisect
import math

ETA = math.sqrt(0.9)


def stage_segments(w, nets, U, D, pmax, eta=ETA):
    """c~(z) = c(-z) on z in [-U, D] as ascending (slope, length) segments."""
    K = len(nets)
    bps = {-D, 0.0, U}
    for n in nets:
        p = -n
        if 0.0 < p < pmax:
            bps.add(p * eta)
        elif -pmax < p < 0.0:
            bps.add(p / eta)
    bps = sorted(b for b in bps if -D <= b <= U)
    segs = []
    for a, b in zip(bps[:-1], bps[1:]):
        if b - a <= 1e-9:
            continue
        mid = 0.5 * (a + b)
        if mid > 0:
            p, dpd = mid / eta, 1.0 / eta
        else:
            p, dpd = mid * eta, eta
        frac = sum(1 for n in nets if n + p > 0.0) / K
        segs.append((-(w * frac * dpd), b - a))
    segs.reverse()
    return segs


def backward(ws, netss, cap, pmax, v_term=0.0, eta=ETA):
    """Cost-to-go V_0 (after applying all stages) as sorted (slope, length) list covering SOC in [0, cap]."""
    U, D = pmax * eta, pmax / eta
    V = [(-v_term, cap)]
    for w, nets in zip(reversed(ws), reversed(netss)):
        segs = list(V)
        for s in stage_segments(w, nets, U, D, pmax, eta):
            bisect.insort(segs, s)
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
        # merge equal slopes to keep the list short
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


def decide(V1, w0, nets0, soc, cap, pmax, eta=ETA):
    """argmin_D c0(D) + V1(soc + D); returns grid-side power p (kW)."""
    U, D = pmax * eta, pmax / eta
    lo, hi = max(-D, -soc), min(U, cap - soc)
    if hi - lo <= 1e-9:
        return 0.0
    K = len(nets0)
    pts = {lo, hi, 0.0}
    for n in nets0:
        p = -n
        d = p * eta if p > 0 else p / eta
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
            p, dpd = mid / eta, 1.0 / eta
        else:
            p, dpd = mid * eta, eta
        frac = sum(1 for n in nets0 if n + p > 0.0) / K
        g = w0 * frac * dpd + _slope_at(V1, soc + mid)
        if g >= 0.0:
            best = a
            break
        best = b
    # prefer idling inside a flat region that contains 0
    d = best
    return d / eta if d > 0 else d * eta


def plan(ws, netss, soc, cap, pmax, v_term=0.0):
    V1 = backward(ws[1:], netss[1:], cap, pmax, v_term)
    return decide(V1, ws[0], netss[0], soc, cap, pmax)


# ---------------------------------------------------------------- v4 additions
def _count_gt(sorted_vals, x):
    return len(sorted_vals) - bisect.bisect_right(sorted_vals, x)


def stage_segments(w, nets, U, D, pmax, eta=ETA):  # noqa: F811  (faster: nets sorted, bisect counting)
    ns = sorted(nets)
    K = len(ns)
    bps = {-D, 0.0, U}
    for n in ns:
        p = -n
        if 0.0 < p < pmax:
            bps.add(p * eta)
        elif -pmax < p < 0.0:
            bps.add(p / eta)
    bps = sorted(b for b in bps if -D <= b <= U)
    segs = []
    for a, b in zip(bps[:-1], bps[1:]):
        if b - a <= 1e-9:
            continue
        mid = 0.5 * (a + b)
        if mid > 0:
            p, dpd = mid / eta, 1.0 / eta
        else:
            p, dpd = mid * eta, eta
        segs.append((-(w * (_count_gt(ns, -p) / K) * dpd), b - a))
    segs.reverse()
    return segs


def back_step(V, w, nets, cap, pmax, eta=ETA):
    U, D = pmax * eta, pmax / eta
    segs = list(V)
    for s in stage_segments(w, nets, U, D, pmax, eta):
        bisect.insort(segs, s)
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
    return [tuple(x) for x in merged]


def backward(ws, netss, cap, pmax, v_term=0.0, eta=ETA):  # noqa: F811
    V = [(-v_term, cap)]
    for w, nets in zip(reversed(ws), reversed(netss)):
        V = back_step(V, w, nets, cap, pmax, eta)
    return V


def grid_slopes(V, grid):
    out, acc, i = [], 0.0, -1
    sl = 0.0
    for g in grid:
        while g >= acc and i + 1 < len(V):
            i += 1
            sl = V[i][0]
            acc += V[i][1]
        out.append(sl)
    return out


def decide(V1, w0, nets0, soc, cap, pmax, eta=ETA):  # noqa: F811
    U, D = pmax * eta, pmax / eta
    lo, hi = max(-D, -soc), min(U, cap - soc)
    if hi - lo <= 1e-9:
        return 0.0
    ns = sorted(nets0)
    K = len(ns)
    pts = {lo, hi, 0.0}
    for n in ns:
        p = -n
        d = p * eta if p > 0 else p / eta
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
            p, dpd = mid / eta, 1.0 / eta
        else:
            p, dpd = mid * eta, eta
        if w0 * (_count_gt(ns, -p) / K) * dpd + _slope_at(V1, soc + mid) >= 0.0:
            best = a
            break
        best = b
    return best / eta if best > 0 else best * eta


def plan(ws, netss, soc, cap, pmax, v_term=0.0):  # noqa: F811
    V1 = backward(ws[1:], netss[1:], cap, pmax, v_term)
    return decide(V1, ws[0], netss[0], soc, cap, pmax)
