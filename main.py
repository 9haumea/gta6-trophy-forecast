"""GTA VI trophy earn-rate forecaster for make.fun's gta6-trophy-rarity-forecast-01.

Python 3.13, standard library only, deterministic, no network. Runs once per case:
reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output).

An Exophase earn rate is holders / players-tracked. The page refreshes stepwise, so
the history is collapsed into refresh steps. During launch week the tracked-player
pool N(t) grows far faster than most trophies' holder counts H_i(t), so most rates
fall by dilution while a few early-story trophies climb. The forecast models N(t)
with a damped growth-curve ensemble and each H_i(t) with a monotone saturating fit
shrunk toward sticky holders, then reads rate_i = H_i / N at the target and spreads
five quantiles on the logit scale, widening with the horizon. A flat launch-night
history instead uses the documented first-refresh prior.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]
# Release epoch the house curves measure from: 2026-11-19T00:00:00Z.
T0 = 1795046400.0
# Documented first refresh of launch night: 638 -> 3868 tracked players.
LAUNCH_POOL_MULT = 3868.0 / 638.0


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def pct(lg):
    return 100.0 * sigmoid(lg)


def hours(iso):
    # ISO 8601 UTC -> hours since T0, without datetime (kept tiny and deterministic).
    s = iso.replace("Z", "+00:00")
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (dt.timestamp() - T0) / 3600.0


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000.0
    return int(v) if float(v).is_integer() else v


# ---- small Nelder-Mead (stdlib, fixed iteration budget) ----
def nelder_mead(f, x0, step=0.5, iters=500, tol=1e-8):
    n = len(x0)
    pts = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step if p[i] == 0 else step * abs(p[i])
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if vals[-1] - vals[0] < tol:
            break
        cen = [sum(pts[i][j] for i in range(n)) / n for j in range(n)]
        xr = [2.0 * cen[j] - pts[-1][j] for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [cen[j] + 2.0 * (xr[j] - cen[j]) for j in range(n)]
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [cen[j] + 0.5 * (pts[-1][j] - cen[j]) for j in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = [(pts[i][j] + pts[0][j]) / 2.0 for j in range(n)]
                    vals[i] = f(pts[i])
    i = min(range(n + 1), key=lambda k: vals[k])
    return pts[i], vals[i]


# ---- tracked-player growth N(t) ----
def _logloss(preds, ns):
    return sum((math.log(max(p, 1.0)) - math.log(n)) ** 2 for p, n in zip(preds, ns))


def fit_logistic(ts, ns):
    def loss(p):
        K, A, r = math.exp(p[0]), math.exp(p[1]), math.exp(p[2])
        return _logloss([K / (1.0 + A * math.exp(-r * t)) for t in ts], ns)
    p, v = nelder_mead(loss, [math.log(max(ns) * 3.0), math.log(20.0), math.log(0.08)], iters=700)
    K, A, r = math.exp(p[0]), math.exp(p[1]), math.exp(p[2])
    return (lambda t: K / (1.0 + A * math.exp(-r * t))), v


def fit_gompertz(ts, ns):
    def loss(p):
        K, b, c = math.exp(p[0]), math.exp(p[1]), math.exp(p[2])
        return _logloss([K * math.exp(-b * math.exp(-c * t)) for t in ts], ns)
    p, v = nelder_mead(loss, [math.log(max(ns) * 3.0), math.log(3.0), math.log(0.05)], iters=700)
    K, b, c = math.exp(p[0]), math.exp(p[1]), math.exp(p[2])
    return (lambda t: K * math.exp(-b * math.exp(-c * t))), v


def fit_power(ts, ns):
    xs = [math.log(1.0 + t) for t in ts]
    ys = [math.log(n) for n in ns]
    m = len(xs)
    mx = sum(xs) / m
    my = sum(ys) / m
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    a = math.exp(my - b * mx)
    v = sum((math.log(a) + b * x - y) ** 2 for x, y in zip(xs, ys))
    return (lambda t: a * (1.0 + t) ** b), v


def fit_linear(ts, ns):
    m = len(ts)
    mt = sum(ts) / m
    mn = sum(ns) / m
    den = sum((t - mt) ** 2 for t in ts) or 1e-9
    b = sum((t - mt) * (n - mn) for t, n in zip(ts, ns)) / den
    a = mn - b * mt
    v = sum((a + b * t - n) ** 2 for t, n in zip(ts, ns))
    return (lambda t: max(a + b * t, 1.0)), v


def refresh_steps(reads):
    steps = []
    for t, n, r in reads:
        if steps and steps[-1][1] == n and steps[-1][2] == r:
            continue
        steps.append((t, n, r))
    return steps


def forecast_pool(ts, ns, t_target, horizon_h):
    """Ensemble median of N(target) and a log-scale spread.

    Curve families (logistic, Gompertz, power, linear) capture saturation when the
    history is long enough to identify it; a damped-trend continuation built from the
    recent log-growth rate stays honest when refresh steps are few and a sigmoid would
    lock its ceiling onto the last observation. The spread carries the disagreement.
    """
    n_last = float(ns[-1])
    t_last = ts[-1]
    k = min(3, len(ts))
    if k >= 2 and ts[-1] > ts[-k]:
        g = (math.log(ns[-1]) - math.log(ns[-k])) / (ts[-1] - ts[-k])
    else:
        g = 0.0
    g = max(g, 0.0)
    dt = max(t_target - t_last, 0.0)
    # a minimal continuation floor: growth may decelerate, but the pool does not shrink
    floor = math.log(n_last) + g * dt * 0.10
    scenarios = []  # (logN_target, weight)
    for fitter in (fit_logistic, fit_gompertz, fit_power, fit_linear):
        try:
            f, v = fitter(ts, ns)
            np_ = f(t_target)
        except Exception:
            continue
        if not math.isfinite(np_) or np_ < n_last:
            continue
        lg = math.log(np_)
        # distrust a curve that saturates far below the continuation floor
        w = math.exp(-(v / max(len(ts), 1)) / 0.15)
        if lg < floor:
            w *= 0.15
        scenarios.append((lg, w))
    for tau in (24.0, 60.0, 120.0):
        lg = math.log(n_last) + g * tau * (1.0 - math.exp(-dt / tau))
        scenarios.append((max(lg, math.log(n_last)), 1.0))
    if not scenarios:
        return n_last, 0.4
    logs = [lg for lg, _ in scenarios]
    weights = [w for _, w in scenarios]
    wsum = sum(weights)
    logN = sum(w * lg for w, lg in zip(weights, logs)) / wsum
    N_med = math.exp(logN)
    disp = max(logs) - min(logs)
    sig = max(0.30, 0.55 * disp) + 0.12 * math.sqrt(max(horizon_h, 0.0) / 24.0)
    return N_med, sig


# ---- per-trophy holder growth H_i(t) ----
def fit_holders(ts, hs):
    """Saturating exponential H(t)=Hinf-(Hinf-H0)exp(-k(t-t0)), shrunk to sticky."""
    h0 = hs[0]
    hlast = hs[-1]
    if len(ts) < 2 or hlast <= h0:
        return float(hlast), 0.0, float(h0)
    span = ts[-1] - ts[0] or 1.0

    def loss(p):
        Hinf = hlast + math.exp(p[0])
        k = math.exp(p[1])
        s = 0.0
        for t, h in zip(ts, hs):
            pred = Hinf - (Hinf - h0) * math.exp(-k * (t - ts[0]))
            s += (pred - h) ** 2
        return s

    p, v = nelder_mead(loss, [math.log(max(hlast - h0, 1.0)), math.log(0.05)], iters=350)
    Hinf = hlast + math.exp(p[0])
    k = math.exp(p[1])
    # shrink the asymptote toward the last observation when growth is weakly identified
    rel = (hlast - h0) / max(hlast, 1.0)
    shrink = 0.5 if rel < 0.25 else 0.85
    Hinf = hlast + (Hinf - hlast) * shrink
    return Hinf, k, float(h0)


def forecast_case(case):
    reads = [(hours(r["at"]), r["players_tracked"], tuple(r["rates"])) for r in case["history"]]
    horizon_h = case["horizon_minutes"] / 60.0
    t_target = hours(case["target"])
    t_last = reads[-1][0]
    trophies = case["trophies"]

    # Launch-night flat history: the documented first refresh (pool ~6x, sticky holders).
    n0 = reads[0][1]
    r0 = reads[0][2]
    flat = all(n == n0 and r == r0 for _, n, r in reads)
    if flat and n0 < 2000 and t_last < 8.0:
        N1 = n0 * LAUNCH_POOL_MULT
        rows = []
        for i, tr in enumerate(trophies):
            rate0 = r0[i] if r0[i] is not None else 0.0
            if rate0 >= 99.99:
                rows.append({"id": tr["id"], "q": [99.9, 99.95, 99.99, 99.99, 99.99]})
                continue
            H = int(round(rate0 * n0 / 100.0))
            lg = logit(max(H / N1 * 100.0, 0.01))
            sig = 0.22
            rows.append({"id": tr["id"], "q": [round4(pct(lg + z * sig)) for z in Z]})
        return {"trophies": rows}

    steps = refresh_steps(reads)
    ts = [s[0] for s in steps]
    ns = [s[1] for s in steps]
    N_med, sig_logN = forecast_pool(ts, ns, t_target, horizon_h)

    rows = []
    for i, tr in enumerate(trophies):
        hs = []
        hts = []
        for t, n, r in steps:
            if r[i] is not None:
                hs.append(int(round(r[i] * n / 100.0)))
                hts.append(t)
        if not hs:
            rows.append({"id": tr["id"], "q": [round4(pct(logit(0.01) + z * 0.5)) for z in Z]})
            continue
        Hinf, k, h0 = fit_holders(hts, hs)
        H_last = hs[-1]
        if k > 0:
            H_target = Hinf - (Hinf - h0) * math.exp(-k * (t_target - hts[0]))
        else:
            H_target = float(H_last)
        H_target = max(H_target, float(H_last))
        rate_med = clamp(H_target / N_med * 100.0)
        lg = logit(rate_med)
        sig_logH = 0.16 + 0.10 * math.sqrt(max(horizon_h, 0.0) / 24.0)
        if H_last <= 6:
            sig_logH += 0.25
        sig = math.sqrt(sig_logN ** 2 + sig_logH ** 2)
        rows.append({"id": tr["id"], "q": [round4(pct(lg + z * sig)) for z in Z]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
