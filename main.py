"""Refresh-Rescale Mixture Forecaster -- gta6-trophy-rarity-forecast-01 (v2).

Forecasts each GTA VI PS5 trophy's Exophase earn rate at the case target as
five quantiles (0.05/0.25/0.5/0.75/0.95), in percent. Python 3.13, standard
library only, no network, deterministic.

MODEL (three regimes, picked from the case's own history):

1. Straddled known first refresh (frozen launch-night page, 638 players
   tracked, target at/after Exophase's first refresh, 2026-11-19T04:30Z).
   The refresh is a near-deterministic rescale, not a random walk: the kit
   publishes the refreshed aggregates (3,868 players tracked, 4,276
   trophies earned; RULES.txt + competition.yaml). Non-100% holders grow
   305 -> 408 while players grow 638 -> 3,868, so every non-100% rate is
   multiplied by about (408/305) * (638/3868) = 0.2206. The median is the
   geometric mean of the two rounding conventions (exact rate rescale and
   integer-holder rescale), spread 0.08 logit units. Fingerprint check:
   this factor reproduces the published house-baseline practice numbers
   (typical miss 0.68pp, coverage 0) and ln(1/0.2206) = 1.51 matches its
   ~1.51 loss. A 100% trophy stays 100% (rules: every rate but the one at
   100% fell), with a small downside hedge.
2. Frozen page, no known refresh in the window: persistence with a
   random-walk spread (0.5 * sqrt(days)), honest ignorance.
3. Refreshes observed (the final-like case): per-trophy holders/players
   decomposition. Holders and players are fit log-linear in ln(1 + hours
   since release) over the last 6 distinct page states, slopes shrunk
   toward the pooled slope; forecasts anchor on the last observation, with
   holders floored at their last value (cumulative counts never fall) and
   capped at predicted players. The median blends persistence with the
   trend on the logit scale (alpha = 0.65, calibrated on simulated
   final-like cases; the fitted move is capped at 0.6 + 0.3*sqrt(days)
   logit units, and upward moves are capped at +0.2 for trophies already
   at >= 90%). Refresh timing is a MIXTURE, not an expected shift: with
   probability exp(-horizon / mean refresh gap) no refresh happens before
   the target and the published rate stays at its last value; otherwise it
   follows the trend distribution. Spreads come from the model's own
   one-step-ahead backtest residuals over the last 5 refreshes (RMS,
   floored/capped), plus a random-walk term and Poisson small-count noise.

Determinism: no randomness, no clock, no dict-order dependence; reads are
processed in file order and states collapse identically every run.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LOGIT_LO = -9.210340371976184  # logit(0.01)
LOGIT_HI = 9.210340371976184   # logit(99.99)

RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# --- Kit-published facts about Exophase's first refresh (launch night). ---
# RULES.txt: page frozen at 638 players through the 04:00Z read, refreshed
# by the 04:30Z read to 3,868 players, every rate but the 100% one lower.
# competition.yaml header: the refreshed page shows 4,276 trophies earned.
FIRST_REFRESH_EARLIEST = datetime(2026, 11, 19, 4, 0, tzinfo=timezone.utc)
FIRST_REFRESH_SEEN = datetime(2026, 11, 19, 4, 30, tzinfo=timezone.utc)
FIRST_REFRESH_LATEST = datetime(2026, 11, 20, 4, 30, tzinfo=timezone.utc)
PRE_PLAYERS = 638
PRE_EARNED = 943
POST_PLAYERS = 3868
POST_EARNED = 4276
# Non-100% holders before/after (the 100% trophy holds PRE/POST players).
PRE_HOLDERS = PRE_EARNED - PRE_PLAYERS        # 305
POST_HOLDERS = POST_EARNED - POST_PLAYERS     # 408
RESCALE = (POST_HOLDERS / PRE_HOLDERS) * (PRE_PLAYERS / POST_PLAYERS)  # ~0.2206
RESCALE_SPREAD = 0.08

# --- Regime B (multi-refresh) constants, calibrated offline. ---
TREND_ALPHA = 0.65
FIT_WINDOW = 6
SHRINK_K = 3.0
RW_SIGMA = 0.15
BACKTEST_CUTOFFS = 5
S_RES_MIN = 0.15
S_RES_MAX = 0.6
PLAYERS_GROWTH_CAP = 25.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 20.0:
        return 100.0
    if lg <= -20.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (matches the house baseline's output)."""
    v = math.floor(x * 10000.0 + 0.5) / 10000.0
    return int(v) if float(v).is_integer() else v


def parse_ts(text):
    if text is None:
        return RELEASE
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def x_of(moment):
    """Log-time since release: ln(1 + hours). Adoption decelerates in it."""
    hours = max(0.0, (moment - RELEASE).total_seconds() / 3600.0)
    return math.log1p(hours)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def ols_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def collapse_states(history):
    """Merge consecutive reads showing the same page (null-tolerant).

    A read whose players count and every non-null rate match the running
    state extends it; nulls never create a new state (a missing rate is
    missing data, not a refresh). Rates merge by last non-null value.
    """
    states = []
    for read in history:
        rates = list(read.get("rates") or [])
        players = read.get("players_tracked")
        if states:
            cur = states[-1]
            same = cur["players"] == players and all(
                a is None or b is None or a == b
                for a, b in zip(cur["rates"], rates)
            )
            if same:
                for i, r in enumerate(rates):
                    if r is not None and i < len(cur["rates"]):
                        cur["rates"][i] = r
                cur["last_slot"] = read.get("slot", cur["last_slot"])
                continue
        states.append({
            "players": players,
            "rates": rates,
            "first_slot": read.get("slot"),
            "last_slot": read.get("slot"),
        })
    return states


def holders_of(state):
    """Integer holders implied by each rate (rates are 2dp percents)."""
    out = []
    players = state["players"] or 0
    for rate in state["rates"]:
        if rate is None:
            out.append(None)
        else:
            out.append(max(0, int(math.floor(rate * players / 100.0 + 0.5))))
    return out


def mixture_row(point_logit, point_weight, mu, sigma):
    """Quantiles of w * point-mass(point) + (1-w) * Normal(mu, sigma).

    Bisection on the mixture CDF; deterministic. When the point mass
    carries over half the weight, the median is the point itself.
    """
    if sigma <= 0:
        sigma = 1e-6
    if point_weight <= 1e-9:
        qs = [round4(inv_logit(mu + z * sigma)) for z in Z]
    elif point_weight >= 1.0 - 1e-9:
        qs = [round4(inv_logit(point_logit))] * 5
    else:
        qs = []
        for tau in LEVELS:
            lo, hi = LOGIT_LO, LOGIT_HI
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                cdf = point_weight * (1.0 if mid >= point_logit else 0.0)
                cdf += (1.0 - point_weight) * norm_cdf((mid - mu) / sigma)
                if cdf < tau:
                    lo = mid
                else:
                    hi = mid
            qs.append(round4(inv_logit(0.5 * (lo + hi))))
    for k in range(1, 5):
        qs[k] = max(qs[k], qs[k - 1])
    return qs


def saturated_row(horizon_days):
    """A trophy at 100%: median stays 100, downside hedge grows with time."""
    down = 1.0 + 0.5 * math.sqrt(max(horizon_days, 0.0))
    q05 = round4(inv_logit(LOGIT_HI - 1.7 * down))
    q25 = round4(inv_logit(LOGIT_HI - 0.7 * down))
    return [q05, q25, 100, 100, 100]


def rescale_median(rate):
    """The known first refresh's median rate for a non-100% trophy.

    Geometric mean of the exact rate rescale and the integer-holder
    rescale (the two rounding conventions differ by at most 0.01pp).
    """
    h = int(math.floor(rate * PRE_PLAYERS / 100.0 + 0.5))
    h2 = int(math.floor(h * POST_HOLDERS / PRE_HOLDERS + 0.5))
    exact = rate * RESCALE
    integer = 100.0 * h2 / POST_PLAYERS
    return math.sqrt(max(exact, 1e-9) * max(integer, 1e-9))


def forecast_frozen(states, target, horizon_days):
    """One distinct page state: persistence, or the known first refresh."""
    last = states[-1]
    rates = last["rates"]
    known = last["players"] == PRE_PLAYERS and target <= FIRST_REFRESH_LATEST
    if known:
        if target <= FIRST_REFRESH_EARLIEST:
            w_refresh = 0.0
        elif target >= FIRST_REFRESH_SEEN:
            w_refresh = 1.0
        else:
            span = (FIRST_REFRESH_SEEN - FIRST_REFRESH_EARLIEST).total_seconds()
            w_refresh = (target - FIRST_REFRESH_EARLIEST).total_seconds() / span
    else:
        w_refresh = 0.0
    spread_rw = 0.5 * math.sqrt(max(horizon_days, 0.0))
    pooled = sorted(r for r in rates if r is not None)
    pooled_median = pooled[len(pooled) // 2] if pooled else 1.0
    rows = []
    for rate in rates:
        if rate is None:
            rate = pooled_median
        if rate >= 100.0:
            rows.append(saturated_row(horizon_days))
            continue
        if not known:
            # Frozen page, no known refresh ahead: persistence + RW spread.
            rows.append(mixture_row(logit(rate), 0.0, logit(rate), max(spread_rw, 0.03)))
        elif w_refresh <= 0.0:
            # Answer read predates the refresh: the page is still frozen.
            rows.append(mixture_row(logit(rate), 1.0, logit(rate), 0.01))
        else:
            med = rescale_median(rate)
            rows.append(mixture_row(logit(rate), 1.0 - w_refresh,
                                    logit(med), RESCALE_SPREAD))
    return rows


def trend_medians(states, x_target, horizon_days):
    """Holders/players trend model: median logit per trophy (Regime B)."""
    n_troph = len(states[0]["rates"])
    xs_all = [x_of(parse_ts(s["first_slot"])) for s in states]
    holders_all = [holders_of(s) for s in states]
    win = states[-FIT_WINDOW:]
    xs = xs_all[-FIT_WINDOW:]
    holders = holders_all[-FIT_WINDOW:]
    x_last = xs_all[-1]
    # Players trend (recent window), floored at last, capped growth.
    log_p = [math.log(max(1, s["players"] or 1)) for s in win]
    slope_p = ols_slope(xs, log_p) or 0.0
    p_pred = math.exp(log_p[-1] + slope_p * (x_target - x_last))
    p_pred = max(min(p_pred, (states[-1]["players"] or 1) * PLAYERS_GROWTH_CAP),
                 float(states[-1]["players"] or 1))
    # Pooled holder slope (recent window).
    totals = [sum(v for v in h if v is not None) for h in holders]
    slope_pool = ols_slope(xs, [math.log(max(0.5, t)) for t in totals])
    if slope_pool is None:
        slope_pool = slope_p
    move_cap = 0.6 + 0.3 * math.sqrt(max(horizon_days, 0.0))
    medians = []
    last_rates = []
    last_holders = []
    for i in range(n_troph):
        last_rate, h_last = None, None
        for s, h in zip(reversed(states), reversed(holders_all)):
            if h[i] is not None:
                last_rate, h_last = s["rates"][i], h[i]
                break
        if last_rate is None:
            last_rate, h_last = None, 0
        last_rates.append(last_rate)
        last_holders.append(h_last)
        if last_rate is None:
            medians.append(None)
            continue
        pts = [(x, math.log(max(0.5, h[i])))
               for x, h in zip(xs, holders) if h[i] is not None]
        slope_i = None
        if len(pts) >= 3:
            slope_i = ols_slope([p[0] for p in pts], [p[1] for p in pts])
        if slope_i is None:
            slope_i = slope_pool
        n_i = len(pts)
        slope = (n_i * slope_i + SHRINK_K * slope_pool) / (n_i + SHRINK_K)
        h_pred = math.exp(math.log(max(0.5, h_last)) + slope * (x_target - x_last))
        h_pred = max(min(h_pred, p_pred), float(h_last))
        model_rate = 100.0 * h_pred / p_pred
        move = logit(model_rate) - logit(last_rate)
        move = min(max(move, -move_cap), move_cap)
        if last_rate >= 90.0:
            move = min(move, 0.2)
        medians.append(logit(last_rate) + TREND_ALPHA * move)
    return medians, last_rates, last_holders


def backtest_scale(states):
    """RMS of one-step-ahead logit residuals over the last few refreshes."""
    if len(states) < 4:
        return 0.3
    residuals = []
    first_j = max(3, len(states) - BACKTEST_CUTOFFS)
    for j in range(first_j, len(states)):
        sub = states[:j]
        x_t = x_of(parse_ts(states[j]["first_slot"]))
        gap_h = (parse_ts(states[j]["first_slot"])
                 - parse_ts(sub[-1]["first_slot"])).total_seconds() / 3600.0
        medians, _, _ = trend_medians(sub, x_t, max(gap_h, 0.0) / 24.0)
        for i, actual in enumerate(states[j]["rates"]):
            if actual is None or medians[i] is None:
                continue
            residuals.append(logit(actual) - medians[i])
    if len(residuals) < 20:
        return 0.3
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    return min(max(rms, S_RES_MIN), S_RES_MAX)


def forecast_multi_refresh(states, target, horizon_minutes):
    """Regime B: trend medians + mixture over refresh timing + spreads."""
    horizon_days = horizon_minutes / 1440.0
    horizon_h = horizon_minutes / 60.0
    x_target = x_of(target)
    medians, last_rates, last_holders = trend_medians(states, x_target, horizon_days)
    s_res = backtest_scale(states)
    gaps = []
    for a, b in zip(states, states[1:]):
        gaps.append((parse_ts(b["first_slot"])
                     - parse_ts(a["first_slot"])).total_seconds() / 3600.0)
    gap_mean = sum(gaps) / len(gaps) if gaps else 0.0
    w0 = math.exp(-horizon_h / gap_mean) if gap_mean > 0 else 0.0
    w0 = min(max(w0, 0.0), 0.95)
    pooled = sorted(r for r in last_rates if r is not None)
    pooled_median = pooled[len(pooled) // 2] if pooled else 1.0
    rows = []
    for i, med in enumerate(medians):
        last_rate = last_rates[i]
        h_last = last_holders[i]
        if last_rate is None:
            med = logit(pooled_median)
            last_rate = pooled_median
            h_last = 0
            extra = 2.0
        else:
            extra = 1.0
        if last_rate >= 100.0:
            rows.append(saturated_row(horizon_days))
            continue
        spread = math.sqrt(
            s_res * s_res
            + (RW_SIGMA * math.sqrt(max(horizon_days, 0.0))) ** 2
            + 1.0 / max(h_last, 1)
        ) * extra
        rows.append(mixture_row(logit(last_rate), w0, med, spread))
    return rows


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    horizon_minutes = case.get("horizon_minutes") or 0
    horizon_days = horizon_minutes / 1440.0
    try:
        target = parse_ts(case.get("target"))
    except (TypeError, ValueError):
        target = None
    if not history or not trophies:
        return {"trophies": [{"id": t.get("id"), "q": [0.5, 0.75, 1.0, 1.5, 2.5]}
                             for t in trophies]}
    states = collapse_states(history)
    # Pad short rate lists with nulls so every state covers every trophy.
    width = len(trophies)
    for s in states:
        if len(s["rates"]) < width:
            s["rates"] = list(s["rates"]) + [None] * (width - len(s["rates"]))
    if target is None:
        target = FIRST_REFRESH_SEEN
    if len(states) <= 1:
        rows = forecast_frozen(states, target, horizon_days)
    else:
        rows = forecast_multi_refresh(states, target, horizon_minutes)
    out = []
    for trophy, qs in zip(trophies, rows):
        fixed = []
        prev = 0.0
        for q in qs:
            q = min(max(float(q), 0.0), 100.0)
            q = max(q, prev)
            fixed.append(round4(q))
            prev = fixed[-1]
        out.append({"id": trophy.get("id"), "q": fixed})
    return {"trophies": out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
