"""Forecast GTA VI PS5 trophy earn rates on Exophase, one week after launch.

Model: a cohort blend, in logit space, of two members.

Member 1, cohort (weight 0.75). Players arrive over launch week and
each trophy has a time to earn. Earners for a trophy at time t are the
arrivals convolved with a lognormal delay distribution: a player who
arrived at s has earned it by t with probability Phi((ln(t - s) -
ln mu) / 0.7), where mu is the trophy median delay in hours. The
players curve comes from the snapshots (log players against log time,
quadratic fit, linear extrapolation from the end slope). Each trophy
mu is fitted by grid search on the snapshot logit error, then shrunk
toward its tier median mu. The forecast rate is earners over players
at the target. This separates dilution (players joining) from
progression (the cohort earning), which a rate-only fit confuses.

Member 2, house trend (weight 0.25). The rules house curve baseline:
ordinary least squares of logit(rate) on ln(1 + hours since release)
over the last 24 hours of reads, read at the target. It is the hedge:
when the cohort delay shape is wrong, the trend member pulls the
median back.

Spread: 0.45 logit from the cohort calibration, widened by half the
distance between the two members medians. Floor 0.30, cap 1.10.

Fallbacks. Flat history (one snapshot): the first refresh documented
in the rules, players 638 to 3868 with earners grown by 1.25 (that
factor reproduces the rules published house starter practice
aggregates, loss about 1.51 and miss about 0.68), spread 0.45. Fewer
than 3 snapshots: persistence median with the house spread. A trophy
at 100 percent in every snapshot keeps [99.0, 99.99, 99.99, 99.99,
99.99]: the rate survived a sixfold player jump on launch night, so
it is structural, and 99.0 is the cheap lower hedge.

Standard library only. Deterministic: sorted reads, fixed float ops,
no clock, no randomness, no network. Reads every *.in in the input
directory and writes one *.out per case.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

FIRST_REFRESH_PLAYERS = 3868
STALE_EARNERS_GROWTH = 1.25
FLAT_SPREAD = 0.45

COHORT_WEIGHT = 0.75
COHORT_SPREAD = 0.45
COHORT_SIGMA = 0.7
MU_GRID = [1.5 ** k for k in range(-2, 22)]
SPREAD_FLOOR = 0.30
SPREAD_CAP = 1.10
HOUSE_SPREAD_PER_ROOT_DAY = 0.5
MEDIAN_MOVE_CAP_LOGIT = 3.0
SHRINK_PRIOR_COUNT = 4.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    text = text.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def log_time(moment):
    hours = (moment - RELEASE).total_seconds() / 3600.0
    return math.log(max(0.01, 1.0 + hours))


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def ols(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = n - 2
    s2 = sum(r * r for r in resid) / dof if dof > 0 else 0.0
    return {"n": n, "intercept": intercept, "slope": slope, "s2": s2,
            "xmean": mx, "sxx": sxx}


def ols_quad(xs, ys):
    """Quadratic least squares. Returns coefficients in centered x."""
    n = len(xs)
    if n < 4:
        return None
    mx = sum(xs) / n
    xc = [x - mx for x in xs]
    mat = [[0.0] * 3 for _ in range(3)]
    vec = [0.0] * 3
    for xi, yi in zip(xc, ys):
        row = [1.0, xi, xi * xi]
        for i in range(3):
            vec[i] += row[i] * yi
            for j in range(3):
                mat[i][j] += row[i] * row[j]
    aug = [mat[i] + [vec[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for r in range(col + 1, 3):
            f = aug[r][col] / aug[col][col]
            for c in range(col, 4):
                aug[r][c] -= f * aug[col][c]
    coef = [0.0] * 3
    for i in (2, 1, 0):
        coef[i] = (aug[i][3]
                   - sum(aug[i][j] * coef[j] for j in range(i + 1, 3))) / aug[i][i]
    a, b, c = coef
    resid = [y - (a + b * xi + c * xi * xi) for xi, y in zip(xc, ys)]
    dof = n - 3
    s2 = sum(r * r for r in resid) / dof if dof > 0 else 0.0
    return {"a": a, "b": b, "c": c, "s2": s2, "n": n, "mx": mx,
            "xmax": max(xs)}


def quad_value(fit, x):
    xc = x - fit["mx"]
    return fit["a"] + fit["b"] * xc + fit["c"] * xc * xc


def quad_boundary_value(fit, x):
    """Quadratic value inside the observed range; past xmax, linear
    extrapolation from the fitted value and slope at xmax."""
    if x <= fit["xmax"]:
        return quad_value(fit, x)
    xc_max = fit["xmax"] - fit["mx"]
    val_max = fit["a"] + fit["b"] * xc_max + fit["c"] * xc_max * xc_max
    slope_end = fit["b"] + 2 * fit["c"] * xc_max
    slope_end = min(3.0, max(-1.0, slope_end))
    return val_max + slope_end * (x - fit["xmax"])


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def snapshots(case):
    seen = []
    for read in case.get("history", []):
        players = read.get("players_tracked")
        rates = read.get("rates")
        if players is None or rates is None:
            continue
        key = (players, tuple(rates))
        if seen and seen[-1][0] == key:
            continue
        moment = parse_time(read["at"])
        seen.append((key, moment, players, list(rates)))
    return [(moment, players, rates) for _, moment, players, rates in seen]


def last_known_rate(snaps, idx):
    for _, _, rates in reversed(snaps):
        value = rates[idx]
        if value is not None:
            return value
    return None


def quantiles_from_logit(median_logit, spread):
    values = []
    for z in Z:
        values.append(round4(clamp_pct(inv_logit(median_logit + z * spread))))
    for i in range(1, len(values)):
        if values[i] < values[i - 1]:
            values[i] = values[i - 1]
    return values


def trend_member(case, snaps):
    """House trend baseline medians in logit space, per trophy."""
    at = parse_time(case["at"])
    cut = at - timedelta(hours=24)
    target = parse_time(case["target"])
    x_target = log_time(target)
    medians = []
    for idx in range(len(case["trophies"])):
        xs, ys = [], []
        for read in case.get("history", []):
            moment = parse_time(read["at"])
            if moment < cut:
                continue
            rate = read["rates"][idx]
            if rate is None:
                continue
            xs.append(log_time(moment))
            ys.append(logit(rate))
        last_rate = last_known_rate(snaps, idx)
        if last_rate is None:
            last_rate = 0.01
        fit = ols(xs, ys) if len(xs) >= 3 else None
        if fit is None:
            medians.append(logit(last_rate))
            continue
        med = fit["intercept"] + fit["slope"] * x_target
        last_logit = logit(last_rate)
        move = med - last_logit
        if move > MEDIAN_MOVE_CAP_LOGIT:
            med = last_logit + MEDIAN_MOVE_CAP_LOGIT
        elif move < -MEDIAN_MOVE_CAP_LOGIT:
            med = last_logit - MEDIAN_MOVE_CAP_LOGIT
        medians.append(med)
    return medians


def cohort_member(case, snaps):
    """Cohort model medians in logit space, per trophy, or None."""
    if len(snaps) < 3:
        return None
    trophies = case["trophies"]
    xs = [log_time(m) for m, _, _ in snaps]
    target = parse_time(case["target"])
    t_target = (target - RELEASE).total_seconds() / 3600.0
    snap_hours = [(m - RELEASE).total_seconds() / 3600.0
                  for m, _, _ in snaps]
    # Players curve.
    log_players = [math.log(max(1.0, float(p))) for _, p, _ in snaps]
    quad = ols_quad(xs, log_players)
    lin = ols(xs, log_players)

    def players_at(t_hours):
        x = math.log(max(0.01, 1.0 + t_hours))
        if quad is not None:
            value = quad_boundary_value(quad, x)
        elif lin is not None:
            value = lin["intercept"] + lin["slope"] * x
        else:
            return float(snaps[-1][1])
        return math.exp(min(700.0, value))

    p_target = players_at(t_target)
    snap_players = [players_at(h) for h in snap_hours]
    # Hourly arrivals from the fitted players curve.
    n_grid = int(math.ceil(t_target)) + 1
    arrivals = []
    prev = players_at(-0.5)
    for h in range(n_grid):
        cur = players_at(float(h))
        arrivals.append(max(0.0, cur - prev))
        prev = cur
    # Earners trajectory per mu, at snapshot times and at the target.
    # F(age, mu) with integer ages 0..168.
    max_age = 168
    cdf_rows = {}
    for mu in MU_GRID:
        row = [0.0] * (max_age + 1)
        log_mu = math.log(mu)
        for age in range(1, max_age + 1):
            row[age] = norm_cdf((math.log(age) - log_mu) / COHORT_SIGMA)
        cdf_rows[mu] = row
    times = snap_hours + [t_target]
    earners_by_mu = {}
    for mu in MU_GRID:
        row = cdf_rows[mu]
        traj = []
        for t_hours in times:
            total = 0.0
            for s, cohort in enumerate(arrivals):
                if cohort <= 0:
                    continue
                age = int(t_hours - s)
                if age <= 0:
                    continue
                if age > max_age:
                    age = max_age
                total += cohort * row[age]
            traj.append(total)
        earners_by_mu[mu] = traj
    # Fit mu per trophy on snapshot logit error.
    mus = []
    tier_log_mu = {}
    for idx, tr in enumerate(trophies):
        obs = []
        for k, (_, _, rates) in enumerate(snaps):
            rate = rates[idx]
            if rate is not None:
                obs.append((k, rate))
        if not obs:
            mus.append(None)
            continue
        best_mu, best_err = None, None
        for mu in MU_GRID:
            traj = earners_by_mu[mu]
            err = 0.0
            for k, rate in obs:
                p = snap_players[k]
                pred = 100.0 * traj[k] / p if p > 0 else 0.0
                err += (logit(pred) - logit(rate)) ** 2
            err /= len(obs)
            if best_err is None or err < best_err:
                best_mu, best_err = mu, err
        mus.append(best_mu)
        tier_log_mu.setdefault(tr.get("tier"), []).append(math.log(best_mu))
    tier_med = {t: median(v) for t, v in tier_log_mu.items()}
    all_mu = [math.log(m) for m in mus if m is not None]
    global_mu = median(all_mu) if all_mu else math.log(50.0)
    medians = []
    for idx, tr in enumerate(trophies):
        last_rate = last_known_rate(snaps, idx)
        if last_rate is None:
            last_rate = 0.01
        if mus[idx] is None:
            medians.append(logit(last_rate))
            continue
        n_obs = sum(1 for _, _, rates in snaps if rates[idx] is not None)
        w = n_obs / (n_obs + SHRINK_PRIOR_COUNT)
        tier_m = tier_med.get(tr.get("tier"), global_mu)
        mu_shrunk = math.exp(w * math.log(mus[idx]) + (1 - w) * tier_m)
        mu_use = min(MU_GRID,
                     key=lambda m: abs(math.log(m) - math.log(mu_shrunk)))
        e_hat = min(earners_by_mu[mu_use][-1], p_target)
        med_rate = 100.0 * e_hat / p_target if p_target > 0 else last_rate
        med = logit(med_rate)
        last_logit = logit(last_rate)
        move = med - last_logit
        if move > MEDIAN_MOVE_CAP_LOGIT:
            med = last_logit + MEDIAN_MOVE_CAP_LOGIT
        elif move < -MEDIAN_MOVE_CAP_LOGIT:
            med = last_logit - MEDIAN_MOVE_CAP_LOGIT
        medians.append(med)
    return medians


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(name + " asks for levels " + repr(case.get("levels")))
    trophies = case["trophies"]
    snaps = snapshots(case)
    if not snaps:
        raise ValueError(name + " has no usable history")
    horizon_days = case["horizon_minutes"] / 1440.0
    house_spread = HOUSE_SPREAD_PER_ROOT_DAY * math.sqrt(horizon_days)
    last_players = snaps[-1][1]
    n_snap = len(snaps)

    trend_med = trend_member(case, snaps)
    cohort_med = cohort_member(case, snaps) if n_snap >= 3 else None

    rows = []
    for idx, trophy in enumerate(trophies):
        last_rate = last_known_rate(snaps, idx)
        if last_rate is None:
            rows.append({"id": trophy["id"],
                         "q": quantiles_from_logit(logit(0.01), SPREAD_CAP)})
            continue
        if all(rates[idx] == 100 for _, _, rates in snaps
               if rates[idx] is not None) and any(
                   rates[idx] is not None for _, _, rates in snaps):
            rows.append({"id": trophy["id"],
                         "q": [99.0, 99.99, 99.99, 99.99, 99.99]})
            continue
        if n_snap == 1:
            median_rate = (last_rate * last_players / FIRST_REFRESH_PLAYERS
                           * STALE_EARNERS_GROWTH)
            rows.append({"id": trophy["id"],
                         "q": quantiles_from_logit(
                             logit(median_rate), FLAT_SPREAD)})
            continue
        if cohort_med is None:
            rows.append({"id": trophy["id"],
                         "q": quantiles_from_logit(
                             logit(last_rate), house_spread)})
            continue
        med = (COHORT_WEIGHT * cohort_med[idx]
               + (1.0 - COHORT_WEIGHT) * trend_med[idx])
        gap = abs(cohort_med[idx] - trend_med[idx])
        spread = math.sqrt(COHORT_SPREAD ** 2 + (0.5 * gap) ** 2)
        spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))
        rows.append({"id": trophy["id"],
                     "q": quantiles_from_logit(med, spread)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
