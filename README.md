# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Yogii123-qw/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Yogii123-qw/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each of the 55 Grand Theft Auto VI PlayStation 5 trophies,
as five quantiles of its Exophase earn rate at the case target.
Entry for make.fun competition `gta6-trophy-rarity-forecast-01`.

## Run

```
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only. No network, no clock, no
randomness. One case runs in under 0.5 seconds on one CPU.

## Method

The forecast is a blend, in logit space, of two members.

Cohort member (weight 0.75). Players arrive over launch week and each
trophy has a time to earn. Earners for a trophy at time t are the
arrivals convolved with a lognormal delay distribution with median mu
and sigma 0.7. The players curve comes from the case snapshots (log
players against log time, quadratic fit, linear extrapolation from
the end slope). Each trophy mu is fitted by grid search on the
snapshot logit error, then shrunk toward its tier median mu. The
forecast rate is earners over players at the target. This keeps
dilution (players joining) apart from progression (the cohort
earning), which a rate-only fit mixes together.

Trend member (weight 0.25). The house curve baseline from the rules:
least squares of logit(rate) on ln(1 + hours since release) over the
last 24 hours of reads, read at the target. It is the hedge for the
case where the cohort delay shape is wrong.

Spread is 0.45 logit from the cohort calibration, widened by half the
distance between the two members medians. Floor 0.30, cap 1.10.

Fallbacks. Flat history (one snapshot): the first refresh documented
in the rules, players 638 to 3868 with earners grown by 1.25, spread
0.45. Fewer than 3 snapshots: persistence median with the house
spread. A trophy at 100 percent in every snapshot keeps the row
[99.0, 99.99, 99.99, 99.99, 99.99].

## Validation

There is no local grade for this competition: the kit states the
scorer and reference data are not published, so grading happens on
the server only. The numbers below are honest substitutes, each
labeled as such.

Practice backtest (p00/p01), under the refresh answer implied by the
rules (earners held across the first refresh, rates rescaled by
638/3868; the rules publish the house starter at loss about 1.51 and
miss about 0.68, and the rescale assumption reproduces 1.69 and 0.73,
so the truth sits near earners grown by 1.2 to 1.3):

| entry | loss | typical miss | 90% coverage |
| --- | --- | --- | --- |
| house starter | 1.6877 | 0.73 points | 0.0000 |
| this entry | 0.1342 | 0.04 points | 0.9818 |

Synthetic final-like backtest (5 fixed seeds, 2.5 day history, 4.5
day horizon, known ground truth, pinball scorer verified against the
rules worked example 0.403404):

| entry | mean loss |
| --- | --- |
| house persistence | 1.4612 |
| house trend | 0.7178 |
| this entry | 0.2459 |

The same cohort blend was also tested on four synthetic world
families (power-law and logistic arrivals, fast and slow trophy
delays, sharp delays): it beat the house trend in every family and
kept 90% coverage near 0.89 on the base family. Synthetic results
test the estimator mechanics, not the real week.

Checks run: output shape (55 rows, non-decreasing quantiles, 0 to
100), determinism (two runs byte-identical), edge cases (2-snapshot
history, null rates, single read), runtime 0.37 seconds on a full
final-like case against the 8 second cap, output 3008 bytes against
the 65536 byte cap.

## Status

Built, measured, and reviewed. Not submitted. Submission waits for
explicit approval.
