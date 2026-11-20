# Refresh-Rescale Mixture Forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/veneraxyz/gta6-trophy-rarity-forecast-01-entry-v2/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/veneraxyz/gta6-trophy-rarity-forecast-01-entry-v2/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Entry for the [make.fun](https://make.fun/c/gta6-trophy-rarity-forecast-01)
competition **gta6-trophy-rarity-forecast-01**: forecast, for every GTA VI
PlayStation 5 trophy (55), the percent of Exophase-tracked players who will
have earned it at 2026-11-26T00:00:00Z, as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95) per trophy. Scored by mean pinball loss on
the logit scale.

## How it works

Exophase refreshes a game's counts on its own schedule, hours apart, so the
published page is a frozen snapshot that jumps at each refresh. This
forecaster treats the refresh as the first-class event it is:

- **Straddled known first refresh.** When the case history is the frozen
  launch-night page (638 players tracked) and the target sits at or after
  Exophase's first refresh (04:30Z, 19 Nov 2026), the refresh is a
  near-deterministic rescale, not a random walk. The kit publishes the
  refreshed aggregates — 3,868 players tracked, 4,276 trophies earned —
  so non-100% holders grow 305 → 408 while players grow 638 → 3,868, and
  every non-100% rate is multiplied by about 0.2206. The median averages
  the two rounding conventions (exact rate rescale, integer-holder
  rescale); the spread is a tight 0.08 logit units. Fingerprint: this
  factor reproduces the published house-baseline practice numbers
  (typical miss 0.68pp, coverage 0, ln(1/0.2206) = 1.51 ≈ its ~1.51 loss).
  The one trophy at 100% stays at 100%, with a small downside hedge.
- **Frozen page, no known refresh ahead.** Persistence with a random-walk
  spread — honest ignorance.
- **Refreshes observed (the final-like case).** A holders/players
  decomposition: holders and players are fit log-linear in ln(1 + hours
  since release) over the last six distinct page states, slopes shrunk
  toward the pooled slope, anchored on the last observation (holders
  floored at their last value, since cumulative counts never fall). The
  median blends persistence with the trend on the logit scale
  (alpha = 0.65, calibrated on simulated final-like cases), with the
  fitted move capped. Refresh timing is a **mixture**, not an expected
  shift: with probability exp(-horizon / mean refresh gap) no refresh
  happens before the target and the rate stays put; otherwise it follows
  the trend distribution. Spreads come from the model's own one-step-ahead
  backtest residuals over the last few refreshes, plus a random-walk term
  and Poisson small-count noise.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only, no network) reads every
`*.in` case in the input directory and writes one `*.out` forecast per case
to the output directory. It is deterministic: two runs produce byte-identical
output.

## Files

- `main.py` — the forecaster (see its header comment for the full model).
- `README.md` — this file.
- `LICENSE` — MIT.

## License

MIT — see `LICENSE` (`SPDX-License-Identifier: MIT`).
