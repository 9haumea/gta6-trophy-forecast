# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/9haumea/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/9haumea/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A structural forecaster for GTA VI trophy earn rates on Exophase, entered in make.fun's
"Forecast how many GTA VI players earn each trophy in week one" bounty.

An earn rate is holders / players-tracked. Exophase refreshes stepwise, and during launch
week the tracked-player pool grows far faster than most trophies' holder counts, so most
rates fall by dilution while a few early-story trophies climb. This program reads the
case's recorded history, collapses it into refresh steps, and forecasts five quantiles
(5%, 25%, 50%, 75%, 95%) of each trophy's rate at the case's target:

- tracked-player growth N(t): a damped logistic/Gompertz ensemble fitted to the refresh
  steps, extrapolated to the target with horizon-widening uncertainty;
- per-trophy holder growth H_i(t): monotone saturating fits with shrinkage toward sticky
  holders, so rare trophies stay rare and story trophies progress;
- a launch-night refresh prior for flat early histories (the documented first refresh),
  and refresh-staleness handling at the target;
- quantiles on the logit scale, honest spread that widens with the horizon.

Python 3.13, standard library only, deterministic, no network. Runs once per case:
reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input`), writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=dhbjkc63&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
