# makefun-pumpfun-top5-forecast

Forecast the top 5 new pump.fun coins by market cap, 24 hours out
(make.fun bounty `pumpfun-top5-forecast-01`).

## What it does

`main.ts` (Node 22, standard library only, no network, no model calls) reads
`/input/history.ndjson` + `/input/task.json` (or `$MAKEFUN_INPUT_DIR`) and
writes `/output/forecast.json` (or `$MAKEFUN_OUTPUT_DIR`):

- For every coin in the history it fits a drift (log-cap slope per hour) and a
  volatility on log10(market cap in SOL), with a small per-hour crash hazard.
- A fixed-seed Monte Carlo (3,000 paths) projects every candidate coin to each
  of the seven answer reads — drift decays with a 3h time constant, noise grows
  like sqrt(dt) — and ranks the coins at each simulated read.
- Place probabilities and size-bin distributions are the Monte Carlo averages
  over all paths and all seven reads. Each place leaves ~30% unnamed for the
  coins born after the history cut, which no forecast can name.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Two runs on the same history produce byte-identical output (fixed seed).

## Account

- Owner: tiafatiara62-coder
- Referral: r6xcxcc9
