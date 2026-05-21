# US / Global Futures Scanner

Same trading-signal logic as
[Allanli1011/cn_stock_scan](https://github.com/Allanli1011/cn_stock_scan)
(MACD triple divergence + PA three-push 75% retracement + ICT weekly/monthly
PDA), but operating on **every continuous front-month futures contract on
Yahoo Finance** (`=F` symbols) instead of A-shares.

The universe covers ~57 contracts across Energy, Metals, Grains, Softs,
Livestock, Equity Indices, Currencies, Interest Rates, Volatility and Crypto.

## Composite scoring (max 3.0)

| Signal | Score | Notes |
|---|---|---|
| MACD strict 5/5 divergence | +1.0 | Purple markers on chart |
| MACD loose 4/5 divergence | +0.5 | Orange markers |
| Three-push w/ 60-90% retracement | +1.0 | H1/H2/H3 markers |
| Weekly/Monthly OB+FVG overlap | +1.0 | "Premium-discount array" |
| Weekly/Monthly single OB or FVG | +0.5 | |

Scores ≥ 2.5 are flagged as **triple convergence**.

## Quick start

```bash
pip install -r requirements.txt

# First run: build universe + download daily history
python scripts/scan_full.py --refresh-data --min-score 1.5 --plot-top 5

# Subsequent runs: incremental price update only
python scripts/scan_full.py --update-prices --min-score 1.5

# Scan one category
python scripts/scan_full.py --category Energy --plot-top 3
python scripts/scan_full.py --category Metals
python scripts/scan_full.py --category Grains

# Only short signals
python scripts/scan_full.py --direction top
```

## Outputs

* `output/futures_signals_<date>.csv` — every contract that meets `--min-score`,
  sorted by composite score
* `output/charts/<date>_<ticker>_<direction>.png` — annotated candlestick
  chart with MACD subplot, three-push markers, PDA zone, and entry/stop/target
  trade plan

## Daily automation (GitHub Actions)

A scheduled workflow at `.github/workflows/daily_scan.yml` runs the scan
**every weekday at 23:00 UTC** (≈ 6 pm ET / 7 pm EDT, after US futures
settle). It commits the day's CSV + top-5 charts to a `signals/` folder
in the repo and also uploads them as a 90-day artifact.

* Browse historical signals → [`signals/`](signals/) folder
* Trigger an ad-hoc run → Actions tab → **Daily Futures Scan** → "Run workflow"
* Adjust the schedule, score threshold, or plot count → edit the workflow
  file (`cron`, `--min-score`, `--plot-top`)

For the auto-commit to work, **Settings → Actions → General → Workflow
permissions** must be set to *"Read and write permissions"*. The workflow
declares `permissions: contents: write` so this is the only repo-level
toggle needed.

## Universe

The list of futures is in `src/universe.py` (`FUTURES_UNIVERSE`). Yahoo
Finance occasionally retires or renames a contract; if a ticker stops
returning data, just delete its row.

| Category | Examples |
|---|---|
| Energy | CL=F WTI, BZ=F Brent, NG=F Nat Gas, HO=F, RB=F |
| Metals | GC=F Gold, SI=F Silver, HG=F Copper, PL=F, PA=F |
| Grains | ZC=F Corn, ZS=F Soybeans, ZW=F Wheat, ZL=F, ZM=F, ZO=F, ZR=F |
| Softs | KC=F Coffee, SB=F Sugar, CC=F Cocoa, CT=F Cotton, OJ=F |
| Livestock | LE=F Live Cattle, HE=F Lean Hogs, GF=F Feeder Cattle |
| Index | ES=F S&P, NQ=F Nasdaq, YM=F Dow, RTY=F Russell, NKD=F |
| Currency | 6E=F Euro, 6J=F Yen, 6B=F GBP, DX=F US Dollar Index |
| Rates | ZN=F 10Y, ZB=F 30Y, ZF=F 5Y, ZT=F 2Y, TN=F, UB=F |
| Crypto | BTC=F Bitcoin, ETH=F Ether |

`DX=F` (US Dollar Index) and `VX=F` (VIX) currently return no data from
Yahoo Finance, so they're excluded. Use the index/ETF proxies (`DX-Y.NYB`,
`^VIX`, `UUP`, `VXX`) if you need that exposure.

## Key parameters (config.yaml)

| Setting | Default | Notes |
|---|---|---|
| `swing.pct_threshold` | 0.03 | ZigZag threshold — bump to 0.04–0.05 for NG=F/VX=F if noisy |
| `macd.divergence.min_area_reduction` | 0.10 | Each histogram leg must shrink ≥10% |
| `macd.divergence.recency_bars` | 30 | 3rd push must be within last 30 daily bars |
| `three_push.pullback_target_pct` | 0.75 | 75% retracement target |
| `three_push.pullback_tolerance` | 0.15 | Accept pullbacks in [60%, 90%] |
| `ob_fvg.ob_displacement_atr` | 2.0 | OB requires 2×ATR follow-through |
| `prices.lookback_days` | 800 | ~3 years of daily bars |

## Differences vs cn_stock_scan

* **Data source**: yfinance instead of akshare
* **Universe**: curated futures list (no market-cap filter)
* **Weekly resample anchor**: `W-SUN` (futures trade Globex Sun-Fri) vs `W-FRI`
* **Color convention**: green=up, red=down (US convention; A-shares use the
  opposite)
* **Currency symbol**: `$` instead of `¥`
* **Output language**: English

The MACD divergence rules, three-push retracement logic, and OB/FVG/PDA
mechanics are byte-for-byte the same.
