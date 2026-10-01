# Working notes

Practical stuff: what is on disk, how to rebuild it, and the traps that cost me
hours. Numbers live in `results/release/RESULTS.md` and nowhere else. Anything
quoted in older files predates the audit rebuild and is superseded.

Read order if you are picking this up cold: `README.md`, then
`results/release/RESULTS.md`, then `results/STRATEGY_AUDIT.md` (the closure table
at the end says what I actually fixed), then `results/SUMMARY.md` for the
chronological log of everything tried, failures included.

## Where the numbers come from

`make reproduce` runs `scripts/build_release.py`, which rebuilds
`results/release/` from the data on disk and writes `manifest.json` with input
hashes, row counts and package versions. `make test` runs 18 regression tests:
ticker dates, bracket bounds, units, fees, decision time, the leakage cutoff, and
a hash of the frozen trade file. If a change moves a number, a test should fail
before the number reaches RESULTS.md.

Three defects the tests caught, each of which had been quietly inflating things:

1. Kalshi's API omits strike fields for the settled-YES market of many events, and
   for whole events before 2025. Those brackets had NaN bounds and the events were
   being dropped silently. Bounds now come from subtitles and tickers in
   `weather_core.py`, with `data/kalshi/exclusions.parquet` recording a reason code
   per exclusion.
2. "h=8" meant hours since the market opened, which was 22:00Z only 75% of the
   time. The release uses a clock-based D-1 22:00Z decision instead, which is 22Z
   100% of the time.
3. Polymarket bracket strips sum to about 1.03. The signal uses normalised prices;
   execution is charged at the raw midpoint. Not doing this was worth about 0.5c.

Withdrawn along the way: the Hasbrouck information share of 0.74 to 0.81 (see
REPORT §7), any dollar capacity figure for Polymarket, the word "pre-registered",
per-trade t-statistics as a headline, and Sharpe computed over active days only.

Still open, and not closeable with what is on disk: winter run-level forecast
timestamps (needs the NOAA AWS archives), historical executable books on either
venue, live/backtest feature parity in `live_signal.py`, an external timestamp for
the holdout chronology, and a second winter.

## Data on disk (about 2 GB, all free sources)

Not in the repo. `.gitignore` excludes `data/`, so this table is provenance rather
than something you can clone. The downloaders in `scripts/` will rebuild it, but
budget hours rather than minutes: both exchange APIs throttle.

| Path | What | Coverage | Notes |
|---|---|---|---|
| `data/kalshi/markets_*.pkl` | market metadata (ticker, strikes, result, expiration_value, rules) | 25 series: KXHIGH{NY,CHI,MIA,AUS,LAX,DEN,PHIL}, KXHIGHT{DAL,HOU,SFO,SEA,DC,BOS,ATL,PHX,LV,MIN,OKC}, KXLOWT{NYC,CHI,MIA,AUS,LAX,DEN,PHIL}; 2021-08 to 2026-09 | `/trade-api/v2/historical/markets` plus `/markets?status=settled` |
| `data/kalshi/candles/*.json` | hourly candlesticks (bid/ask/last OHLC, volume, OI) per market | complete: 73,902 files, every market above | `/historical/markets/{t}/candlesticks` for older, `/series/{s}/markets/{t}/candlesticks` for recent |
| `data/kalshi/candles_long.parquet` | parsed candles, one row per market-hour | rebuilt by `scripts/parse_candles.py` | |
| `data/kalshi/brackets.parquet` | bracket lo/hi per market | between/less/greater mapped to half-open intervals | |
| `data/kalshi/poly_temp_events.json` | Polymarket closed temperature events (gamma API) | 1,938 events, 2024-05 to 2026-03 | 2025 slugs have no year in them, use `endDate` |
| `data/kalshi/poly_prices/*.json` | Polymarket hourly CLOB price history per bracket token | 2,606 events: NYC-LGA, London City, Seoul-Incheon, Toronto, Dallas Love, Atlanta, Seattle, Chicago-ORD, Miami | `prices-history` needs `startTs`/`endTs` for settled markets |
| `data/kalshi/xvenue_*.parquet` | cross-venue panels | Mar 2025 to Sep 2026, 6 cities | rebuilt by `xvenue_build.py`, about 4 minutes |
| `data/weather/fcst*/*.json` | Open-Meteo previous-runs hourly temps, leads 0 to 3 days, 17 models | fcst: 7 core stations 2021-08 on; fcst2: 12 more models 2023-07 on; fcst3: 11 tier-2 stations 2024 on; fcst4: LGA, ORD, DAL, EGLC, RKSI, CYYZ 2024 on | lead-0 is look-ahead. `previous_day1` is issued 24h before valid time |
| `data/weather/fcst_daily.parquet` | daily max/min per station, model and lead | built by `build_weather2.py` | |
| `data/weather/runs/*.json` | Open-Meteo single-run forecasts (D-1 00Z, 06Z, 12Z inits) | Apr 15 to Sep 26 2026, 7 core stations | the only strictly run-time-safe forecasts; archive starts Apr 2026 |
| `data/weather/obs/cli_*.json` | NWS daily climate reports (official high/low, time of high) via IEM | 25 stations, 2021 or 2024 on | this is what Kalshi settles against |
| `data/weather/cli_daily.parquet` | unified daily truth table | `build_cli.py` plus manual appends | |
| `data/weather/comb_coefs.json` | fitted logistic coefficients (const, lmid, lpm) | h=8/10/12, all-year and winter | used by `live_signal.py` |
| `data/tennis/*`, `data/nba/*` | ATP odds 2000-2026, WTA 2007-19, TML match stats, NBA closing lines 2011-22 | | the graveyard; see SUMMARY |

IEM has no CLI product for KDAL, EGLC, RKSI or CYYZ, so those use `daily.py` ASOS
maxima instead. The server intermittently returns "over capacity"; retry.

## Rebuilding the weather pipeline

```
kalshi_list_markets.py <series,...> <name>    # metadata -> data/kalshi/markets_<name>.pkl
kalshi_candles3.py <threads> <interval> <grp> # candles, winter first, skips existing files
rebuild.sh                                    # parse_candles -> weather_core (events, truth,
                                              #   target date from ticker) -> weather_snap
build_weather2.py                             # forecast archives -> fcst_daily.parquet
build_cli.py                                  # CLI json -> cli_daily.parquet
weather_edge2.py w ext [low] [hours]          # the walk-forward backtest
stress.py                                     # delays, worst fills, halves, placebos
live_signal.py <thr>                          # live paper signals, 18 cities
```

Release build:

```
edge_lib.py        # shared: paths, fee, ticker_date, bracket_bounds, clustered_t,
                   #   block bootstrap, capital_sharpe, build_blend(ols|ridge)
release_weather.py # strategy 1 -> results/release/weather.json + weather_trades.parquet
release_xvenue.py  # strategy 2 -> results/release/xvenue.json + xvenue_trades.parquet
build_release.py   # both, plus manifest.json and RESULTS.md
make_figures.py    # the README figures, from the frozen trade files
```

To add a Kalshi series: add it to `SER` in `weather_core.py` (station code, CLI
station, timezone, kind), list markets, download candles, pull forecasts for the
station coordinates (copy `openmeteo_download3.py`), get CLI from
`mesonet.agron.iastate.edu/json/cli.py?station=KXXX&year=YYYY` or ASOS maxima from
`daily.py`, add the station to the `CLI` dicts in `weather_edge2.py` and
`live_signal.py`, then run `rebuild.sh`.

To add a Polymarket city: add it to `CITY` in `poly_edge.py`, download forecasts
for the station, append truth to `cli_daily.parquet`.

## Traps

Each of these produced a fake edge before I found it.

1. Kalshi's `close_time` ignores DST, which put the target date off by one for
   roughly 40% of summer events. Parse the date out of the ticker instead
   (`KXHIGHNY-26JUL29`).
2. Open-Meteo's lead-0 daily max includes same-day model runs from 12Z to 18Z,
   which is pure look-ahead. `previous_day1` is only safe from about D-1 22Z.
   Snapshots at the D-1 open (14Z to 20Z) looked profitable and were not: a strict
   single-run test found nothing there.
3. Taking an implied mean from bracket centres manufactures a spurious
   underreaction signal, because the tails are open-ended. Test at the bracket
   level.
4. Polymarket: `prices-history?interval=max` returns nothing for settled markets;
   2025 slugs lack the year; London brackets were Fahrenheit in 2025 and Celsius in
   2026, so read the unit off each market title.
5. A high daily Sharpe here is diversification across about 7 small bets a day,
   not a big per-trade edge. Per-trade IR is 0.14. Report per-contract edge and a
   clustered t-statistic.
6. Kalshi's tier-2 series (ATL, SEA, DAL and friends) did not stop in March 2026.
   The market listing was stale. Re-list before concluding a series ended.
7. Placebos that permute days have to shift timestamps by the same day offset, or
   the merge silently returns nothing and you "fail" the placebo for free.

## Not tried yet, roughly in order of how promising it looks

1. **Kalshi sports prices against sharp-book closing lines.** `markets_tennis.pkl`
   plus candles for 18,710 ATP/WTA match markets are on disk. The hypothesis is
   that the retail venue lags Pinnacle and Bet365 closing odds. Match on surname
   and date, compare Kalshi's last pre-match price to the closing implied
   probability, trade at the ask with the `0.07*P*(1-P)` fee. Same shape as the
   cross-venue result. The earlier tennis work showed Pinnacle's closing line is
   itself efficient, so any edge is in the venue gap.
2. **Run-level model archives for a strict winter test.** NOAA GFS, HRRR and NBM on
   AWS Open Data have every run since 2020; byte-range fetch just the 2m
   temperature field. This would settle the timing question and also test whether
   trading straight after each new run beats the 22Z snapshot. The freshness
   mechanism suggests it would.
3. **Cross-venue extensions.** London, Seoul and Toronto have Polymarket markets and
   Kalshi now lists KXHIGHT{EHAM,WSSS,OMDB,VABB,ZBAA,SBGR} and
   KXLOWT{CYYZ,RKSI}. Add the low-temperature pairs, measure the 2026-27 winter out
   of sample, and price the settlement-mismatch risk of a two-legged hedge
   explicitly.
4. **Polymarket depth.** The CLOB `/book` endpoint for signalled brackets, plus
   `data-api.polymarket.com/trades` for realised fills. No fee there, so the
   maker/taker economics differ from Kalshi.
5. **Hourly directional temperature markets** (`KXTEMP*H`, `KXHIGHNYD`): same models,
   HRRR hourly forecasts, retail-heavy, untested.
6. **Precipitation and snowfall** (`KXRAIN*`, `*SNOWM`): NBM PoP and QPF against the
   market. Settlement quirks around trace amounts may be where the mispricing is.
7. **Economic-data markets against public nowcasts:** Kalshi CPI, NFP and Fed
   markets against the Cleveland Fed inflation nowcast and GDPNow. Few events per
   year, so power is the problem, but each market is large.
8. **Opening rather than closing lines in sports.** Only closing was ever tested.
   Pinnacle opening odds are not in the free data.
9. **A second winter.** The single biggest open question, and nothing to do about it
   but run `live_signal.py` daily from November.
