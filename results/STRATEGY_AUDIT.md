# Surviving-strategy audit: problems, risks, and closure criteria

Date: 2026-09-28

Scope: this document audits only the two strategies currently presented as winners:

1. The weather-forecast blend traded against Kalshi and transferred to Polymarket.
2. The Kalshi-to-Polymarket cross-venue price-discovery signal.

It does not discuss discarded hypotheses. The purpose is to identify everything that could make the reported winning results incorrect, overstated, non-reproducible, non-executable, or unsafe to describe as deployable.

## Executive verdict

Both strategies are promising research results. Neither is presently proven live, foolproof, or production-robust.

The highest-priority blockers are:

1. **Weather forecast availability is not proven for the winter sample.** The strongest weather result may still contain forecast-timing contamination.
2. **The saved report does not reproduce from the current data and code.** The headline weather result changes from 4,023 trades at +5.6 cents to 6,970 signals at +4.89 cents on a current rerun.
3. **Historical execution is approximated, not demonstrated.** Neither strategy has historical depth or actual-fill data sufficient to support the capacity claims.
4. **Inference treats correlated positions too much like independent trades.** Event/day-clustered inference must replace per-trade t-statistics everywhere.
5. **The cross-venue “price discovery” interpretation is not yet fully identified.** The venues often settle against different sources, prices are asynchronously sampled, and the information-share calculation needs a more standard VECM implementation.
6. **The codebase is exploratory research code.** It lacks a frozen data manifest, pinned environment, tests, CI, portable paths, and a single deterministic report build.

Performance language should therefore use **historical backtest**, **simulated signal**, **walk-forward**, and **post-specification holdout**. It should not use **earned**, **shipped**, **foolproof**, **proven deployable**, or **live strategy**.

## Severity definitions

- **P0 — validity blocker:** could invalidate the edge or make the headline result unverifiable.
- **P1 — material:** probably does not erase the entire signal, but can materially reduce confidence, P&L, or the strength of the claim.
- **P2 — hardening:** necessary for a polished research artifact or safe operational use.

---

# 1. Shared problems

## 1.1 The report, tear sheet, and current pipeline disagree — P0

The saved artifacts report a weather baseline of 4,023 trades at +5.6 cents per contract. Running the current `weather_final.py` against the current data produces:

- 6,970 signals;
- +4.89 cents per contract;
- 8.2% simulated ROI;
- daily one-lot Sharpe 7.17;
- +3.39 cents under the next-hour, additional-1-cent-slippage, nonzero-volume stress.

Other stale statements include the market count, quote count, number of forecast models, start dates for second-tier cities, capacity, and some city-level results.

This is not evidence that the edge disappeared. It is evidence that the reported result has no frozen source of truth.

### Why it matters

An interviewer or reviewer cannot reproduce the number being claimed. Any future data download silently changes the evaluated universe and can rewrite historical results.

### Required closure

- Freeze a named release dataset.
- Store a manifest containing every input path, row count, date range, source URL, download timestamp, and SHA-256 hash.
- Make report generation read only that manifest.
- Generate every number in `REPORT.md`, `NOTES.md`, and the tear sheet from one machine-readable result file.
- Fail the build if any displayed number differs from the generated result.
- Tag the code and result together in version control.

## 1.2 There is no externally verifiable chronology — P0 for “pre-registered,” P1 otherwise

The cross-venue code says that the April–September 2026 specification was fixed before those data were downloaded. In the current workspace there is no Git history or external timestamp proving that chronology.

### Why it matters

“Pre-registered” is a factual claim about when the design was frozen, not merely a synonym for holdout. Without an immutable timestamp, a skeptical reader cannot distinguish a genuine holdout from a retrospective split.

### Required closure

- Produce a Git commit, OSF registration, signed archive, or other immutable timestamp predating the holdout acquisition; or
- Replace “pre-registered” everywhere with “post-specification holdout” and describe the chronology as self-reported.

## 1.3 “Walk-forward” is being used too close to “fully out of sample” — P1

The parameters are refit using prior data, which is correct. But the hypothesis, market universe, snapshot time, season, feature set, and threshold were chosen during research on overlapping history. Walk-forward estimation does not remove research-selection bias.

### Required closure

- Separate three concepts in every table: parameter fitting, specification selection, and final untouched evaluation.
- Call historical rolling predictions “walk-forward.”
- Reserve “out of sample” for observations not used in any strategy or threshold choice.
- Maintain a locked final protocol for the next untouched period.

## 1.4 Per-trade t-statistics overstate effective sample size — P1

Multiple bracket trades share one city-day outcome. Trades across nearby cities share weather systems. Cross-venue positions from the same event share the same settlement. These P&Ls are not independent Bernoulli observations.

For the cross-venue six-month holdout, clustering by event or day produces a t-statistic of approximately 2.3 rather than treating all 1,306 positions as independent. The difference is currently small, but the clustered statistic is the defensible one.

### Required closure

- Remove per-trade t-statistics from headline tables.
- Report event-clustered and calendar-day-clustered standard errors.
- Add a moving-block bootstrap over dates to preserve serial and regional dependence.
- Report confidence intervals for per-contract P&L and capital-normalized daily returns.
- Predefine which statistic is primary.

## 1.5 The reported Sharpe ratios are not portfolio Sharpe ratios — P1

The code sums one-contract P&L by day and annualizes with `sqrt(365)`. Gross exposure varies with the number and price of signals. The statistic is therefore a one-lot signal-book Sharpe, not a return-on-capital Sharpe. Winter-only Sharpe is also annualized from active winter days without inserting the inactive part of the year.

### Required closure

- Define capital, collateral, position limits, and daily equity.
- Calculate returns on beginning-of-day capital or peak collateral usage.
- Insert zero-return inactive days for seasonal strategies.
- Report one-lot P&L Sharpe only as a diagnostic and label it explicitly.
- Use a risk-sized portfolio simulation for the headline Sharpe.

## 1.6 Threshold and model-selection uncertainty is not propagated — P1

The 5-cent threshold, snapshot window, season rule, model inputs, city universe, and several transformations were selected after comparing variants. The final standard errors condition on those choices as though they were fixed in advance.

### Required closure

- Lock one final protocol.
- Evaluate it on a new untouched period.
- Alternatively, use nested time-series cross-validation in which all specification choices are made inside each training window.
- Report a specification curve showing all reasonable thresholds and costs, not only the selected point.

## 1.7 Data-universe completeness is not proven — P1

The pipelines assemble markets from API listings and local files, then silently deduplicate. There is no audit proving that delisted, voided, malformed, missing-history, or failed-download markets are represented and handled consistently.

### Required closure

- Reconcile listed market counts against downloaded, parsed, joined, scored, and traded counts.
- Produce explicit exclusion reason codes.
- Treat voids, cancellations, missing outcomes, schema changes, and incomplete strips separately.
- Fail on unexplained count differences rather than silently dropping rows.

## 1.8 The project cannot currently be independently reproduced — P1

The workspace is approximately 2.2 GB and contains no public-data bootstrap, dependency lockfile, portable configuration, or single entry point.

### Required closure

- Add `pyproject.toml` or an equivalent locked environment.
- Replace absolute paths with configuration and paths relative to the project root.
- Add a `make reproduce` or equivalent command.
- Publish a small frozen evaluation sample or a scripted public-data rebuild.
- Record Python and package versions in generated outputs.

---

# 2. Weather-forecast strategy

## 2.1 Winter forecast timestamps are not proven safe — P0

The headline winter result uses Open-Meteo `previous_day1` archive fields and assumes they represent forecasts publicly available by the D-1 22Z trade time. The strict single-run archive covers only April–September 2026, when the strategy is much weaker. It therefore cannot validate the winter period that drives the headline.

Earlier snapshots were already shown to be contaminated by forecasts that were not yet public. That establishes that timing semantics are capable of manufacturing a false edge in this exact pipeline.

### Why venue replication does not close this problem

The Polymarket transfer uses the same forecast archive. If the forecast contains information published after the simulated trade, it will appear predictive on both venues. A second exchange is an independent market sample, but not an independent validation of forecast availability.

### Required closure

- Rebuild at least one complete winter using run-level forecasts with explicit initialization and publication timestamps.
- For every simulated signal, save the exact model run identifier and prove `publication_time <= decision_time`.
- Prefer the NOAA/NBM/HRRR/GFS source archives or another immutable run archive.
- Re-run the locked strategy without using `previous_day1` convenience fields.
- Do not headline the weather strategy as deployable until this passes.

## 2.2 Backtest and live features are not identical — P0 for deployment, P1 for research

The backtest uses Open-Meteo previous-run archive fields. `live_signal.py` queries the current forecast endpoint and takes daily maxima from whatever model runs the endpoint returns at runtime. It does not record or verify model initialization times.

The live model list contains 18 models, while historical fits use different subsets according to coverage—one fallback model in some early station-years, 12 in 2025, and 13 in 2026. The published description simply says “13-model blend.”

### Required closure

- Define the exact eligible model set by date and station.
- Save run initialization, availability, and retrieval times with every live prediction.
- Run the live feature builder against archived run data and prove byte-for-byte or tolerance-level parity with the backtest feature builder.
- Reject a signal if the required run identity or freshness cannot be established.

## 2.3 The forecast blend is rank-deficient — P1

Current reruns emit repeated `SingularMatrixWarning` messages from the per-station OLS blend. The weather models are highly collinear, so fitted weights are not uniquely identified.

Predictions can remain stable despite unstable coefficients, but that stability has not been measured and the pipeline does not fail on the warning.

### Required closure

- Replace OLS with ridge regression or a constrained ensemble.
- Select the regularization strength within the training window only.
- Report coefficient stability, prediction stability, and condition numbers by station-year.
- Convert numerical warnings into explicit test failures.
- Re-run all strategy results using the stabilized blend.

## 2.4 The calibration model described in prose differs from the code — P1

The strategy specification says the logistic combination is fit on approximately the last 12 months. The implementation trains on all prior months once the warm-up threshold is met. The paper signal then loads a fixed coefficient file fitted on settled 2025–26 events.

### Required closure

- Choose expanding or fixed-window estimation and document only that version.
- Generate `comb_coefs.json` through the same locked pipeline.
- Store coefficient provenance, training cutoff, training row count, and feature schema in the file.
- Prevent the live script from loading coefficients whose cutoff is later than the intended deployment cutoff.

## 2.5 Quote timestamps and decision timestamps need explicit verification — P1

Snapshot hour `h` is derived from candle end timestamps relative to market open. The result is described as a D-1 22Z executable quote. There is no unit-tested proof for every series, DST regime, and API candle convention that `h=8` always represents the intended decision instant.

### Required closure

- Build fixtures covering standard time, daylight time, every series open time, and candle boundary semantics.
- Print UTC decision timestamps directly in the scored-trade table.
- Verify a sample against raw API responses.
- Reject candles whose quote timestamp is after the decision time.

## 2.6 A candle-close ask or bid is not a demonstrated fill — P1

The baseline assumes the complete desired quantity can trade at the hourly candle’s closing ask or bid. A closing quote establishes neither displayed depth nor that the quote persisted long enough to execute.

The next-hour and worse-price tests are valuable, but they still do not model queue position, partial fills, order rejection, price impact, or size at the touch.

### Required closure

- Collect live order-book snapshots and actual paper-order acknowledgements at the decision time.
- Measure depth at each price, quote lifetime, and fill probability.
- Simulate partial fills and cancel/replace behavior.
- Separate signal alpha from execution alpha.
- Report results at several fixed contract sizes before reporting scalable capacity.

## 2.7 The capacity calculation uses future volume — P1

`weather_final.py` sizes each signal using volume that trades over the next six hours. That volume is unknown when the initial signal is generated. This does not contaminate the one-contract signal P&L, but it makes the capacity-sized backtest infeasible as written.

### Required closure

- If the order is submitted once, size only from information known at submission, such as trailing volume and current depth.
- If orders are worked over six hours, simulate the sequential policy hour by hour and let size respond only to volume already observed.
- Deduplicate positions across the 22Z, 00Z, and 02Z checks exactly as the live specification requires.
- Recompute capacity from historical or captured depth rather than future aggregate volume.

## 2.8 The current season story has drifted — P1

The saved report says summer is approximately +1 cent and should be skipped. The current rerun produces +2.6 cents over June–September, but with large month and city dispersion. The original season conclusion may still be directionally right, but the artifact no longer matches the evaluated universe.

### Required closure

- Regenerate season results from the frozen release.
- Test a season interaction within a locked model rather than selecting months from realized P&L alone.
- Report calendar-month confidence intervals and year-by-year season results.
- Do not encode a skip-summer trading rule until tested on untouched data.

## 2.9 Cross-city and within-event risk are understated — P1

The strategy may hold multiple mutually exclusive brackets in one event and positions in cities affected by the same regional weather system. Counting roughly seven or twelve signals per day as near-independent is too optimistic without an empirical covariance analysis.

### Required closure

- Construct the actual daily portfolio with net YES/NO exposure by event.
- Report within-event, within-region, and cross-city P&L correlations.
- Stress common forecast busts and station-report anomalies.
- Limit risk by event and weather region rather than by ticker alone.

## 2.10 The probability model assumes a stationary empirical residual distribution — P2

Bracket probabilities are created by adding historical training residuals to the point forecast. Forecast error variance can change by season, lead time, station, model availability, and weather regime.

### Required closure

- Test residual calibration by station, month, forecast temperature, and model-set regime.
- Compare empirical residuals with season-conditioned and heteroskedastic alternatives.
- Verify probability calibration on untouched events rather than only aggregate Brier score.

## 2.11 Operational safety is minimal — P2

The paper script has no robust API status checks, schema validation, retry/backoff policy, stale-data alarm, idempotency, or atomic output. Broad exception handling can silently skip a city. An empty or partial result can look like “no signals.”

### Required closure

- Validate HTTP status, schema, market count, quote freshness, model count, and forecast run time.
- Distinguish “no signal” from “data unavailable.”
- Add retries with bounded backoff and structured logs.
- Write outputs atomically with a run identifier.
- Add circuit breakers for missing brackets, probability sums, stale quotes, and abnormal forecast dispersion.

---

# 3. Polymarket frozen-parameter weather transfer

## 3.1 It shares the weather-timing risk — P0

The transfer is independent by venue but not by forecast input. It cannot be used as proof that the weather forecast was public at the simulated trade time.

### Required closure

Re-run the transfer after the winter run-level forecast rebuild. Until then, present it as a venue transfer conditional on the archive timing assumption.

## 3.2 The historical Polymarket execution price is not a recorded executable ask/bid — P1

The price-history value is described as “midpoint-ish,” and execution is represented by adding an assumed half-spread. Historical spread and depth are not observed for each signal.

### Required closure

- Establish precisely what the API `p` field represents for the historical period.
- Capture or obtain contemporaneous bid/ask and depth.
- Re-run with observed side-specific execution prices.
- Report sensitivity by city, price level, time of day, and required size.

## 3.3 The “no refitting” claim needs provenance — P1

The strategy reads `comb_coefs.json`, but that file does not carry an immutable training-data manifest and cutoff. The code alone does not prove which observations were used to create it.

### Required closure

- Embed training cutoff, source hashes, code commit, and coefficients in the result artifact.
- Add an automated assertion that no Polymarket rows can enter coefficient estimation.
- Rebuild the file through the reproducible release pipeline.

## 3.4 Venue-level replication is not the same as independent scientific replication — P2

The hypothesis, forecast source, temperature stations, dates, and many weather regimes overlap the Kalshi analysis. The result is valuable external-market validation, but “independent venue transfer” is more precise than “independent replication.”

---

# 4. Cross-venue Kalshi-to-Polymarket strategy

## 4.1 The contracts are not economically identical — P0 for the strongest price-discovery interpretation

Even the same-station, same-bracket pairs settle from different sources. The project finds that approximately 16% of matched brackets and 34% of matched events have at least one settlement disagreement. Other city pairs use different weather stations.

That is sufficient to show this is not risk-free arbitrage, but it also weakens the claim that both prices are noisy observations of one common efficient price. Part of the persistent gap may be legitimate settlement-basis risk.

### Required closure

- Model each venue’s settlement probability separately using its own oracle.
- Estimate the historical oracle-basis probability before each event using prior data only.
- Decompose the price gap into expected settlement basis and residual venue lag.
- Restrict the strongest price-discovery claims to economically identical settlement definitions, if such a subset exists.
- Otherwise describe the result as cross-venue predictive information, not pure price discovery.

## 4.2 Asynchronous and stale prices can create artificial lead-lag — P0/P1

The panel uses the last Polymarket price within a three-hour freshness window and hourly Kalshi candles. Quotes or trades from different instants are then compared on the same hourly grid. A faster-updating or more frequently traded venue can appear to lead simply because the other observation is stale.

### Required closure

- Record price age for both venues at every grid point.
- Re-run using tight synchronization windows such as 5, 15, and 30 minutes.
- Match like with like: executable quote to executable quote, or trade to trade.
- Include quote age and update intensity as controls.
- Show that the lead remains when both venues updated recently.

## 4.3 The Hasbrouck information-share calculation needs methodological validation — P0 for the 0.74–0.81 headline

The current code manually derives information shares from two error-correction regressions and their residual covariance. Standard Hasbrouck information share is normally obtained from a properly specified cointegrated VAR/VECM and its common-trend innovation representation.

The current data are short, bounded probability panels that reset for each contract, and the “identical” contracts can settle differently. Pooling these observations may violate the common-price and cointegration assumptions.

### Required closure

- Reimplement the calculation with a standard VECM library or independently reviewed derivation.
- Test cointegration and lag order.
- Explain how short event panels are pooled and how contract fixed effects/resets are handled.
- Bootstrap by contract/event rather than by row.
- Compare Hasbrouck, Gonzalo–Granger, and simpler forecast-error metrics.
- Remove the 0.74–0.81 headline if it is not stable across defensible specifications.

## 4.4 The fitted normal-strip transformation may be generating part of the signal — P1

Kalshi’s six-bracket strip is normalized and fit to a normal distribution, then mapped into Polymarket brackets. This imposes smoothing, tail behavior, and unimodality. The parameter-free same-contract test is weak outside Miami and degrades substantially in the holdout, so much of the broader result depends on the fitted transformation and calibration.

### Required closure

- Compare normal, logistic, spline, maximum-entropy, and nonparametric strip mappings.
- Use only transformations selected inside the training window.
- Report performance on exact-bin matches separately from station-converted matches.
- Attribute the result to the transformation only if it survives model-class sensitivity.

## 4.5 The holdout protocol is fixed, but the parameters continue learning — P1

During April–September 2026, the monthly model refits on all prior months, including earlier months from the holdout period. This is a valid live-like walk-forward test, but it is not a frozen-parameter test.

### Required closure

- Call it a fixed-specification walk-forward holdout.
- Also report a genuinely frozen March-2026 coefficient model across the entire holdout.
- Compare degradation between frozen and adaptive variants.

## 4.6 The holdout covers no new winter — P1

The holdout is almost entirely shoulder and summer data. It does not test a second winter or the regime in which the forecast-related strategy is strongest.

### Required closure

- Keep the protocol locked through the next full winter.
- Report the six-month result as the evidence actually observed, without generalizing to all seasons.

## 4.7 Results are concentrated and heterogeneous by city — P1

New York and Miami contribute much of the sample. Atlanta and Dallas are flat or negative. On exact same-contract comparisons, the parameter-free version works primarily in Miami.

### Required closure

- Report leave-one-city-out performance.
- Report contribution to P&L and t-statistic by city.
- Predefine whether negative cities remain in the strategy.
- Validate any city exclusion on a later untouched sample.

## 4.8 Historical Polymarket prices are not historical executable books — P1

The strategy executes at raw historical price plus a fixed 2-cent half-spread. It is not established that the price is a midpoint, that the chosen side was offered there, or that sufficient size existed.

The next-hour test controls some adverse movement but does not prove fillability.

### Required closure

- Obtain historical books or collect forward books.
- Use side-specific best bid/ask and depth.
- Model partial fills, price impact, and order latency.
- Stress the holdout—not only the full sample—at larger spreads.

## 4.9 Capacity uses future lifetime volume — P1

Position size is calculated from total bracket volume over the market’s full life. Much of that volume arrives after the signal and cannot establish available liquidity at entry. Dividing notional by a low contract price can also imply very large contract counts without depth support.

### Required closure

- Remove the current annual capacity estimate from headline materials.
- Base capacity on contemporaneous depth or volume observed before entry.
- Cap contracts, not just dollars, and simulate walking the book.
- Report capacity separately by city and price level.

## 4.10 Repeated outcomes in the calibration data require clustered estimation — P1

The logistic calibration contains repeated hourly rows for the same bracket and outcome. The information regressions use event-clustered covariance, but the monthly prediction model is fit as ordinary Logit and effectively weights events according to their number of available hourly rows.

### Required closure

- Give each event or bracket equal total training weight.
- Compare one-snapshot-per-event, GEE, hierarchical, and clustered alternatives.
- Verify that unequal quote-history density is not driving the fitted coefficients.

## 4.11 Survivor and invalid-market handling is not explicit — P1

The Polymarket pipeline starts from discovered closed events with usable price history and valid final outcome prices. It is not clear how missing, resolved-by-clarification, invalid, or delisted markets affect the universe.

### Required closure

- Produce a complete event-reconciliation table from discovery through inclusion.
- Include invalid/void outcomes explicitly.
- Test whether missing histories correlate with city, liquidity, outcome, or price gap.

## 4.12 The mechanism claim is stronger than the observed convergence P&L — P2

Exiting after 3 or 6 hours loses after another spread; only by 12 hours is convergence P&L slightly positive. Most backtest profit is realized at settlement rather than captured from short-horizon price convergence.

### Why it matters

The signal can still be predictive, but “Polymarket catches up to Kalshi” is not the same economic claim as “Kalshi helps predict Polymarket settlement.”

### Required closure

- Separate settlement prediction from convergence trading.
- Measure markout before spread at several horizons.
- Avoid describing the settlement-held strategy as monetizing rapid convergence unless executable exit P&L supports it.

---

# 5. Code and operational engineering problems

## 5.1 Hard-coded paths and script composition — P1

Many scripts hard-code `/Users/abebraham/Desktop/resume/edge/data/`. Several scripts execute fragments of other scripts using `exec(...split(...))`. This makes behavior dependent on file text layout and global mutable state.

### Required closure

- Convert shared logic into importable modules with typed functions.
- Use a project-root configuration object.
- Eliminate `exec`-based composition.
- Add explicit command-line arguments and validated configuration files.

## 5.2 No automated tests — P1

The most dangerous bugs in this project involved dates, forecast availability, bracket boundaries, settlement truth, and units. These are exactly the areas that need regression tests.

### Required closure

Add tests for:

- ticker-to-target-date parsing across months and DST;
- bracket boundary construction for less/between/greater contracts;
- Fahrenheit/Celsius conversion;
- one and only one winning bracket per valid event;
- fee rounding;
- quote/candle timestamp semantics;
- forecast run availability;
- station mapping and settlement source;
- no training rows after the prediction cutoff;
- reproduction of a frozen set of expected trades.

## 5.3 No data contracts or schema validation — P1

API fields are accessed directly, sometimes with fallbacks, and malformed or absent data can be silently skipped.

### Required closure

- Define schemas for each raw and processed dataset.
- Validate types, units, allowed ranges, uniqueness, and null rates.
- Version schemas and stop on unexpected changes.

## 5.4 No deterministic report build — P1

The tear sheet embeds stale numbers and hand-written claims. Some result files were produced before later data downloads.

### Required closure

- Generate all tables and prose variables from one result bundle.
- Store the evaluation manifest in the HTML.
- Include build timestamp, code commit, data hash, and exact command.
- Add a test that rebuilds the release and compares expected checksums or tolerances.

## 5.5 Warnings are ignored or suppressed — P1

Rank-deficiency warnings appear in the weather model. Several cross-venue scripts globally suppress warnings. A production research pipeline must distinguish expected warnings from invalid numerical results.

### Required closure

- Remove blanket warning suppression.
- Promote convergence, singularity, invalid probability, and failed-optimization warnings to errors.
- Store model diagnostics for every refit.

## 5.6 The paper runner is not a monitored service — P2

The live weather runner prints a table and writes a CSV. It has no scheduler guarantees, health checks, alerting, state, deduplication, or recovery behavior.

### Required closure

- Record every scheduled run, input version, success/failure, and emitted signal.
- Make repeated runs idempotent.
- Add stale-data and missing-city alerts.
- Persist paper positions and settlement P&L.
- Prove that no duplicate position is generated across repeated snapshots.

---

# 6. Claims that should not be made yet

Do not currently claim:

- “foolproof”;
- “proven deployable edge”;
- “live trading strategy”;
- realized or “earned” P&L;
- externally verified preregistration without a timestamped artifact;
- exact capacity in dollars per season;
- that venue replication eliminates forecast-timing leakage;
- that 0.74–0.81 Hasbrouck information share is definitive before a standard VECM validation;
- that every reported trade was fillable at the modeled price;
- that the strategy is fully out of sample merely because parameters were refit walk-forward;
- that the current report is reproducible from the current workspace.

Defensible current language:

- historical or simulated backtest;
- walk-forward parameter estimation;
- frozen-parameter transfer to another venue, conditional on forecast timing;
- fixed-specification walk-forward holdout for the cross-venue signal;
- quoted-price or assumed-spread execution model;
- promising evidence that requires run-level timing and live execution validation.

---

# 7. Minimum release gate

The project should not be called robust until all of the following pass:

1. One frozen dataset and code tag reproduce every public number.
2. Winter weather results survive true run-level publication timestamps.
3. Backtest and live feature generation are shown equivalent.
4. Rank-deficient weather fits are replaced and results remain positive.
5. Event/day-clustered and block-bootstrap intervals are reported everywhere.
6. Historical or forward-captured executable books support the cost assumptions.
7. Capacity is recomputed without future volume.
8. The cross-venue lead survives tight quote synchronization and quote-age controls.
9. The information-share calculation is independently validated with a standard methodology.
10. The next untouched winter and a meaningful forward paper-trading period remain positive.
11. Tests cover dates, brackets, units, fees, forecast availability, and leakage cutoffs.
12. The entire release builds from a pinned environment with no suppressed numerical warnings.

Until then, the correct verdict is: **strong and unusually thoughtful research evidence, but not yet a foolproof or production-proven trading system.**


---

# Closure status (added 2026-09-28 by the release build; see results/release/RESULTS.md)

| Item | Status | What was done |
|---|---|---|
| 1.1 report/tear sheet/pipeline disagree | **Closed** | `build_release.py` writes `results/release/{weather.json, xvenue.json, manifest.json (SHA-256, rows, ranges, versions), RESULTS.md}`; tear sheet rendered from a template with a value-consistency assertion; `frozen_trades_sha256.txt` + test. |
| 1.2 chronology of the holdout | **Relabelled** | "pre-registered" replaced by "post-specification holdout, chronology self-reported"; git initialised 2026-09-28 (external timestamp from now on). |
| 1.3 walk-forward vs OOS | **Closed** | RESULTS.md separates parameter fitting (walk-forward), specification (fixed on ≤2026-03) and holdout; "out of sample" no longer used for walk-forward rows. |
| 1.4 per-trade t-stats | **Closed** | Event- and day-clustered SEs, 95% CIs and moving-block bootstrap of daily P&L in every summary; per-trade t shown only as `t_iid` in JSON. |
| 1.5 Sharpe definition | **Closed** | Sharpe on capital = daily P&L / max daily gross cost over all calendar days (zeros when inactive); seasonal rows evaluated over all days; 1-lot Sharpe labelled diagnostic. |
| 1.6 selection uncertainty | **Partially** | Specification curves (threshold × cost) for both strategies; frozen-coefficient holdout; nested selection not implemented; locked protocol = RESULTS.md spec. |
| 1.7 universe completeness | **Closed** | `exclusions.parquet` with reason codes; reconciliation counts in both JSONs (discovered → priced → matched → scored). Pre-2023 Kalshi markets excluded (settlement/date convention). |
| 1.8 reproducibility | **Partially** | `requirements.txt` pins, `Makefile`, `EDGE_ROOT`-relative paths in the release code, versions in manifest; legacy scripts still have absolute paths; no public-data bootstrap. |
| 2.1 winter forecast timestamps | **Open (P0)** | Needs NOAA run-level archives; all winter figures labelled conditional. |
| 2.2 backtest/live parity | **Open** | `live_signal.py` unchanged; comb_coefs.json now carries provenance (cutoff, rows, hashes). |
| 2.3 rank-deficient blend | **Closed** | Condition numbers logged (36/50 fits > 1e8); ridge (alpha by in-window CV) reported alongside; results unchanged. |
| 2.4 prose vs code (12 months vs expanding) | **Closed** | Documented as expanding window; coefficient file regenerated by the release with training cutoff and input hashes. |
| 2.5 quote/decision timestamps | **Closed** | Clock-based D-1 22:00Z decision (100% of rows); `decision_utc` stored in the trade table; tests. |
| 2.6 fills | **Open / labelled** | Next-hour and worst-fill stresses kept; no depth data; all results labelled modelled execution. |
| 2.7 capacity uses future volume | **Closed** | Trailing-6h-volume sizing is primary; future-volume figure retained only as "INFEASIBLE" comparison. |
| 2.8 season drift | **Closed** | Season rows regenerated from the release; no skip-summer rule encoded. |
| 2.9 cross-city/within-event risk | **Partially** | Mean pairwise daily station P&L correlation and multi-position share reported; no regional stress. |
| 2.10 residual stationarity | **Open** | Not addressed. |
| 2.11 operational safety | **Open** | Not addressed. |
| 3.1–3.4 Polymarket transfer | **Relabelled** | Shares the timing caveat; "independent venue transfer" wording; not rebuilt in the release (pre-audit figure kept as historical). |
| 4.1 non-identical contracts | **Closed (claim narrowed)** | Disagreement measured (16% brackets, 34% events); model-free trade also scored on Kalshi's settlement; price-discovery claim withdrawn. |
| 4.2 asynchronous prices | **Closed** | Quote age recorded; 97% within 15 min; fresh-only information and trading results unchanged. |
| 4.3 Hasbrouck methodology | **Closed by withdrawal** | Library VECM per contract (median Kalshi share 0.40) and a simulation check that fails the pooled estimator; headline removed. |
| 4.4 strip mapping | **Closed** | Piecewise-linear CDF alternative reported (z 6.7; holdout +2.9c). |
| 4.5 frozen vs adaptive holdout | **Closed** | Both reported. |
| 4.6 no winter in holdout | **Labelled** | Stated in RESULTS.md and tear sheet. |
| 4.7 city heterogeneity | **Closed** | Leave-one-city-out and holdout-by-city tables. |
| 4.8 executable books | **Open / labelled** | Half-spread sensitivity 1–5c; no depth data. |
| 4.9 capacity | **Closed by withdrawal** | Dollar figure removed. |
| 4.10 repeated rows in calibration | **Closed** | Equal-weight GLM variant reported. |
| 4.11 survivor/invalid markets | **Closed** | Reconciliation counts from files → endDate → priced → matched → scored. |
| 4.12 mechanism language | **Closed** | Markouts before/after spread reported; described as settlement prediction. |
| 5.1 paths/exec | **Partially** | Release code importable and root-relative; legacy scripts unchanged. |
| 5.2 tests | **Closed** | 18 pytest tests incl. dates/DST, brackets, units, fees, decision time, leakage, frozen trades. |
| 5.3 schemas | **Open** | Not addressed. |
| 5.4 deterministic report | **Closed** | See 1.1. |
| 5.5 warnings | **Partially** | Rank deficiency now measured (condition numbers) instead of warned; cross-venue scripts still suppress warnings in the VECM loop. |
| 5.6 paper runner | **Open** | Not addressed. |
