# Housing Prices in King County — Report

Seattle area, May 2014 – May 2015. Dataset: Kaggle `harlfoxem/housesalesprediction`
(21,613 rows x 21 cols; the notebook keeps a synthetic same-schema fallback so
Run All still succeeds without Kaggle auth, but every number below comes from the
real-data run).

**Investor lens (fix-and-flip):** which grades and neighborhoods pay a renovation
premium, and how wide is the appraisal risk on any single listing?

**KPI scorecard (real run, 21,421 sales after cleaning):** median price $450,000
(median $244.54/sqft) | best model HistGradientBoosting RMSE $121,583, R² 0.896 |
forward time splits RF R² ≈ 0.71 | quantile 80% band coverage 0.79 |
renovation premium by grade: mid +22.5%, high +50.5%, low −9.5%.

## Why this project matters
A house is the largest purchase most households make, and small pricing errors cost
tens of thousands of dollars. This project shows whether listing features can price
a home fairly, where value concentrates, and what can (and cannot) be said about
where the market heads next. Employers care about exactly this: clean data, honest
evaluation, decisions under uncertainty.

## Questions it answers
1. What drives price? (living area, grade, location — quantified)
2. Where is value concentrated? (geo + grade pockets)
3. How did the market move month to month? (trend + volume)
4. Can we predict a listing's price? (baseline vs RandomForest, shuffled + forward splits)
5. What is a defensible near-term scenario? (6-month scenario with bands; 7-12 months stretch)

## Data
- King County sales May 2014–May 2015: price, bedrooms, bathrooms, sqft_living/lot,
  floors, waterfront, view, condition, grade, sqft_above/basement, yr_built/renovated,
  zipcode, lat/long, sqft_living15/lot15, date.
- Cleaning: drop unparseable dates, median-fill bathrooms, remove 0-bedroom and
  out-of-range prices, dedupe on id+date with repeat sales flagged. Real run:
  21,613 → 21,421 rows.
- Optional external context only: FRED Case-Shiller Seattle HPI
  (`data/external/seattle_hpi.csv`). Never a training feature. Skipped gracefully
  when absent (as in the Colab run).

## Methods
- EDA: distributions, price vs sqft, grade medians, monthly trend + volume, geo scatter,
  correlation ranking.
- Features: age_at_sale, renovated flag, price_per_sqft, sale_month.
- Property models: LinearRegression baseline vs RandomForest (100 trees), 80/20 shuffled
  split (RMSE/R2) plus 3 forward TimeSeriesSplits (train past → test future).
- Market scenario: monthly median series (12 periods asserted), backtest naive
  carry-forward vs damped Holt-Winters on last 3 months (MAE), then 12-month projection
  with ~80% bands. Labeled SCENARIO throughout.

## Findings (real run, 21,421 sales)
- Price: median $450,000 (25% $322,500 / 75% $645,000; min $75k, max $7.7M),
  mean $540,605; median living area 1,920 sqft at $244.54/sqft; median grade 7.
- Living area, grade, and location dominate price — confirmed by permutation
  importance (HistGB: sqft_living 0.21, zip_median_oof 0.19, grade 0.18,
  long 0.054, lat 0.050), not just correlation.
- Shuffled 80/20: LinearRegression RMSE $217,745 R² 0.668; RandomForest RMSE
  $194,322 R² 0.735. Real markets are noisier than the linear fallback world.
- Forward splits hold up (train past → test future): LR R² 0.660 / 0.663 / 0.620,
  RF R² 0.717 / 0.713 / 0.713 — the model generalizes to future listings.
- Correlation ranking (real data): sqft_living 0.70, grade 0.67, sqft_above 0.61,
  bathrooms 0.53, view 0.40 — weaker than any linear world, which is why trees win.
- Error is uneven: luxury-quartile median error ($153,689) runs ~2.5× the middle
  band ($60,992); waterfront ($434,126) ~5× non-waterfront ($83,466). Manual
  appraisal at the top end.
- Backtest (last 3 months): naive MAE $36,083 vs Holt-Winters MAE $41,792 — the
  naive holds its own on 12 points, which is itself a finding about small samples.
- Scenario: damped Holt-Winters, 6-month end $457,240 with ±$18,544 ~80% band;
  months 7–12 are stretch. SCENARIO, not a forecast.

## Decisions and recommendations
1. Price mid-grade 3–4 bedroom homes from the model; competition is thickest there.
2. Flag luxury and waterfront listings for manual appraisal — quote the quantile
   band (10th–90th), not the point prediction.
3. Renovate to flip in mid (7–8) and high (9+) grades, where the $/sqft premium is
   +22.5% and +50.5%; low grades show −9.5% (renovations don't pay there).
4. Watch volume with median: a rising median on thinning volume is weaker signal than on broad volume.
5. Quote only the 6-month scenario with bands to stakeholders; call months 7–12 stretch. Never quote a multi-year price from these 12 points.

## What this does not claim
- No multi-year price prediction. Twelve months of 2014–2015 microdata cannot support that.
- No causal claim (renovation ≠ automatic uplift) and no interest-rate/inventory adjustment.
- 2014–2015 Seattle only; do not use for 2026 pricing.

## Phase 3 findings (real run)
- **Log-target:** LR on log(price) RMSE $214,713 R² 0.677 vs $217,745 raw — a small
  win on skewed real prices, as theory predicts (see `price_dist_log.png`).
- **Lasso RMSE $176,051 R² 0.783** — regularization beats plain linear by a clear
  margin on real data.
- **Boosting wins:** HistGradientBoosting RMSE $121,583 R² 0.896, far ahead of RF
  ($194,322). Learning curve (`learning_curve.png`) shows the gap path.
- **Permutation importance** (`perm_importance.png`): HistGB top drivers are
  sqft_living (0.21), zip_median_oof (0.19), grade (0.18), long (0.054), lat (0.050)
  — location matters on real data, with no impurity bias toward high-cardinality features.
- **Quantile bands cover 0.79 of test homes** (nominal 0.80) — quote the band, not
  the point (`quantile_band.png`).
- **Location:** neighborhood medians and coordinates carry real signal (unlike the
  random fallback zips). `price_vs_zip_median` stays analysis-only (contains the
  target) and is never a feature.

## Renovation premium + long-run context (real run)
- Renovated vs original median $/sqft within grade bands: low ≤6: 241 vs 218
  (−9.5%); mid 7–8: 234 vs 286 (+22.5%); high 9+: 261 vs 393 (+50.5%).
  Flips pay in mid and high grades, not in low grades (`renovation_premium.png`).
- FRED Case-Shiller Seattle index (monthly, 1990–2026) with the study window
  shaded (`fred_context.png`, produced locally from public FRED data — the Colab
  run skipped it): the 2014–15 sales sit on a long climb, which is why
  the report refuses multi-year forecasts from 12 points. Context only, never input.

## Screenshots (report/figures/ — all from the real run)
1. `price_dist.png` — right-skewed prices, luxury tail.
2. `price_vs_sqft.png` — living area vs price, 21k sales.
3. `price_by_grade.png` — grade steps carry large median jumps (grade 13 ≈ $3M).
4. `price_trend.png` — monthly median path, May 2014 – May 2015.
5. `geo.png` — value concentration by location.
6. `pred_vs_actual.png` — model fit with luxury-tail spread.
7. `feature_importance.png` — grade and sqft_living on top (impurity).
8. `perm_importance.png` — permutation-based top drivers with test-set rigor.
9. `scenario.png` — history + holdout + 12-mo scenario with SCENARIO label.
10. `renovation_premium.png` — renovated vs original $/sqft by grade.
11. `fred_context.png` — 35-year Seattle index with study window shaded.
Removed from this commit (fallback-only, no real source): volume/residual/error-band/
learning-curve/quantile-band/log-dist charts, the interactive map HTML, and
`scenario_projection.csv` — all regenerate on the next real Run All.

## Reproduce
- Colab: open `house_price_analysis.ipynb` → Run All with Kaggle auth for the real
  numbers above (works without auth via the synthetic fallback, but then figures
  and metrics will differ).
- Local: `source .venv/bin/activate && pip install -r requirements.txt && python analysis.py`
- Outputs: `figures/` + `report/figures/` (11 real-run PNGs each); extended charts
  (volume, residuals, error bands, learning curve, quantile bands, log-dist, map)
  and `scenario_projection.csv` regenerate on Run All.

## Next steps
Repeat-sale appreciation (353 repeat-sale rows are now flagged, not dropped), a holdout
test on a second year of data if sourced, and zipcode-level investment medians.
