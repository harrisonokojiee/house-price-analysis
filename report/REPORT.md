# Housing Prices in King County — Report

Seattle area, May 2014 – May 2015. Dataset: Kaggle `harlfoxem/housesalesprediction`
(~21,613 rows x 21 cols; local runs use a same-schema fallback when no Kaggle auth).

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
  out-of-range prices, dedupe ids. Fallback run: 5,000 → 4,980 rows.
- Optional external context only: FRED/FHFA Seattle HPI (`data/external/seattle_hpi.csv`,
  columns date,value). Never a training feature. Skipped gracefully when absent.

## Methods
- EDA: distributions, price vs sqft, grade medians, monthly trend + volume, geo scatter,
  correlation ranking.
- Features: age_at_sale, renovated flag, price_per_sqft, sale_month.
- Property models: LinearRegression baseline vs RandomForest (100 trees), 80/20 shuffled
  split (RMSE/R2) plus 3 forward TimeSeriesSplits (train past → test future).
- Market scenario: monthly median series (12 periods asserted), backtest naive
  carry-forward vs damped Holt-Winters on last 3 months (MAE), then 12-month projection
  with ~80% bands. Labeled SCENARIO throughout.

## Findings (fallback run; real data shows the same pattern)
- Correlation with price: sqft_living 0.92, sqft_above 0.87, grade 0.15; everything else < 0.06.
- Shuffled split: LinearRegression RMSE $89,118 R2 0.889; RandomForest RMSE $94,453 R2 0.875.
- Forward splits hold up across all 3 windows (e.g. split-2 LR RMSE $90,023 R2 0.889),
  so the model generalizes to future listings.
- Error is uneven: median abs error Q2 $54,623 vs luxury Q4 $67,237; waterfront
  $96,249 vs non-waterfront $64,932; high grade 9+ $67,820. Luxury/sparse segments
  predict worse.
- Backtest (last 3 months): naive MAE $17,890 vs Holt-Winters MAE $19,124 — on 12 flat-ish
  fallback points the naive holds its own, which is itself a finding about small samples.
- Scenario: damped Holt-Winters, residual std $14,542; 6-month scenario end ≈ $903,857
  with ±$18,600 80% band on fallback data. Rerun on real Kaggle data before publishing numbers.

## Decisions and recommendations
1. Price mid-grade 3–4 bedroom homes from the model; competition is thickest there and error lowest.
2. Flag luxury (Q4), waterfront, and grade 9+ for manual appraisal — add a premium buffer; model error runs ~50% higher.
3. Watch volume with median: a rising median on thinning volume is weaker signal than on broad volume.
4. Quote only the 6-month scenario with bands to stakeholders; call months 7–12 stretch. Never quote a multi-year price from these 12 points.

## What this does not claim
- No 2026–2030 price prediction. Twelve months of 2014–2015 microdata cannot support that.
- No causal claim (renovation ≠ automatic uplift) and no interest-rate/inventory adjustment.
- Fallback numbers are pipeline demos; the Colab rerun with Kaggle auth is the publishable run.

## Screenshots (report/figures/)
1. `price_dist.png` — right-skewed prices, luxury tail.
2. `price_vs_sqft.png` — living area is the dominant driver.
3. `price_by_grade.png` — grade steps carry large median jumps.
4. `price_trend.png` — monthly median path.
5. `sales_volume.png` — seasonal transaction rhythm.
6. `geo.png` — value concentration by location.
7. `pred_vs_actual.png` — model fit with luxury-tail spread.
8. `feature_importance.png` — sqft_living and grade on top.
9. `error_by_band.png` — luxury predicts worse.
10. `scenario.png` — history + holdout + 6-mo scenario + 7–12-mo stretch with bands.

## Reproduce
- Colab: open `house_price_analysis.ipynb` → Run All (works without auth via fallback; add Kaggle token for real data).
- Local: `source .venv/bin/activate && pip install -r requirements.txt && python analysis.py`
- Outputs: `figures/` (11 PNGs), `scenario_projection.csv` (12 rows), `report/figures/` (10 heroes).

## Next steps
Zipcode-level medians, FRED HPI join for long-run context chart, and a holdout test on a second year of data if sourced.
