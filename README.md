# House Price and Sales Analysis — King County, USA

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/harrisonokojiee/house-price-analysis/blob/main/house_price_analysis.ipynb)

Portfolio project: what drives house prices in King County (Seattle area), May 2014 – May 2015.

## Business questions
1. What features correlate most with price (living area, grade, location)?
2. Where is value concentrated geographically?
3. How did median prices trend over the year?
4. Can a simple model predict price from listing features?

## Dataset
- Kaggle: `harlfoxem/housesalesprediction` — House Sales in King County, USA (~21,613 rows x 21 cols).
- The notebook fetches it with `kagglehub.dataset_download(...)`. If no Kaggle auth is present (e.g. fresh Colab), it falls back to a synthetic same-schema sample so the notebook still runs end-to-end.

## Run in Colab
1. Open `house_price_analysis.ipynb` in Colab.
2. Run all cells. First cell installs deps: `pandas matplotlib scikit-learn kagglehub kaggle`.
3. For real data: add your Kaggle token via Colab Secrets (`KAGGLE_USERNAME` / `KAGGLE_KEY`) or upload `kaggle.json` when prompted. Otherwise the synthetic fallback runs automatically.

## Run locally
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```
Figures land in `figures/`.

## What is in here
- `house_price_analysis.ipynb` — narrative notebook (Colab-ready).
- `analysis.py` — same pipeline as a script for reproducibility.
- `figures/` — 11 real-run charts (distribution, drivers, trend, geo, fit, importance, scenario, renovation, FRED context).
- `requirements.txt`, `data/` (ignored raw CSVs go here).

## Key findings (real run: 21,613 → 21,421 sales after cleaning)
- Investor lens: renovate to flip in mid-grade (7–8, +22.5% $/sqft) and high-grade (9+, +50.5%) homes; low grades show −9.5% — flips don't pay there. Luxury and waterfront listings get manual appraisal with quantile bands.
- Model table (shuffled 80/20, RMSE in dollars): HistGradientBoosting $121,583 R² 0.896 (best) · Lasso $176,051 R² 0.783 · RandomForest $194,322 R² 0.735 · LR-log $214,713 R² 0.677 · LinearRegression $217,745 R² 0.668.
- Forward time splits hold up (train past → test future, RF R² 0.717 / 0.713 / 0.713), so the model generalizes to future listings.
- Quantile 80% bands cover 0.79 of test homes — quote the band, not the point.
- Market scenario (damped Holt-Winters, backtested vs naive on last 3 months): 6-month end $457,240 ±$18,544; months 7–12 are stretch. SCENARIO, not a forecast — no multi-year prediction from 12 points.
- Limitation: 2014–2015 Seattle only (median $450,000, $244.54/sqft); no interest-rate features; do not use for 2026 pricing.

## Report and screenshots
- Full write-up: `report/REPORT.md` (investor lens, KPI scorecard, findings, decisions, limits).
- 11 real-run screenshots in `report/figures/` (mirrored in `figures/`), plus FRED context from `data/external/seattle_hpi.csv` (public, committed).
- Extended charts (volume, residuals, error bands, learning curve, quantile bands, log-dist, interactive map) and `scenario_projection.csv` regenerate on Run All.

<!-- Portfolio blurb (hidden from render; copy-paste source for your website)
> King County House Prices — I analyzed 21,421 Seattle-area home sales (2014–2015) to find what drives price and how far the data can project. A HistGradientBoosting model (R² 0.90) holds up under forward time splits (R² ≈ 0.71); renovations pay in mid-grade (+22%) and high-grade (+51%) homes but not low-grade (−9%), and every appraisal ships with a quantile band (80% coverage 0.79). A damped Holt-Winters scenario projects the monthly median 6 months out — honestly labeled as scenario, not forecast, since 12 months cannot support multi-year claims. Median sale: $450,000. Built with pandas, scikit-learn, statsmodels, and matplotlib; reproducible in one Colab Run All via Kaggle. [notebook] [report] [hero charts]
-->

## Skills shown
pandas cleaning, EDA, feature engineering, matplotlib visualization, shuffled + time-forward evaluation (RMSE/R2), Holt-Winters scenarios with backtest, Kaggle ingest, Colab reproducibility, written reporting.

## License
MIT — see [LICENSE](LICENSE).

## Sister project
[Netflix Movies & Shows Explorer](https://github.com/harrisonokojiee/netflix-titles) — content-strategy EDA showing breadth beside this project's depth.
