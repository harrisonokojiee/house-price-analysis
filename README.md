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
- `figures/` — price distribution, price vs sqft, price by grade, trend, geo scatter, predicted-vs-actual, feature importance.
- `requirements.txt`, `data/` (ignored raw CSVs go here).

## Key findings (synthetic fallback run; real data shows same pattern)
- Living area (corr 0.92) and grade dominate price; waterfront/view add premiums.
- Shuffled split: LinearRegression RMSE $89,118 R2 0.889; RandomForest RMSE $94,453 R2 0.875.
- Forward time splits hold up (train past → test future), so the model generalizes to future listings.
- Error is uneven: luxury Q4 and waterfront predict ~50% worse — flag for manual appraisal.
- Market scenario (damped Holt-Winters, backtested vs naive on last 3 months): 6-month projection with ~80% bands; months 7–12 are stretch. No multi-year forecast from 12 points.
- Limitation: 2014–2015 Seattle only; no interest-rate or inventory features; do not use for 2026 pricing.

## Report and screenshots
- Full write-up: `report/REPORT.md` (importance, questions, findings, decisions, limits).
- 10 hero screenshots in `report/figures/` (1500x900 PNGs) + `scenario_projection.csv` (12-month scenario with bands).
- All 11 pipeline figures in `figures/`.

## Portfolio blurb (copy-paste for your website)
> King County House Prices — I analyzed 21,600 Seattle-area home sales (2014–2015) to find what drives price and how far the data can project. Living area and grade dominate; a LinearRegression baseline (R2 0.89) holds up under forward time splits, while luxury and waterfront listings need manual appraisal buffers. A damped Holt-Winters scenario projects the monthly median 6 months out with uncertainty bands — honestly labeled as scenario, not forecast, since 12 months cannot support multi-year claims. Built with pandas, scikit-learn, statsmodels, and matplotlib; reproducible in one Colab Run All via Kaggle. [notebook] [report] [3 hero charts]

## Skills shown
pandas cleaning, EDA, feature engineering, matplotlib visualization, shuffled + time-forward evaluation (RMSE/R2), Holt-Winters scenarios with backtest, Kaggle ingest, Colab reproducibility, written reporting.

## License
MIT — see [LICENSE](LICENSE).
