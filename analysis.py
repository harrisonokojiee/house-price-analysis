"""House Price and Sales Analysis — King County, USA.
Runs locally (.venv) and in Colab. Fetches Kaggle data when credentials
exist, otherwise uses a synthetic fallback with the same schema.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 11,
                     "axes.titlesize": 13, "axes.labelsize": 11})
SOURCE_TAG = "Source: Kaggle harlfoxem/housesalesprediction (May 2014–May 2015) or same-schema fallback"


def _shot(name, title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.figtext(0.01, 0.01, SOURCE_TAG, fontsize=8, color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(FIGDIR, name), dpi=150)
    plt.close()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
DATADIR = os.path.join(HERE, "data")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)

EXPECTED_COLS = ["id", "date", "price", "bedrooms", "bathrooms", "sqft_living",
                 "sqft_lot", "floors", "waterfront", "view", "condition", "grade",
                 "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
                 "zipcode", "lat", "long", "sqft_living15", "sqft_lot15"]


def load_data(n_fallback=5000, seed=42):
    """Try Kaggle (kagglehub), else build synthetic fallback."""
    # 1. Local CSV if user already downloaded it
    for f in os.listdir(DATADIR) if os.path.isdir(DATADIR) else []:
        if f.endswith(".csv"):
            p = os.path.join(DATADIR, f)
            df = pd.read_csv(p)
            print(f"Loaded local CSV: {p} {df.shape}")
            return df, "local_csv"
    # 2. kagglehub
    try:
        import kagglehub
        path = kagglehub.dataset_download("harlfoxem/housesalesprediction")
        csvs = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
        if csvs:
            df = pd.read_csv(csvs[0])
            print(f"Loaded Kaggle: {csvs[0]} {df.shape}")
            return df, "kagglehub"
    except Exception as e:
        print(f"kagglehub fetch skipped ({type(e).__name__}: {e})")
    # 3. Synthetic fallback with King County-like distributions
    rng = np.random.default_rng(seed)
    n = n_fallback
    sqft_living = rng.normal(2080, 920, n).clip(370, 13540).astype(int)
    grade = rng.normal(7.6, 1.3, n).clip(1, 13).astype(int)
    bedrooms = rng.choice([1, 2, 3, 3, 4, 4, 5, 6], size=n)
    bathrooms = (rng.normal(2.1, 0.8, n).clip(0.5, 8).round(1))
    price = (50000 + sqft_living * 280 + grade * 35000
             + bedrooms * 8000 + rng.normal(0, 90000, n)).clip(75000, 7_700_000)
    df = pd.DataFrame({
        "id": np.arange(10_000_000, 10_000_000 + n),
        "date": pd.to_datetime(rng.choice(pd.date_range("2014-05-01", "2015-05-31"), n)),
        "price": price.astype(int),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": rng.normal(15100, 12000, n).clip(520, 800000).astype(int),
        "floors": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], size=n),
        "waterfront": rng.choice([0, 1], size=n, p=[0.993, 0.007]),
        "view": rng.choice([0, 0, 0, 1, 2, 3, 4], size=n),
        "condition": rng.choice([1, 2, 3, 3, 4, 5], size=n),
        "grade": grade,
        "sqft_above": (sqft_living * rng.uniform(0.6, 1.0, n)).astype(int),
        "sqft_basement": np.maximum(sqft_living - (sqft_living * rng.uniform(0.6, 1.0, n)).astype(int), 0),
        "yr_built": rng.integers(1900, 2015, n),
        "yr_renovated": np.where(rng.random(n) < 0.08, rng.integers(1990, 2015, n), 0),
        "zipcode": rng.choice([98001, 98002, 98003, 98028, 98103, 98115, 98125], size=n),
        "lat": rng.normal(47.56, 0.14, n).clip(47.1, 47.8),
        "long": rng.normal(-122.21, 0.14, n).clip(-122.5, -121.3),
        "sqft_living15": (sqft_living * rng.uniform(0.85, 1.15, n)).astype(int),
        "sqft_lot15": rng.normal(12700, 9000, n).clip(600, 500000).astype(int),
    })
    # Inject a few dirty rows so cleaning step is real
    df.loc[rng.choice(n, 20, replace=False), "bedrooms"] = 0
    df.loc[rng.choice(n, 15, replace=False), "bathrooms"] = np.nan
    print(f"Using synthetic fallback {df.shape} (real Kaggle set is ~21613x21)")
    return df, "synthetic"


def clean(df):
    before = len(df)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["price", "date"])
    df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
    df = df[(df["bedrooms"] > 0) & (df["bedrooms"] <= 10)]
    df = df[(df["price"] >= 50000) & (df["price"] <= 8_000_000)]
    df = df.drop_duplicates(subset=["id"])
    after = len(df)
    print(f"Cleaning: {before} -> {after} rows")
    return df


def engineer(df):
    df = df.copy()
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df["age_at_sale"] = df["sale_year"] - df["yr_built"]
    df["renovated"] = (df["yr_renovated"] > 0).astype(int)
    df["price_per_sqft"] = df["price"] / df["sqft_living"].clip(lower=1)
    return df


def monthly_series(df):
    m = df.set_index("date").resample("ME").agg(median_price=("price", "median"),
                                                sales=("price", "size"))
    assert 12 <= len(m) <= 13, f"expected 12-13 monthly periods, got {len(m)}"
    return m


def plot_all(df):
    plt.figure()
    df["price"].hist(bins=60)
    _shot("price_dist.png", "Sale price distribution", "Price (USD)", "Count")

    plt.figure()
    plt.scatter(df["sqft_living"], df["price"], s=4, alpha=0.3)
    _shot("price_vs_sqft.png", "Price vs living area", "Sqft living", "Price (USD)")

    plt.figure()
    df.groupby("grade")["price"].median().plot(kind="bar")
    _shot("price_by_grade.png", "Median price by grade", "Grade", "Median price (USD)")

    m = monthly_series(df)
    plt.figure()
    m["median_price"].plot(marker="o")
    _shot("price_trend.png", "Median price trend May 2014 - May 2015",
          "Month", "Median price (USD)")

    plt.figure()
    m["sales"].plot(kind="bar")
    _shot("sales_volume.png", "Sales volume by month", "Month", "Transactions")

    plt.figure()
    plt.scatter(df["long"], df["lat"], s=3, alpha=0.3,
                c=df["price"].clip(upper=df["price"].quantile(0.95)))
    _shot("geo.png", "Sales location (color = price)", "Longitude", "Latitude")


FEATURES = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
            "waterfront", "view", "condition", "grade", "sqft_above",
            "sqft_basement", "age_at_sale", "renovated", "sale_month"]


def model(df):
    X = df[FEATURES].fillna(df[FEATURES].median())
    y = df["price"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    out = {}
    for name, m in [("LinearRegression", LinearRegression()),
                    ("RandomForest", RandomForestRegressor(n_estimators=100,
                                                           random_state=42, n_jobs=-1))]:
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        out[name] = {"rmse": float(np.sqrt(mean_squared_error(yte, p))),
                     "r2": float(r2_score(yte, p))}
        print(f"{name}: RMSE=${out[name]['rmse']:,.0f} R2={out[name]['r2']:.3f}")
    # Residual + predicted-vs-actual for best model
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    plt.figure()
    plt.scatter(yte, pred, s=5, alpha=0.3)
    _shot("pred_vs_actual.png", "RandomForest: predicted vs actual",
          "Actual price", "Predicted price")
    resid = yte.values - pred
    plt.figure()
    plt.scatter(pred, resid, s=5, alpha=0.3)
    plt.axhline(0, color="red", linewidth=1)
    _shot("residuals.png", "Residuals vs predicted (RandomForest)",
          "Predicted price", "Actual - predicted")
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Top features:\n" + imp.head(6).to_string())
    imp.head(8).plot(kind="barh")
    _shot("feature_importance.png", "Top price drivers (RandomForest)",
          "Importance", "Feature")
    return out, rf


def time_aware_eval(df):
    """Forward splits: train on earlier sales, test on later sales. No shuffling."""
    d = df.sort_values("date").reset_index(drop=True)
    X = d[FEATURES].fillna(d[FEATURES].median())
    y = d["price"]
    tss = TimeSeriesSplit(n_splits=3)
    rows = []
    for i, (tr, te) in enumerate(tss.split(X)):
        tr_dates = (d.loc[tr, "date"].min().date(), d.loc[tr, "date"].max().date())
        te_dates = (d.loc[te, "date"].min().date(), d.loc[te, "date"].max().date())
        assert d.loc[te, "date"].min() >= d.loc[tr, "date"].min(), "time leakage"
        for name, mk in [("LR", LinearRegression),
                         ("RF", lambda: RandomForestRegressor(n_estimators=100,
                                                              random_state=42, n_jobs=-1))]:
            m = mk()
            m.fit(X.iloc[tr], y.iloc[tr])
            p = m.predict(X.iloc[te])
            rows.append({"split": i, "model": name,
                         "rmse": float(np.sqrt(mean_squared_error(y.iloc[te], p))),
                         "r2": float(r2_score(y.iloc[te], p)),
                         "train": f"{tr_dates[0]}..{tr_dates[1]}",
                         "test": f"{te_dates[0]}..{te_dates[1]}"})
    fwd = pd.DataFrame(rows)
    print("Forward-split evaluation (train past -> test future):\n" + fwd.to_string(index=False))
    # Error by segment from a full-data RF, for decisions
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf.fit(Xtr, ytr)
    e = pd.DataFrame({"actual": yte, "pred": rf.predict(Xte)})
    e["abs_err"] = (e["actual"] - e["pred"]).abs()
    e["band"] = pd.qcut(e["actual"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 luxury"])
    seg = e.groupby("band", observed=True)["abs_err"].median().astype(int)
    print("Median abs error by price band:\n" + seg.to_string())
    g = d.loc[yte.index].copy()
    g["abs_err"] = e["abs_err"].values
    gseg = g.groupby(pd.cut(g["grade"], [0, 6, 8, 13],
                            labels=["low<=6", "mid 7-8", "high 9+"]),
                     observed=True)["abs_err"].median().astype(int)
    print("Median abs error by grade:\n" + gseg.to_string())
    wseg = g.groupby("waterfront")["abs_err"].median().astype(int)
    print("Median abs error waterfront 0/1:\n" + wseg.to_string())
    seg.plot(kind="bar")
    _shot("error_by_band.png", "Median prediction error by price band",
          "Price band", "Median abs error (USD)")
    return fwd


def backtest_and_scenario(df):
    """Backtest naive vs Holt-Winters on last 3 months, then 6/12-mo scenarios."""
    m = monthly_series(df)
    y = m["median_price"]
    train, holdout = y.iloc[:-3], y.iloc[-3:]
    naive = pd.Series([train.iloc[-1]] * 3, index=holdout.index)
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hw_fit = ExponentialSmoothing(train, trend="add", damped_trend=True,
                                      seasonal=None).fit(optimized=True)
        hw_pred = hw_fit.forecast(3)
        hw_mae = float((holdout - hw_pred).abs().mean())
        hw_ok = True
    except Exception as e:
        print(f"Holt-Winters backtest failed ({e}); using naive trend fallback")
        hw_pred = naive
        hw_mae = float((holdout - naive).abs().mean())
        hw_ok = False
    naive_mae = float((holdout - naive).abs().mean())
    print(f"Backtest MAE last 3 months: naive=${naive_mae:,.0f} "
          f"holtwinters=${hw_mae:,.0f} (hw_ok={hw_ok})")
    # Scenario fit on full series
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        full = ExponentialSmoothing(y, trend="add", damped_trend=True,
                                    seasonal=None).fit(optimized=True)
        f12 = full.forecast(12)
        resid_std = float((y - full.fittedvalues).std())
        method = "holtwinters-damped"
    except Exception as e:
        print(f"Scenario fit failed ({e}); naive-trend fallback")
        slope = float((y.iloc[-1] - y.iloc[0]) / max(len(y) - 1, 1))
        idx = pd.date_range(y.index[-1] + pd.offsets.MonthEnd(1), periods=12, freq="ME")
        f12 = pd.Series([y.iloc[-1] + slope * (i + 1) for i in range(12)], index=idx)
        resid_std = float(y.diff().std())
        method = "naive-trend-fallback"
    resid_std = max(resid_std, float(y.std() * 0.05))
    scen = pd.DataFrame({"month": f12.index, "median_price_scenario": f12.values})
    scen["lower"] = scen["median_price_scenario"] - 1.28 * resid_std
    scen["upper"] = scen["median_price_scenario"] + 1.28 * resid_std
    scen.to_csv(os.path.join(HERE, "scenario_projection.csv"), index=False)
    print(f"Scenario method={method} resid_std=${resid_std:,.0f}; "
          f"6-mo end=${scen['median_price_scenario'].iloc[5]:,.0f}")
    plt.figure()
    plt.plot(y.index, y.values, marker="o", label="History (monthly median)")
    plt.plot(holdout.index, holdout.values, marker="o", linestyle="--", label="Holdout (last 3 mo)")
    plt.plot(scen["month"].values[:6], scen["median_price_scenario"].values[:6],
             marker="o", label="Scenario 1-6 mo")
    plt.plot(scen["month"].values[5:], scen["median_price_scenario"].values[5:],
             linestyle=":", label="Scenario 7-12 mo (stretch)")
    plt.fill_between(scen["month"].values, scen["lower"].values, scen["upper"].values,
                     alpha=0.2, label="~80% band")
    plt.figtext(0.01, 0.01, SOURCE_TAG + " | SCENARIO, not a forecast: 12 monthly points only.",
                fontsize=8, color="gray")
    plt.title("Market scenario: monthly median price + 12-mo projection")
    plt.xlabel("Month")
    plt.ylabel("Median price (USD)")
    plt.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(FIGDIR, "scenario.png"), dpi=150)
    plt.close()
    return {"naive_mae": naive_mae, "hw_mae": hw_mae, "method": method,
            "monthly_periods": len(y)}


def fred_overlay():
    """Optional long-run context. Looks for data/external/seattle_hpi.csv; skips if absent."""
    p = os.path.join(DATADIR, "external", "seattle_hpi.csv")
    if not os.path.exists(p):
        print("FRED overlay skipped (no data/external/seattle_hpi.csv). "
              "To add: download a Seattle monthly HPI CSV with columns date,value.")
        return False
    h = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    plt.figure()
    plt.plot(h["date"], h["value"])
    plt.figtext(0.01, 0.01, "Source: FRED/FHFA Seattle HPI (external context, not model input).",
                fontsize=8, color="gray")
    plt.title("Seattle house-price index, long run (external context)")
    plt.xlabel("Date")
    plt.ylabel("Index")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(os.path.join(FIGDIR, "fred_context.png"), dpi=150)
    plt.close()
    print(f"FRED overlay plotted from {p} ({len(h)} rows)")
    return True


def main():
    df, source = load_data()
    assert all(c in df.columns for c in ["price", "sqft_living", "grade", "date"]), "schema mismatch"
    df = clean(df)
    df = engineer(df)
    print(df[["price", "sqft_living", "grade", "price_per_sqft"]].describe().to_string())
    print("Corr with price:\n" + df[FEATURES + ["price"]].corr(numeric_only=True)["price"].sort_values(ascending=False).to_string())
    plot_all(df)
    metrics, _ = model(df)
    fwd = time_aware_eval(df)
    scen = backtest_and_scenario(df)
    fred = fred_overlay()
    print(f"source={source} rows={len(df)} figures={sorted(os.listdir(FIGDIR))}")
    return {"metrics": metrics, "forward": fwd.to_dict("records"),
            "scenario": scen, "fred": fred}


if __name__ == "__main__":
    main()
