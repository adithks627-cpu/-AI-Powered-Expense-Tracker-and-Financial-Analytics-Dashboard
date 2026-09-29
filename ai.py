"""ML / analytics helpers: categorisation, anomaly detection, forecasting."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def train_categorizer(df: pd.DataFrame):
    """TF-IDF + Logistic Regression that predicts a category from the description.
    Returns (model, holdout_accuracy) or (None, None) if not enough data."""
    data = df[df["category"] != "Uncategorized"]
    if data["category"].nunique() < 2 or len(data) < 30:
        return None, None
    X, y = data["description"].str.lower(), data["category"]
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
    model.fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    model.fit(X, y)  # refit on everything for use
    return model, acc


def predict_category(model, text: str):
    proba = model.predict_proba([text.lower()])[0]
    i = int(np.argmax(proba))
    return model.classes_[i], float(proba[i])


def detect_anomalies(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Flag unusually large expenses using a per-category z-score on log(amount)."""
    out = df.copy()
    out["log_amt"] = np.log1p(out["amount"])
    g = out.groupby("category")["log_amt"]
    std = g.transform("std").replace(0, np.nan)
    out["z_score"] = ((out["log_amt"] - g.transform("mean")) / std).fillna(0)
    out["is_anomaly"] = out["z_score"] > threshold
    return out.drop(columns="log_amt")


def forecast_spending(df: pd.DataFrame, horizon: int = 3):
    """Linear-trend forecast of monthly spending. Returns (history_df, forecast_df)."""
    monthly = df.set_index("date")["amount"].resample("MS").sum().reset_index()
    train = monthly.iloc[:-1] if len(monthly) > 2 else monthly  # drop partial latest month
    if df["date"].min().day > 7 and len(train) > 3:  # drop partial first month too
        train = train.iloc[1:]
    if len(train) < 2:
        return monthly, pd.DataFrame(columns=["date", "amount"])
    X = np.arange(len(train)).reshape(-1, 1)
    lr = LinearRegression().fit(X, train["amount"])
    future_x = np.arange(len(train), len(train) + horizon).reshape(-1, 1)
    dates = pd.date_range(train["date"].iloc[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
    fc = pd.DataFrame({"date": dates, "amount": np.maximum(lr.predict(future_x), 0)})
    return monthly, fc


def generate_insights(df: pd.DataFrame, budgets: dict | None = None) -> list[str]:
    """Plain-English insights computed from the data."""
    if df.empty:
        return ["No data in the selected range."]
    tips = []
    by_cat = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    top, share = by_cat.index[0], by_cat.iloc[0] / by_cat.sum() * 100
    tips.append(f"💡 **{top}** is your biggest spending area ({share:.0f}% of total).")

    monthly = df.set_index("date")["amount"].resample("MS").sum()
    if len(monthly) >= 3:
        last, prev = monthly.iloc[-2], monthly.iloc[-3]  # last full month vs the one before
        if prev > 0:
            ch = (last - prev) / prev * 100
            arrow = "📈" if ch > 0 else "📉"
            tips.append(f"{arrow} Spending in {monthly.index[-2]:%b %Y} was {abs(ch):.0f}% {'higher' if ch > 0 else 'lower'} than the month before.")
    weekend = df[df["date"].dt.dayofweek >= 5]["amount"].mean()
    weekday = df[df["date"].dt.dayofweek < 5]["amount"].mean()
    if weekday and weekend > weekday * 1.15:
        tips.append(f"🎉 Weekend transactions average {weekend / weekday:.1f}× more than weekday ones.")
    if budgets:
        cur = df[df["date"] >= df["date"].max().replace(day=1)].groupby("category")["amount"].sum()
        for c, b in budgets.items():
            if b > 0 and cur.get(c, 0) > b:
                tips.append(f"🚨 Over budget on **{c}** this month: ₹{cur[c]:,.0f} vs ₹{b:,.0f}.")
    return tips
