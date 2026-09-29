import pandas as pd
import plotly.express as px
import streamlit as st

from src.ai import (detect_anomalies, forecast_spending, generate_insights,
                    predict_category, train_categorizer)

st.set_page_config(page_title="AI Expense Tracker", page_icon="💸", layout="wide")
st.title("💸 AI-Powered Expense Tracker & Financial Analytics Dashboard")

# ---------------------------------------------------------------- data loading
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


def standardise(raw: pd.DataFrame) -> pd.DataFrame:
    """Let the user map their own CSV columns onto the ones the app needs."""
    # Kaggle expense datasets often mix income and expenses: keep expenses only
    for col in raw.columns:
        if col.lower().replace(" ", "") in ("income/expense", "type") and raw[col].astype(str).str.lower().isin(["income", "expense"]).all():
            raw = raw[raw[col].astype(str).str.lower() == "expense"]
            break
    cols = ["-- none --"] + list(raw.columns)

    def guess(names):
        for c in raw.columns:
            if c.lower() in names:
                return cols.index(c)
        return 0

    with st.sidebar.expander("Column mapping", expanded=False):
        d = st.selectbox("Date", cols, guess({"date", "transaction date", "timestamp"}))
        a = st.selectbox("Amount", cols, guess({"amount", "price", "value", "expense"}))
        t = st.selectbox("Description", cols, guess({"description", "merchant", "note", "title", "name"}))
        c = st.selectbox("Category", cols, guess({"category"}))
        p = st.selectbox("Payment method", cols, guess({"payment_method", "payment", "mode", "account"}))
    if d == "-- none --" or a == "-- none --":
        st.error("Please map at least the Date and Amount columns.")
        st.stop()
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[d], errors="coerce"),
        "amount": pd.to_numeric(raw[a], errors="coerce").abs(),
        "description": raw[t].fillna("").astype(str) if t != "-- none --" else "",
        "category": raw[c].astype(str) if c != "-- none --" else "Uncategorized",
        "payment_method": raw[p].astype(str) if p != "-- none --" else "Unknown",
    })
    df = df.dropna(subset=["date", "amount"])
    # blank notes -> fall back to the category name so the text model still has something
    blank = df["description"].str.strip().isin(["", "nan"])
    df.loc[blank, "description"] = df.loc[blank, "category"]
    return df.sort_values("date").reset_index(drop=True)


st.sidebar.header("📂 Data")
upload = st.sidebar.file_uploader("Upload expense CSV (Kaggle etc.)", type="csv")
if upload:
    df = standardise(load_csv(upload))
else:
    df = standardise(load_csv("data/expense_data_1.csv"))
    st.sidebar.caption("Using bundled Kaggle dataset (My Expenses Data).")

# session-added expenses
if "extra" not in st.session_state:
    st.session_state.extra = []
if st.session_state.extra:
    df = pd.concat([df, pd.DataFrame(st.session_state.extra)], ignore_index=True)

# ------------------------------------------------------------------- filters
st.sidebar.header("🔎 Filters")
dmin, dmax = df["date"].min().date(), df["date"].max().date()
rng = st.sidebar.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(rng, tuple) and len(rng) == 2:
    df = df[(df["date"].dt.date >= rng[0]) & (df["date"].dt.date <= rng[1])]
cats = sorted(df["category"].unique())
sel = st.sidebar.multiselect("Categories", cats, default=cats)
df = df[df["category"].isin(sel)]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ----------------------------------------------------------------------- KPIs
monthly_total = df.set_index("date")["amount"].resample("MS").sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total spend", f"₹{df.amount.sum():,.0f}")
k2.metric("Avg / month", f"₹{monthly_total.mean():,.0f}")
k3.metric("Transactions", f"{len(df):,}")
k4.metric("Top category", df.groupby("category").amount.sum().idxmax())

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 AI Insights", "🎯 Budgets", "🧾 Transactions"])

# ------------------------------------------------------------------- overview
with tab1:
    c1, c2 = st.columns([3, 2])
    m = monthly_total.reset_index()
    c1.plotly_chart(px.bar(m, x="date", y="amount", title="Monthly spending"), width='stretch')
    bycat = df.groupby("category", as_index=False)["amount"].sum()
    c2.plotly_chart(px.pie(bycat, names="category", values="amount", hole=0.45, title="Spend by category"), width='stretch')
    c3, c4 = st.columns(2)
    piv = df.assign(month=df.date.dt.to_period("M").astype(str)).pivot_table(index="month", columns="category", values="amount", aggfunc="sum").fillna(0)
    c3.plotly_chart(px.area(piv, title="Category trend over time"), width='stretch')
    pay = df.groupby("payment_method", as_index=False)["amount"].sum()
    c4.plotly_chart(px.bar(pay, x="payment_method", y="amount", title="By payment method"), width='stretch')
    dow = df.assign(day=df.date.dt.day_name()).groupby("day")["amount"].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).reset_index()
    st.plotly_chart(px.bar(dow, x="day", y="amount", title="Average transaction by weekday"), width='stretch')

# ---------------------------------------------------------------- AI insights
with tab2:
    st.subheader("Smart insights")
    budgets = st.session_state.get("budgets", {})
    for tip in generate_insights(df, budgets):
        st.markdown(f"- {tip}")

    st.subheader("📈 Spending forecast (next 3 months)")
    hist, fc = forecast_spending(df)
    if fc.empty:
        st.info("Need at least 2 months of data to forecast.")
    else:
        hist = hist.assign(type="Actual")
        fc = fc.assign(type="Forecast")
        st.plotly_chart(px.line(pd.concat([hist, fc]), x="date", y="amount", color="type", markers=True), width='stretch')
        st.caption("Linear-regression trend on monthly totals (partial first/last months excluded from training). With only a few months of data, treat this as indicative.")

    st.subheader("🚨 Anomaly detection")
    thr = st.slider("Sensitivity (z-score threshold – lower = more alerts)", 1.5, 5.0, 2.0, 0.5)
    an = detect_anomalies(df, thr)
    flagged = an[an.is_anomaly].sort_values("z_score", ascending=False)
    st.write(f"**{len(flagged)}** unusual transactions found.")
    st.dataframe(flagged[["date", "description", "category", "amount", "z_score"]].round({"amount": 2, "z_score": 2}), width='stretch')

    st.subheader("🏷️ Auto-categorisation (NLP)")
    model, acc = train_categorizer(df)
    if model is None:
        st.info("Not enough categorised data to train the model.")
    else:
        st.write(f"TF-IDF + Logistic Regression — hold-out accuracy: **{acc:.1%}**")
        txt = st.text_input("Try a description", "Metro ride to college")
        if txt:
            cat, conf = predict_category(model, txt)
            st.success(f"Predicted category: **{cat}** (confidence {conf:.0%})")

# -------------------------------------------------------------------- budgets
with tab3:
    st.subheader("Set monthly budgets (₹)")
    budgets = st.session_state.setdefault("budgets", {})
    cols = st.columns(3)
    for i, c in enumerate(cats):
        budgets[c] = cols[i % 3].number_input(c, min_value=0, value=int(budgets.get(c, 0)), step=500, key=f"b_{c}")
    latest = df[df.date >= df.date.max().replace(day=1)].groupby("category")["amount"].sum()
    rows = [{"category": c, "budget": b, "spent (latest month)": round(latest.get(c, 0), 2)} for c, b in budgets.items() if b > 0]
    if rows:
        bd = pd.DataFrame(rows)
        bd["used %"] = (bd["spent (latest month)"] / bd["budget"] * 100).round(0)
        st.dataframe(bd, width='stretch')
        st.plotly_chart(px.bar(bd, x="category", y=["budget", "spent (latest month)"], barmode="group"), width='stretch')

# --------------------------------------------------------------- transactions
with tab4:
    st.subheader("➕ Add an expense")
    with st.form("add"):
        f1, f2, f3, f4 = st.columns(4)
        nd = f1.date_input("Date", df.date.max().date())
        nt = f2.text_input("Description")
        na = f3.number_input("Amount (₹)", min_value=0.0, step=10.0)
        auto = f4.checkbox("Auto-categorise with AI", True)
        if st.form_submit_button("Add") and nt and na > 0:
            model, _ = train_categorizer(df)
            cat = predict_category(model, nt)[0] if (auto and model) else "Uncategorized"
            st.session_state.extra.append({"date": pd.Timestamp(nd), "description": nt, "amount": na,
                                           "category": cat, "payment_method": "Manual"})
            st.rerun()
    st.dataframe(df.sort_values("date", ascending=False), width='stretch')
    st.download_button("⬇️ Download filtered CSV", df.to_csv(index=False), "expenses_filtered.csv", "text/csv")
