# AI-Powered Expense Tracker & Financial Analytics Dashboard

An interactive Streamlit dashboard that turns raw expense data into insights using machine learning.

🔗 **Live dashboard:** _add your streamlit.app link here_

## Features
- 📊 **Analytics dashboard** – monthly trend, category split, payment/account breakdown, weekday patterns
- 🏷️ **NLP auto-categorisation** – TF-IDF + Logistic Regression predicts a category from the expense note
- 🚨 **Anomaly detection** – per-category z-score on log(amount) flags unusually large expenses
- 📈 **Spending forecast** – linear-regression trend for the next 3 months
- 💡 **Smart insights** – plain-English observations (top category, month-over-month change, weekend habits)
- 🎯 **Budgets** – set monthly limits per category and track usage
- 📂 **Bring your own data** – upload any expense CSV and map its columns
- ➕ **Add expenses** – new entries are auto-categorised by the model

## Tech stack
Python · Streamlit · Pandas · Plotly · scikit-learn

## Dataset
**My Expenses Data** by Tharun Prabu (Kaggle) – `data/expense_data_1.csv`.
Personal transactions in INR (Nov 2021 – Mar 2022) with date, account, category, note and amount.
The app keeps only rows marked `Expense` (231 of 277) and drops income entries.
Source: https://www.kaggle.com/ (search "My Expenses Data")

## Run locally
```bash
git clone https://github.com/<your-username>/AI-Powered-Expense-Tracker-and-Financial-Analytics-Dashboard.git
cd AI-Powered-Expense-Tracker-and-Financial-Analytics-Dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Project structure
```
app.py                    # Streamlit UI
src/ai.py                 # categoriser, anomaly detection, forecasting, insights
data/expense_data_1.csv   # Kaggle dataset
requirements.txt
```

## Methodology
1. **Cleaning** – dates parsed, income rows removed, blank notes replaced by the category name.
2. **Categorisation** – note text lower-cased, vectorised with TF-IDF (1–2 grams), classified by logistic regression; evaluated on an 80/20 hold-out split.
3. **Anomalies** – an expense is flagged if its log-amount is more than *k* standard deviations above its category mean (adjustable in the app).
4. **Forecast** – ordinary least squares on monthly totals; partial first and last months are excluded from training.

## Limitations
The dataset is small (~4 months, 231 expenses), so the forecast and classifier accuracy are indicative rather than production-grade.
