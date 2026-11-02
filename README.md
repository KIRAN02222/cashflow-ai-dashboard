# CashFlow AI — Dashboard Upgrade

## Upgrade highlights
1. Expanded forecasting features: calendar seasonality, 7/14/21/28/35/56-day lags, rolling means and volatility.
2. Chronological 80/20 holdout evaluation, including a 7-day seasonal-naive baseline.
3. Models are refitted on the full available history after evaluation before forecasting.
4. Premium dark interface, executive overview, improved charts and report downloads.
5. AI-assisted business insights generated from transparent data-driven rules: returns adjustment, weekday pattern, recent momentum, forecast comparison and unusual days.

## Install / run (Windows PowerShell)
Run these in the VS Code **Terminal**, not the top-right Run Python File button:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If your `(venv)` is already active, keep using it. Upload `OnlineRetail.csv` in the app sidebar.

## Important interpretation
The Online Retail dataset contains invoice transactions, not bank receipts, settlement dates or bank balances. This app forecasts daily net sales. Positive invoice amounts are called a sales inflow proxy and negative amounts a returns/cancellations adjustment. Do not describe these as verified cash movements.

## Model evaluation notes
The app reports MAE, RMSE and R² on a chronological holdout. It also reports a seasonal-naive 7-day baseline. A model should not be described as better than the baseline unless its holdout metrics support that claim. No accuracy improvement is guaranteed; compare the new metrics with the previous app's metrics using the same dataset and split.
