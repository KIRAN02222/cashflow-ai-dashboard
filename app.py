
import warnings
warnings.filterwarnings("ignore")

from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# ============================================================
# APP CONFIG + PROFESSIONAL DARK THEME
# ============================================================
st.set_page_config(
    page_title="CashFlow AI | Forecast Studio",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080d12; color: #e8eef5; }
[data-testid="stSidebar"] { background: #0b141c; border-right: 1px solid #1e303a; }
[data-testid="stHeader"] { background: rgba(8,13,18,.94); }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; color: #f2f7fa; }
.hero {
  padding: 26px 30px; border-radius: 18px; margin: 8px 0 20px;
  background: linear-gradient(115deg,#102820 0%,#101d27 58%,#142330 100%);
  border: 1px solid #285342;
}
.hero-kicker { color:#62e6aa; font-size: .78rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
.hero h1 { font-size:2.05rem; margin:.35rem 0; }
.hero p { color:#aebdc8; margin:0; }
div[data-testid="stMetric"] {
  background: linear-gradient(145deg,#111e27,#0d171f);
  border: 1px solid #203541; padding: 17px 18px; border-radius: 14px;
}
div[data-testid="stMetricLabel"] { color:#a9bac5; }
div[data-testid="stMetricValue"] { color:#5ce2a5; }
div[data-testid="stMetricDelta"] { font-size:.8rem; }
.insight {
  background: linear-gradient(120deg,#10251f,#101b25);
  border:1px solid #285342; border-left:4px solid #53e0a1;
  padding:14px 16px; border-radius:10px; margin:8px 0;
}
.insight-title { color:#66e6ae; font-weight:700; margin-bottom:4px; }
.insight-body { color:#d7e2e9; font-size:.94rem; }
.muted { color:#98aab6; font-size:.86rem; }
div.stButton > button, div.stDownloadButton > button {
  background:#27b47d; color:#06140e; border:0; border-radius:9px; font-weight:700;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  background:#58e5a5; color:#06140e;
}
</style>
""", unsafe_allow_html=True)

GREEN, BLUE, RED, AMBER = "#53e0a1", "#38bdf8", "#fb7185", "#fbbf24"
TEMPLATE = "plotly_dark"

# ============================================================
# DATA PREPARATION
# ============================================================
def parse_dates(s):
    try:
        return pd.to_datetime(s, format="mixed", dayfirst=False, errors="coerce")
    except (TypeError, ValueError):
        return pd.to_datetime(s, dayfirst=False, errors="coerce")

@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    raw = pd.read_csv(BytesIO(file_bytes), encoding="latin1")
    required = ["InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "Country"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    original = len(raw)
    df = raw.drop_duplicates().copy()
    duplicates = original - len(df)
    df["InvoiceDate"] = parse_dates(df["InvoiceDate"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate", "Quantity", "UnitPrice"])
    df["Country"] = df["Country"].fillna("Unknown").astype(str)
    if "CustomerID" in df:
        df["CustomerID"] = df["CustomerID"].fillna("Unknown")
    if "Description" in df:
        df["Description"] = df["Description"].fillna("Unknown")
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    df["Date"] = df["InvoiceDate"].dt.floor("D")
    return df, {"original": original, "duplicates": duplicates, "clean": len(df)}

def daily_series(df):
    s = df.groupby("Date")["TotalAmount"].sum().sort_index()
    s = s.asfreq("D", fill_value=0).astype(float)
    s.name = "NetSales"
    return s

# Expanded calendar + seasonal lag feature set.
FEATURES = [
    "Day", "Month", "Year", "DayOfWeek", "DayOfYear",
    "WeekOfYear", "IsWeekend", "MonthSin", "MonthCos",
    "DowSin", "DowCos",
    "Lag_1", "Lag_7", "Lag_14", "Lag_21", "Lag_28", "Lag_35",
    "Lag_56", "Rolling_Mean_7", "Rolling_Mean_14",
    "Rolling_Mean_28", "Rolling_Std_7"
]

def feature_row(date, history):
    doy = date.dayofyear
    dow = date.dayofweek
    return {
        "Day": date.day,
        "Month": date.month,
        "Year": date.year,
        "DayOfWeek": dow,
        "DayOfYear": doy,
        "WeekOfYear": int(date.isocalendar().week),
        "IsWeekend": int(dow >= 5),
        "MonthSin": np.sin(2 * np.pi * date.month / 12),
        "MonthCos": np.cos(2 * np.pi * date.month / 12),
        "DowSin": np.sin(2 * np.pi * dow / 7),
        "DowCos": np.cos(2 * np.pi * dow / 7),
        "Lag_1": history.iloc[-1],
        "Lag_7": history.iloc[-7],
        "Lag_14": history.iloc[-14],
        "Lag_21": history.iloc[-21],
        "Lag_28": history.iloc[-28],
        "Lag_35": history.iloc[-35],
        "Lag_56": history.iloc[-56],
        "Rolling_Mean_7": history.iloc[-7:].mean(),
        "Rolling_Mean_14": history.iloc[-14:].mean(),
        "Rolling_Mean_28": history.iloc[-28:].mean(),
        "Rolling_Std_7": history.iloc[-7:].std()
    }

def build_frame(series):
    frame = pd.DataFrame(index=series.index)
    frame["Target"] = series
    for d in series.index:
        pass
    idx = series.index
    frame["Day"] = idx.day
    frame["Month"] = idx.month
    frame["Year"] = idx.year
    frame["DayOfWeek"] = idx.dayofweek
    frame["DayOfYear"] = idx.dayofyear
    frame["WeekOfYear"] = idx.isocalendar().week.astype(int).to_numpy()
    frame["IsWeekend"] = (idx.dayofweek >= 5).astype(int)
    frame["MonthSin"] = np.sin(2 * np.pi * idx.month / 12)
    frame["MonthCos"] = np.cos(2 * np.pi * idx.month / 12)
    frame["DowSin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    frame["DowCos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    for lag in [1, 7, 14, 21, 28, 35, 56]:
        frame[f"Lag_{lag}"] = series.shift(lag)
    shifted = series.shift(1)
    frame["Rolling_Mean_7"] = shifted.rolling(7).mean()
    frame["Rolling_Mean_14"] = shifted.rolling(14).mean()
    frame["Rolling_Mean_28"] = shifted.rolling(28).mean()
    frame["Rolling_Std_7"] = shifted.rolling(7).std()
    return frame.dropna()

def get_models():
    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=240, max_depth=12, min_samples_leaf=2,
            max_features=0.9, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.035, max_depth=2,
            min_samples_leaf=3, loss="huber", random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=240, max_depth=14, min_samples_leaf=2,
            max_features=0.9, random_state=42, n_jobs=-1
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=42
        )
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(
            n_estimators=250, learning_rate=0.035, max_depth=3,
            min_child_weight=3, subsample=0.85, colsample_bytree=0.9,
            reg_lambda=3, objective="reg:squarederror",
            random_state=42, n_jobs=-1, verbosity=0
        )
    return models

@st.cache_resource(show_spinner="Training and evaluating forecasting models...")
def train_evaluate_refit(daily_values):
    idx = pd.to_datetime([x[0] for x in daily_values])
    vals = [float(x[1]) for x in daily_values]
    series = pd.Series(vals, index=idx, name="NetSales").sort_index()
    frame = build_frame(series)
    if len(frame) < 35:
        raise ValueError("Not enough daily history. Please select a country with at least 90 calendar days.")

    split = int(len(frame) * 0.8)
    split = min(max(split, 1), len(frame) - 1)
    train, test = frame.iloc[:split], frame.iloc[split:]
    X_train, y_train = train[FEATURES], train["Target"]
    X_test, y_test = test[FEATURES], test["Target"]

    results, test_predictions, fitted_full = [], {}, {}
    # Seasonal-naive baseline: same weekday from the previous week.
    baseline = test["Lag_7"].to_numpy()
    results.append({
        "Algorithm": "Seasonal Naive (7-day)",
        "MAE": mean_absolute_error(y_test, baseline),
        "RMSE": np.sqrt(mean_squared_error(y_test, baseline)),
        "R2 Score": r2_score(y_test, baseline) if len(y_test) > 1 else np.nan
    })
    test_predictions["Seasonal Naive (7-day)"] = baseline

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Algorithm": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2 Score": r2_score(y_test, pred) if len(y_test) > 1 else np.nan
        })
        test_predictions[name] = pred

        # Refit a fresh copy on ALL feature rows for future forecasting.
        full_model = get_models()[name]
        full_model.fit(frame[FEATURES], frame["Target"])
        fitted_full[name] = full_model

    metrics = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return {
        "series": series, "frame": frame, "test": test,
        "y_test": y_test, "metrics": metrics,
        "test_predictions": test_predictions,
        "models_full": fitted_full
    }

def forecast_future(model, series, days):
    history = series.copy().astype(float)
    future_dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=days, freq="D")
    out = []
    for date in future_dates:
        x = pd.DataFrame([feature_row(date, history)], columns=FEATURES)
        pred = float(model.predict(x)[0])
        out.append(pred)
        history.loc[date] = pred
    return pd.DataFrame({"Date": future_dates, "Predicted_Net_Sales": out})

def fmt_money(v):
    return f"{v:,.2f}"

def style_chart(fig, height=360):
    fig.update_layout(
        template=TEMPLATE, height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce6ed"), margin=dict(l=15,r=15,t=55,b=15),
        legend=dict(orientation="h", y=1.12, x=0)
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#24333d", zeroline=False)
    return fig

def insight_card(title, body):
    st.markdown(
        f'<div class="insight"><div class="insight-title">{title}</div>'
        f'<div class="insight-body">{body}</div></div>',
        unsafe_allow_html=True
    )

def generate_insights(df, daily, forecast, metrics, model_name):
    notes = []
    positive = df.loc[df["TotalAmount"] > 0, "TotalAmount"].sum()
    negative = abs(df.loc[df["TotalAmount"] < 0, "TotalAmount"].sum())
    gross = positive + negative
    return_rate = (negative / gross * 100) if gross else 0
    notes.append(("Returns monitoring",
                  f"Negative invoice adjustments represent {return_rate:.1f}% of the absolute transaction value. Review return reasons and product-level patterns before treating this as a cash refund rate."))

    weekday = daily.groupby(daily.index.day_name()).mean()
    if len(weekday):
        best_day = weekday.idxmax()
        low_day = weekday.idxmin()
        notes.append(("Weekly sales pattern",
                      f"{best_day} has the highest average daily net sales ({fmt_money(weekday.max())}), while {low_day} has the lowest ({fmt_money(weekday.min())}). Use this pattern when planning staffing and inventory."))

    recent = daily.tail(28).mean()
    prior = daily.iloc[-56:-28].mean() if len(daily) >= 56 else daily.head(max(1, len(daily)-28)).mean()
    if pd.notna(prior) and abs(prior) > 1e-9:
        change = (recent - prior) / abs(prior) * 100
        direction = "higher" if change >= 0 else "lower"
        notes.append(("Recent momentum",
                      f"Average daily net sales over the latest 28 days are {abs(change):.1f}% {direction} than the preceding comparison period. This is a descriptive comparison, not proof of the cause."))

    if len(forecast):
        fmean = forecast["Predicted_Net_Sales"].mean()
        histmean = daily.tail(28).mean()
        if abs(histmean) > 1e-9:
            delta = (fmean - histmean) / abs(histmean) * 100
            trend = "above" if delta >= 0 else "below"
            notes.append(("Forecast outlook",
                          f"The selected {model_name} model forecasts an average daily net sales value {abs(delta):.1f}% {trend} the latest 28-day average. Treat this as a model estimate, not guaranteed cash receipts."))

    # Robust anomaly flags based on the interquartile range of daily values.
    q1, q3 = daily.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr > 0:
        unusual = daily[(daily < q1 - 1.5*iqr) | (daily > q3 + 1.5*iqr)]
        if len(unusual):
            date = unusual.abs().idxmax()
            notes.append(("Unusual sales day",
                          f"{len(unusual)} days fall outside the historical 1.5×IQR range. The largest-magnitude flagged day is {date:%d %b %Y} ({fmt_money(daily.loc[date])}). Check promotions, bulk orders, cancellations or data quality."))

    best = metrics.iloc[0]
    notes.append(("Model selection",
                  f"{best['Algorithm']} currently has the lowest holdout RMSE ({fmt_money(best['RMSE'])}). The comparison includes a seasonal-naive benchmark; model quality should be judged against that baseline too."))
    return notes

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 💹 CashFlow AI")
st.sidebar.caption("FORECAST STUDIO  •  FINANCIAL ANALYTICS")
uploaded = st.sidebar.file_uploader("Upload OnlineRetail.csv", type=["csv"])
if uploaded is None:
    st.markdown('<div class="hero"><div class="hero-kicker">Machine Learning Forecast Studio</div><h1>AI-Powered Cash Flow Forecasting</h1><p>Understand historical retail performance, compare forecasting models and explore data-driven business insights.</p></div>', unsafe_allow_html=True)
    st.info("Upload your OnlineRetail.csv file in the sidebar to begin.")
    st.stop()

try:
    df, cleaning = load_data(uploaded.getvalue())
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()
if df.empty:
    st.error("No usable rows were found after preprocessing.")
    st.stop()

st.sidebar.success(f"{len(df):,} cleaned rows")
country_options = ["All countries"] + sorted(df["Country"].unique().tolist())
country = st.sidebar.selectbox("Market / Country", country_options)
market_df = df if country == "All countries" else df[df["Country"] == country].copy()
if market_df.empty:
    st.warning("No rows found for this country.")
    st.stop()

dmin, dmax = market_df["Date"].min().date(), market_df["Date"].max().date()
date_choice = st.sidebar.date_input("Analytics period", (dmin, dmax), min_value=dmin, max_value=dmax)
if isinstance(date_choice, (tuple, list)) and len(date_choice) == 2:
    date_start, date_end = date_choice
else:
    date_start, date_end = dmin, dmax
view_df = market_df[(market_df["Date"].dt.date >= date_start) & (market_df["Date"].dt.date <= date_end)].copy()

horizon = st.sidebar.slider("Forecast horizon", 7, 90, 30)
page = st.sidebar.radio("WORKSPACE", ["Overview", "Cash Flow", "Forecast", "Model Lab", "AI Insights", "Data Explorer"])

# Train on full selected-market history, not just the date filter.
daily = daily_series(market_df)
if len(daily) < 100:
    st.error("This market has fewer than 100 calendar days. Choose All countries or a market with more history.")
    st.stop()

daily_key = tuple((str(pd.Timestamp(d).date()), float(v)) for d, v in daily.items())
try:
    pack = train_evaluate_refit(daily_key)
except Exception as e:
    st.error(f"Forecast model training failed: {e}")
    st.stop()

metrics = pack["metrics"]
# Best predictive ML model (exclude baseline if it wins).
ml_metrics = metrics[metrics["Algorithm"] != "Seasonal Naive (7-day)"]
best_model = str(ml_metrics.sort_values("RMSE").iloc[0]["Algorithm"])
selected_model = st.sidebar.selectbox(
    "Forecast algorithm", list(pack["models_full"].keys()),
    index=list(pack["models_full"].keys()).index(best_model)
)
future = forecast_future(pack["models_full"][selected_model], daily, horizon)

inflow = view_df.loc[view_df["TotalAmount"] > 0, "TotalAmount"].sum()
returns_abs = abs(view_df.loc[view_df["TotalAmount"] < 0, "TotalAmount"].sum())
net = view_df["TotalAmount"].sum()
transactions = view_df["InvoiceNo"].nunique()
customers = view_df["CustomerID"].nunique() if "CustomerID" in view_df else 0
forecast_sum = future["Predicted_Net_Sales"].sum()

st.markdown(
    '<div class="hero"><div class="hero-kicker">Forecast Studio • Live Analytics</div>'
    '<h1>AI-Powered Cash Flow Forecasting</h1>'
    '<p>Historical performance  ·  Predictive modelling  ·  Actionable business intelligence</p></div>',
    unsafe_allow_html=True
)
st.caption(f"Market: {country}  |  Analytics period: {date_start} to {date_end}  |  Forecast model: {selected_model}")

# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    st.subheader("Executive Overview")
    a,b,c,d = st.columns(4)
    a.metric("Sales Inflow Proxy", fmt_money(inflow))
    b.metric("Returns Adjustment", fmt_money(returns_abs))
    c.metric("Net Sales", fmt_money(net))
    d.metric(f"{horizon}-Day Forecast", fmt_money(forecast_sum))
    e,f,g = st.columns(3)
    e.metric("Transactions", f"{transactions:,}")
    f.metric("Customers", f"{customers:,}")
    g.metric("Clean Records", f"{len(view_df):,}")

    left,right = st.columns([1.55,1])
    daily_view = view_df.groupby("Date")["TotalAmount"].sum().asfreq("D", fill_value=0).rename("Net Sales").reset_index()
    fig = px.area(daily_view, x="Date", y="Net Sales", title="Daily Net Sales")
    fig.update_traces(line_color=GREEN, fillcolor="rgba(83,224,161,.12)")
    left.plotly_chart(style_chart(fig, 370), use_container_width=True)
    if country == "All countries":
        country_rank = view_df.groupby("Country")["TotalAmount"].sum().nlargest(10).sort_values().reset_index()
        fig2 = px.bar(country_rank, x="TotalAmount", y="Country", orientation="h", title="Top 10 Markets")
    else:
        monthly = view_df.set_index("InvoiceDate").resample("MS")["TotalAmount"].sum().reset_index()
        monthly["Month"] = monthly["InvoiceDate"].dt.strftime("%b %Y")
        fig2 = px.bar(monthly, x="Month", y="TotalAmount", title="Monthly Net Sales")
    fig2.update_traces(marker_color=BLUE)
    right.plotly_chart(style_chart(fig2, 370), use_container_width=True)

    st.subheader("Forecast Preview")
    history = daily.tail(75).rename("Actual Net Sales").reset_index()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=history["Date"], y=history["Actual Net Sales"], name="Historical", line=dict(color=BLUE,width=2)))
    fig3.add_trace(go.Scatter(x=future["Date"], y=future["Predicted_Net_Sales"], name="Forecast", line=dict(color=GREEN,width=3,dash="dash")))
    st.plotly_chart(style_chart(fig3, 390), use_container_width=True)

    st.subheader("Business Highlights")
    insights = generate_insights(market_df, daily, future, metrics, selected_model)
    for title, body in insights[:3]:
        insight_card(title, body)

# ============================================================
# CASH FLOW
# ============================================================
elif page == "Cash Flow":
    st.subheader("Cash Flow & Sales Analysis")
    cash = view_df.groupby("Date").agg(
        Sales_Inflow_Proxy=("TotalAmount", lambda x: x[x > 0].sum()),
        Returns_Adjustment=("TotalAmount", lambda x: abs(x[x < 0].sum())),
        Net_Sales=("TotalAmount", "sum")
    ).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cash["Date"], y=cash["Sales_Inflow_Proxy"], name="Sales inflow proxy", line=dict(color=GREEN,width=2)))
    fig.add_trace(go.Scatter(x=cash["Date"], y=cash["Net_Sales"], name="Net sales", line=dict(color=BLUE,width=2)))
    st.plotly_chart(style_chart(fig, 410), use_container_width=True)

    monthly = view_df.set_index("InvoiceDate").resample("MS")["TotalAmount"].sum().reset_index()
    monthly["Month"] = monthly["InvoiceDate"].dt.strftime("%b %Y")
    fig2 = px.bar(monthly, x="Month", y="TotalAmount", title="Monthly Net Sales")
    fig2.update_traces(marker_color=GREEN)
    st.plotly_chart(style_chart(fig2, 360), use_container_width=True)

    weekday = view_df.groupby(view_df["Date"].dt.day_name())["TotalAmount"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    ).reset_index()
    weekday.columns = ["Weekday","Average Net Sales"]
    fig3 = px.bar(weekday, x="Weekday", y="Average Net Sales", title="Average Net Sales by Weekday")
    fig3.update_traces(marker_color=BLUE)
    st.plotly_chart(style_chart(fig3, 340), use_container_width=True)
    st.caption("The dataset records retail invoice transactions, not bank settlement dates. Positive amounts are a sales inflow proxy; negative amounts are a returns/cancellations adjustment.")

# ============================================================
# FORECAST
# ============================================================
elif page == "Forecast":
    st.subheader("Future Net Sales Forecast")
    x,y,z = st.columns(3)
    x.metric("Forecast Total", fmt_money(forecast_sum))
    y.metric("Average / Day", fmt_money(future["Predicted_Net_Sales"].mean()))
    z.metric("Forecast Period", f"{horizon} days")
    hist = daily.tail(90).rename("Historical Net Sales").reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Historical Net Sales"], name="Historical", line=dict(color=BLUE,width=2)))
    fig.add_trace(go.Scatter(x=future["Date"], y=future["Predicted_Net_Sales"], name="Predicted", line=dict(color=GREEN,width=3,dash="dash")))
    fig.update_layout(title=f"{selected_model} — Recursive forecast")
    st.plotly_chart(style_chart(fig, 430), use_container_width=True)
    st.dataframe(future.style.format({"Predicted_Net_Sales":"{:,.2f}"}), use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", future.to_csv(index=False).encode(), "cash_flow_forecast.csv", "text/csv")

# ============================================================
# MODEL LAB
# ============================================================
elif page == "Model Lab":
    st.subheader("Model Evaluation Lab")
    st.caption("Chronological 80/20 holdout. Features for each test date use only observations available before that date. Models are then refitted on all history for forecasting.")
    st.dataframe(metrics.style.format({"MAE":"{:,.2f}","RMSE":"{:,.2f}","R2 Score":"{:.4f}"}), use_container_width=True, hide_index=True)

    baseline_row = metrics[metrics["Algorithm"] == "Seasonal Naive (7-day)"]
    chosen_row = metrics[metrics["Algorithm"] == selected_model]
    if not baseline_row.empty and not chosen_row.empty:
        base_rmse = float(baseline_row.iloc[0]["RMSE"])
        model_rmse = float(chosen_row.iloc[0]["RMSE"])
        if model_rmse < base_rmse:
            improvement = (base_rmse-model_rmse)/base_rmse*100
            st.success(f"{selected_model} beats the 7-day seasonal baseline by {improvement:.1f}% RMSE on this holdout.")
        else:
            gap = (model_rmse-base_rmse)/base_rmse*100
            st.warning(f"The 7-day seasonal baseline currently has lower RMSE than {selected_model} by {gap:.1f}%. Do not claim the ML model is better until this is addressed.")

    fig = px.bar(metrics, x="Algorithm", y="RMSE", color="Algorithm", title="Holdout RMSE — lower is better")
    st.plotly_chart(style_chart(fig, 360), use_container_width=True)

    chosen_pred = pack["test_predictions"][selected_model]
    actual = pack["y_test"]
    compare = pd.DataFrame({"Date":actual.index,"Actual":actual.values,"Predicted":chosen_pred})
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=compare["Date"],y=compare["Actual"],name="Actual",line=dict(color=BLUE,width=2)))
    fig2.add_trace(go.Scatter(x=compare["Date"],y=compare["Predicted"],name="Predicted",line=dict(color=GREEN,width=2)))
    st.plotly_chart(style_chart(fig2, 410), use_container_width=True)

    model_obj = pack["models_full"][selected_model]
    if hasattr(model_obj, "feature_importances_"):
        fi = pd.DataFrame({"Feature":FEATURES,"Importance":model_obj.feature_importances_}).sort_values("Importance")
        fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h", title="Feature Importance (refitted model)")
        fig3.update_traces(marker_color=BLUE)
        st.plotly_chart(style_chart(fig3, 430), use_container_width=True)
    st.info(f"Selected future-forecast model: {selected_model}. The forecast model is refitted on the full available daily history.")

# ============================================================
# AI BUSINESS INSIGHTS
# ============================================================
elif page == "AI Insights":
    st.subheader("AI-Assisted Business Insights")
    st.caption("Automatically generated, data-driven observations. These are rule-based analytics, not advice from a generative AI model.")
    insights = generate_insights(market_df, daily, future, metrics, selected_model)
    for title, body in insights:
        insight_card(title, body)

    st.subheader("Top products by net sales")
    if "Description" in view_df:
        products = view_df.groupby("Description")["TotalAmount"].sum().nlargest(10).sort_values().reset_index()
        fig = px.bar(products, x="TotalAmount", y="Description", orientation="h", title="Top 10 Products")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(style_chart(fig, 430), use_container_width=True)

# ============================================================
# DATA EXPLORER
# ============================================================
elif page == "Data Explorer":
    st.subheader("Data Quality & Explorer")
    a,b,c = st.columns(3)
    a.metric("Original Rows", f"{cleaning['original']:,}")
    b.metric("Duplicates Removed", f"{cleaning['duplicates']:,}")
    c.metric("Clean Rows", f"{cleaning['clean']:,}")
    st.dataframe(view_df.head(1500), use_container_width=True, hide_index=True)
    st.caption("Preview limited to 1,500 rows for browser performance.")
    st.download_button("Download cleaned filtered data", view_df.to_csv(index=False).encode(), "cleaned_online_retail.csv", "text/csv")

    st.subheader("Transaction value distribution")
    amounts = view_df["TotalAmount"].replace([np.inf,-np.inf],np.nan).dropna()
    if len(amounts):
        lo,hi = amounts.quantile([.01,.99])
        fig = px.histogram(amounts.clip(lo,hi), nbins=55, title="Transaction amount (1st–99th percentile view)")
        fig.update_traces(marker_color=BLUE)
        st.plotly_chart(style_chart(fig, 350), use_container_width=True)

st.divider()
st.caption("CashFlow AI • Built with Streamlit, Pandas, Scikit-learn, XGBoost and Plotly.")
st.caption("Research limitation: Online Retail is invoice-level sales data, not bank-account cash-flow data. Predictions represent net sales, not verified cash receipts or cash balance.")
