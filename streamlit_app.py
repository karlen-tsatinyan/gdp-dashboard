import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import BayesianRidge
import streamlit as st
import yfinance as yf

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Predictive Quant Engine",
    page_icon="📈",
    layout="wide"
)

# =========================================================
# DATA SETTINGS
# =========================================================
POPULAR_TICKERS = ["NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","AMD","NFLX","AVGO","JPM","XOM","LLY","UNH","SPY"]
POPULAR_BENCHMARKS = ["SPY","QQQ","IWM","DIA","TLT","GLD"]

# =========================================================
# TITLE
# =========================================================
st.title("📊 Quantitative Predictive Engine (Full Stack Backtest)")

# =========================================================
# CONTROLS
# =========================================================
current_year = datetime.datetime.now().year

c1, c2, c3 = st.columns([1,1,2])

with c1:
    ticker = st.selectbox("Ticker", POPULAR_TICKERS, index=POPULAR_TICKERS.index("AAPL"))

with c2:
    benchmark = st.selectbox("Benchmark", POPULAR_BENCHMARKS, index=0)

with c3:
    start_year, end_year = st.slider(
        "Date Range",
        min_value=2000,
        max_value=current_year,
        value=(2015, current_year)
    )

# Fetch extra historical data to fulfill the 200 SMA requirement
FETCH_START = f"{start_year-1}-01-01"
END_DATE = f"{end_year}-12-31"

# =========================================================
# DATA LOADER
# =========================================================
@st.cache_data
def load_data(ticker, benchmark, start, end):
    # Forcing multi_level_index=False flattens the returned DataFrames
    stock = yf.download(ticker, start=start, end=end, progress=False, multi_level_index=False)
    bench = yf.download(benchmark, start=start, end=end, progress=False, multi_level_index=False)

    if stock.empty or bench.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Standardize timezones to avoid merge mismatches
    if stock.index.tz is not None:
        stock.index = stock.index.tz_localize(None)
    if bench.index.tz is not None:
        bench.index = bench.index.tz_localize(None)

    stock.dropna(subset=["Close"], inplace=True)
    bench.dropna(subset=["Close"], inplace=True)

    return stock, bench

stock, bench = load_data(ticker, benchmark, FETCH_START, END_DATE)

if stock.empty or bench.empty:
    st.error("No data loaded from Yahoo Finance. Please check your connection or ticker symbols.")
    st.stop()

# =========================================================
# FEATURE ENGINEERING
# =========================================================
df = stock.copy()
df["ret_1d"] = df["Close"].pct_change()

bench_df = bench[["Close"]].copy().rename(columns={"Close": "bench_close"})
bench_df["bench_ret"] = bench_df["bench_close"].pct_change()

# Clean join on the Date index directly
df = df.join(bench_df[["bench_close", "bench_ret"]], how="inner")

# Indicators
df["vol_20"] = df["ret_1d"].rolling(20).std()
df["sma_200"] = df["Close"].rolling(200).mean()

# RSI
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / (loss + 1e-9)
df["rsi"] = 100 - (100 / (1 + rs))

# Drop rows missing indicators, but preserve data check
df.dropna(subset=["ret_1d", "vol_20", "sma_200", "rsi", "bench_ret"], inplace=True)

if df.empty:
    st.error(f"DataFrame is empty after calculating technical indicators. You need a wider date range to calculate the 200 SMA for {ticker}.")
    st.stop()

# Filter data to match user's requested display range
display_start_date = pd.to_datetime(f"{start_year}-01-01")
plot_df = df[df.index >= display_start_date].copy()

if plot_df.empty:
    plot_df = df.copy()  # Fallback if range filter is too restrictive

# =========================================================
# ML MODEL
# =========================================================
model_df = plot_df.copy()
model_df["future_return"] = model_df["Close"].pct_change().shift(-1)

features = ["ret_1d", "vol_20", "rsi"]

# Separate historical training data from the prediction target
train_df = model_df.dropna(subset=features + ["future_return"]).copy()

if train_df.empty:
    st.error("Not enough data to train the prediction model.")
    st.stop()

X = train_df[features]
y = train_df["future_return"]

model = BayesianRidge()
model.fit(X, y)

# Pull the absolute latest available data row for live prediction
latest = model_df.iloc[-1]
X_future = latest[features].values.reshape(1, -1)

pred, std = model.predict(X_future, return_std=True)

current_price = float(latest["Close"])
pred_price = current_price * (1 + pred[0])

low = pred_price - 1.96 * std[0] * current_price
high = pred_price + 1.96 * std[0] * current_price

direction = "UP 📈" if pred_price > current_price else "DOWN 📉"

# =========================================================
# BACKTEST STRATEGY
# =========================================================
plot_df["signal"] = np.where(plot_df["rsi"] > 50, 1, -1)
plot_df["strategy_returns"] = plot_df["signal"] * plot_df["ret_1d"]

plot_df["equity"] = (1 + plot_df["strategy_returns"]).cumprod()
plot_df["benchmark_equity"] = (1 + plot_df["bench_ret"]).cumprod()

# Metrics
wins = plot_df[plot_df["strategy_returns"] > 0]
losses = plot_df[plot_df["strategy_returns"] < 0]

win_rate = len(wins) / (len(wins) + len(losses) + 1e-9)
profit_factor = plot_df["strategy_returns"][plot_df["strategy_returns"] > 0].sum() / (abs(plot_df["strategy_returns"][plot_df["strategy_returns"] < 0].sum()) + 1e-9)

sharpe = (plot_df["strategy_returns"].mean() / (plot_df["strategy_returns"].std() + 1e-9)) * np.sqrt(252)

drawdown = (plot_df["equity"] / plot_df["equity"].cummax()) - 1
max_dd = drawdown.min()

# =========================================================
# DASHBOARD METRICS DISPLAY
# =========================================================
st.subheader("📊 Live Prediction")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", f"${current_price:.2f}")
col2.metric("Prediction", direction)
col3.metric("Win Rate", f"{win_rate*100:.2f}%")
col4.metric("Sharpe", f"{sharpe:.2f}")

st.info(f"""
**Predicted Target Price:** ${pred_price:.2f}  
**Expected Range (95% CI):** ${low:.2f} → ${high:.2f}  
**Profit Factor:** {profit_factor:.2f}  
**Max Strategy Drawdown:** {max_dd*100:.2f}%
""")

# =========================================================
# EQUITY CURVES PLOT
# =========================================================
st.subheader("📈 Equity Curve vs Benchmark")

fig = go.Figure()
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["equity"], name="Strategy Strategy"))
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["benchmark_equity"], name="Benchmark", line=dict(dash="dot")))
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig, use_container_width=True)

# =========================================================
# DRAWDOWN PLOT
# =========================================================
st.subheader("📉 Drawdown")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=plot_df.index, y=drawdown * 100, fill="tozeroy", name="Drawdown", line=dict(color="red")))
fig2.update_layout(template="plotly_dark", height=250, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# PRICE + SIGNAL PLOT
# =========================================================
st.subheader("📊 Price Chart")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Price"))
fig3.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_200"], name="SMA 200", line=dict(color="orange")))
fig3.update_layout(template="plotly_dark", height=400, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig3, use_container_width=True)