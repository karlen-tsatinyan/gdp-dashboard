import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import BayesianRidge
import streamlit as st
import yfinance as yf

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Institutional Quant Engine",
    page_icon="🔮",
    layout="wide"
)

# =========================================================
# DATA SETTINGS
# =========================================================
POPULAR_TICKERS = ["NVDA","AAPL","MSFT","AMZN","GOOGL","META","TSLA","AMD","NFLX","AVGO","JPM","XOM","LLY","UNH","SPY"]
POPULAR_BENCHMARKS = ["SPY","QQQ","IWM","DIA","TLT","GLD"]

st.title("🔮 Advanced Predictive Quant Engine & Feature Sandbox")

# =========================================================
# CORE FILTER CONTROLS
# =========================================================
current_year = datetime.datetime.now().year

c1, c2, c3 = st.columns(3)
with c1:
    ticker = st.selectbox("Asset Ticker Selection", POPULAR_TICKERS, index=POPULAR_TICKERS.index("AAPL"))
with c2:
    benchmark = st.selectbox("Benchmark Base Anchor", POPULAR_BENCHMARKS, index=0)
with c3:
    start_year, end_year = st.slider(
        "Historical Matrix Horizon",
        min_value=2000,
        max_value=current_year,
        value=(2015, current_year)
    )

FETCH_START = f"{start_year-1}-01-01"
END_DATE = f"{end_year}-12-31"

# =========================================================
# DATA LOADER
# =========================================================
@st.cache_data
def load_data(ticker_sym, bench_sym, start, end):
    t_obj = yf.Ticker(ticker_sym)
    b_obj = yf.Ticker(bench_sym)
    
    stock = t_obj.history(start=start, end=end, auto_adjust=True)
    bench = b_obj.history(start=start, end=end, auto_adjust=True)

    if stock.empty or bench.empty:
        return pd.DataFrame(), pd.DataFrame()

    if stock.index.tz is not None:
        stock.index = stock.index.tz_localize(None)
    if bench.index.tz is not None:
        bench.index = bench.index.tz_localize(None)

    return stock, bench

stock, bench = load_data(ticker, benchmark, FETCH_START, END_DATE)

if stock.empty or bench.empty:
    st.error("Data pipeline empty. Check underlying connections.")
    st.stop()

# =========================================================
# ADVANCED QUANT FEATURE ENGINEERING
# =========================================================
df = stock.copy()
df["ret_1d"] = df["Close"].pct_change()

bench_df = bench[["Close"]].copy().rename(columns={"Close": "bench_close"})
bench_df["bench_ret"] = bench_df["bench_close"].pct_change()
df = df.join(bench_df[["bench_close", "bench_ret"]], how="inner")

# Risk & Trend Basics
df["vol_20"] = df["ret_1d"].rolling(20).std()
df["sma_50"] = df["Close"].rolling(50).mean()
df["sma_200"] = df["Close"].rolling(200).mean()

# Market Regime Architecture
df["regime"] = np.where(df["sma_50"] > df["sma_200"], "BULLISH 🐂", "BEARISH 🐻")

# Bollinger Bands Calculation
df["bb_mid"] = df["Close"].rolling(20).mean()
df["bb_std"] = df["Close"].rolling(20).std()
df["bb_upper"] = df["bb_mid"] + (2 * df["bb_std"])
df["bb_lower"] = df["bb_mid"] - (2 * df["bb_std"])

# MACD Generation
df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["macd_line"] = df["ema_12"] - df["ema_26"]
df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd_line"] - df["macd_signal"]

# RSI Engineering
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / (loss + 1e-9)
df["rsi"] = 100 - (100 / (1 + rs))

df.dropna(subset=["ret_1d", "vol_20", "sma_200", "rsi", "bb_upper", "macd_hist"], inplace=True)

# Truncate to active viewport bounds
display_start = pd.to_datetime(f"{start_year}-01-01")
plot_df = df[df.index >= display_start].copy()

# =========================================================
# STRATEGY SIMULATION (RSI + CROSSOVER MIXED BACKTEST)
# =========================================================
plot_df["signal"] = np.where((plot_df["rsi"] > 50) & (plot_df["Close"] > plot_df["bb_mid"]), 1, -1)
plot_df["strategy_returns"] = plot_df["signal"] * plot_df["ret_1d"]

plot_df["equity"] = (1 + plot_df["strategy_returns"]).cumprod()
plot_df["benchmark_equity"] = (1 + plot_df["bench_ret"]).cumprod()

wins = plot_df[plot_df["strategy_returns"] > 0]
losses = plot_df[plot_df["strategy_returns"] < 0]
win_rate = len(wins) / (len(wins) + len(losses) + 1e-9)
profit_factor = plot_df["strategy_returns"][plot_df["strategy_returns"] > 0].sum() / (abs(plot_df["strategy_returns"][plot_df["strategy_returns"] < 0].sum()) + 1e-9)

# Sharpe Calculations
strat_mean = plot_df["strategy_returns"].mean()
strat_std = plot_df["strategy_returns"].std() + 1e-9
sharpe = (strat_mean / strat_std) * np.sqrt(252)
bench_sharpe = (plot_df["bench_ret"].mean() / (plot_df["bench_ret"].std() + 1e-9)) * np.sqrt(252)

# Sortino Ratio Calculation
downside_returns = plot_df["strategy_returns"].clip(upper=0)
downside_std = downside_returns.std() + 1e-9
sortino = (strat_mean / downside_std) * np.sqrt(252)

# Drawdown & Calmar Ratio
drawdown = (plot_df["equity"] / plot_df["equity"].cummax()) - 1
max_dd = drawdown.min()

# FIXED: Explicit date difference conversion avoiding Pandas Index pollution
days_delta = plot_df.index[-1] - plot_df.index[0]
total_days = int(days_delta.days)

last_equity_val = plot_df["equity"].iloc[-1]
if hasattr(last_equity_val, 'item'):
    last_equity_val = last_equity_val.item()
last_equity_val = float(last_equity_val)

annualized_return = (last_equity_val) ** (365.25 / (total_days + 1e-9)) - 1
calmar = float(annualized_return / (abs(max_dd) + 1e-9))

# Value at Risk (VaR) & Conditional Value at Risk (CVaR)
var_95_hist = np.percentile(plot_df["strategy_returns"], 5)
var_95_param = strat_mean - (1.645 * strat_std)
cvar_95 = plot_df["strategy_returns"][plot_df["strategy_returns"] <= var_95_hist].mean()

# =========================================================
# FORECAST ENGINE (BAYESIAN MODEL)
# =========================================================
model_df = plot_df.copy()
model_df["future_return"] = model_df["Close"].pct_change().shift(-1)
features = ["ret_1d", "vol_20", "rsi", "macd_hist"]
train_df = model_df.dropna(subset=features + ["future_return"]).copy()

X = train_df[features]
y = train_df["future_return"]
model = BayesianRidge()
model.fit(X, y)

latest = model_df.iloc[-1]
X_future = pd.DataFrame([latest[features].values], columns=features)
pred, std = model.predict(X_future, return_std=True)

# Clean element extraction formatting variables
current_price = float(latest["Close"])
pred_price = float(current_price * (1 + pred[0]))
low = float(pred_price - 1.96 * std[0] * current_price)
high = float(pred_price + 1.96 * std[0] * current_price)
direction = "TREND UP 🚀" if pred_price > current_price else "TREND DOWN 📉"
current_regime = latest["regime"]

# =========================================================
# INITIAL METRICS VIEWPORTS
# =========================================================
st.subheader("🏁 Live Structural Regime & Performance Analytics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current Market Regime", current_regime, help="Current market trend state derived from the 50 SMA vs 200 SMA configuration crossover index.")
m2.metric("Predicted Bias Direction", direction, help="Directional price action bias forecast calculated over the next sequence period via Bayesian architectures.")
m3.metric("Expected Next Close", f"${current_price:.2f} ➔ ${pred_price:.2f}", help="Raw projected value expectation point target mapping for the upcoming trading session close.")
m4.metric("Strategy Sharpe Ratio", f"{sharpe:.2f}", help="Annualized return generation per unit of generalized standard-deviation metric portfolio volatility.")
m5.metric("Strategy Sortino Ratio", f"{sortino:.2f}", help="Risk-adjusted baseline tracking metrics penalizing solely downside standard deviation structures.")

# Sandbox Diagnostics Display Panel Wrapper
with st.container(border=True):
    st.markdown("**🔬 Quant Sandbox Diagnostics & Confidence Matrices**")
    sub_col1, sub_col2, sub_col3, sub_col4 = st.columns(4)
    sub_col1.metric("95% CI Lower Boundary", f"${low:.2f}", help="Statistical minimum asset threshold boundary parameters under normal baseline variations (95% Confidence).")
    sub_col2.metric("95% CI Upper Boundary", f"${high:.2f}", help="Statistical maximum potential asset target constraints under standard variance models (95% Confidence).")
    sub_col3.metric("Backtest Win Rate", f"{win_rate*100:.2f}%", help="The count percentage of tracked operational trade modifications printing raw positive profit curves.")
    sub_col4.metric("System Profit Factor", f"{profit_factor:.2f}", help="Gross cumulative trade generation profits over structural gross system losses. System efficiency factor.")

# Advanced Risk Dashboard Section
st.subheader("⚠️ Tail Risk & Capital Protection Matrix (1-Day Horizon)")
r1, r2, r3, r4 = st.columns(4)
r1.metric("Historical 95% VaR", f"{var_95_hist * 100:.2f}%", help="The historical maximum expected loss at 95% confidence bounds over a single day session.")
r2.metric("Parametric 95% VaR", f"{var_95_param * 100:.2f}%", help="The theoretical maximum expected loss assuming standard normal standardizations.")
r3.metric("95% Expected Shortfall (CVaR)", f"{cvar_95 * 100:.2f}%", help="The average expected drawdown on occurrences where the base 95% VaR limits are actively broken.")
r4.metric("Calmar Risk Ratio", f"{calmar:.2f}", help="Annualized return rate divided by the maximum historical peak-to-trough account drawdown.")

# =========================================================
# MASTER SELECTION CONTROLS PANEL
# =========================================================
with st.expander("🛠️ MASTER INITIAL CHART CONTROL BOARD", expanded=True):
    st.write("Toggle any conceptual asset node to project its layout canvas onto the lower viewport.")
    cx1, cx2, cx3 = st.columns(3)
    
    with cx1:
        st.markdown("**Backtest System Layouts**")
        show_equity = st.checkbox("Compounded Equity Curves Plot", value=False)
        show_drawdown = st.checkbox("Peak-to-Trough System Drawdown Profiles", value=False)
        show_signals = st.checkbox("Trading Signals Vector Tracking (-1 vs +1)", value=False)
        
    with cx2:
        st.markdown("**Core Volatility & Price Action**")
        show_price_base = st.checkbox("Pure Price Action Close Line Chart", value=False)
        show_bb = st.checkbox("Bollinger Bands Volatility Bands Overlay", value=False)
        show_price_sig = st.checkbox("Price + Entry/Exit Action Signals Plot", value=False)

    with cx3:
        st.markdown("**Technical Indicator Features & Regimes**")
        show_regime_chart = st.checkbox("Market Regime Trend Crossover Chart (50 vs 200 SMA)", value=False)
        show_macd = st.checkbox("MACD Crossover Momentum Histogram", value=False)
        show_rsi = st.checkbox("Relative Strength Index (RSI) Oscillator", value=False)

# =========================================================
# DYNAMIC PLOTLY RENDERING CANVAS WITH ADJUSTED HEIGHTS
# =========================================================
active_plots = sum([show_equity, show_drawdown, show_signals, show_price_base, show_bb, show_price_sig, show_regime_chart, show_macd, show_rsi])

if active_plots > 0:
    fig = make_subplots(rows=active_plots, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    current_row = 1
    
    if show_price_base:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name=f"{ticker} Close"), row=current_row, col=1)
        fig.update_yaxes(title_text="Price ($)", row=current_row, col=1)
        current_row += 1
        
    if show_bb:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Close"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_upper"], name="BB Upper", line=dict(dash='dash')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_mid"], name="BB Mid", line=dict(color='gray')), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_lower"], name="BB Lower", line=dict(dash='dash')), row=current_row, col=1)
        fig.update_yaxes(title_text="Bollinger Bands", row=current_row, col=1)
        current_row += 1

    if show_regime_chart:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Close"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_50"], name="50 SMA"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_200"], name="200 SMA"), row=current_row, col=1)
        fig.update_yaxes(title_text="Regime Cross", row=current_row, col=1)
        current_row += 1

    if show_equity:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["equity"], name="Strategy Growth"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["benchmark_equity"], name="Bench Growth", line=dict(color='orange')), row=current_row, col=1)
        fig.update_yaxes(title_text="Returns Mult.", row=current_row, col=1)
        current_row += 1

    if show_drawdown:
        fig.add_trace(go.Scatter(x=plot_df.index, y=drawdown * 100, name="Drawdown %", fill='tozeroy', line=dict(color='red')), row=current_row, col=1)
        fig.update_yaxes(title_text="Drawdown %", row=current_row, col=1)
        current_row += 1

    if show_signals:
        fig.add_trace(go.Scatter(
            x=plot_df.index, 
            y=plot_df["signal"], 
            name="Signal State", 
            mode='lines',
            line=dict(color='#00D4B2', shape='hv')
        ), row=current_row, col=1)
        
        fig.update_yaxes(
            tickmode="array",
            tickvals=[-1, 1],
            ticktext=["Bearish / Cash 🐻", "Bullish / Long 🐂"],
            title_text="Strategy Allocation", 
            row=current_row, 
            col=1
        )
        current_row += 1

    if show_price_sig:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Price Line"), row=current_row, col=1)
        buys = plot_df[plot_df["signal"] == 1]
        sells = plot_df[plot_df["signal"] == -1]
        fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode='markers', name='Long Allocation', marker=dict(symbol='triangle-up', color='green', size=7)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode='markers', name='Short/Cash Allocation', marker=dict(symbol='triangle-down', color='red', size=7)), row=current_row, col=1)
        fig.update_yaxes(title_text="Signals Execution", row=current_row, col=1)
        current_row += 1

    if show_macd:
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["macd_hist"], name="MACD Hist"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd_line"], name="MACD Line"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd_signal"], name="Signal Line"), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD Matrix", row=current_row, col=1)
        current_row += 1

    if show_rsi:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi"], name="RSI", line=dict(color='purple')), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI Value", row=current_row, col=1)
        current_row += 1

    # Adaptive Height Logic
    chart_layout_height = 500 if active_plots == 1 else (300 * active_plots)
    
    fig.update_layout(height=chart_layout_height, showlegend=True, template="plotly_dark", margin=dict(t=20, b=20, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select one or more configurations from the control panel board above to render analytics panels.")
