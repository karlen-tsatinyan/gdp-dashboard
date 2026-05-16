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

c1, c2, c3 = st.columns([1,1,2])
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
sharpe = (plot_df["strategy_returns"].mean() / (plot_df["strategy_returns"].std() + 1e-9)) * np.sqrt(252)
bench_sharpe = (plot_df["bench_ret"].mean() / (plot_df["bench_ret"].std() + 1e-9)) * np.sqrt(252)

drawdown = (plot_df["equity"] / plot_df["equity"].cummax()) - 1
max_dd = drawdown.min()

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

current_price = float(latest["Close"])
pred_price = current_price * (1 + pred[0])
low = pred_price - 1.96 * std[0] * current_price
high = pred_price + 1.96 * std[0] * current_price
direction = "TREND UP 🚀" if pred_price > current_price else "TREND DOWN 📉"
current_regime = latest["regime"]

# =========================================================
# INITIAL METRICS VIEWPORTS
# =========================================================
st.subheader("🏁 Live Structural Regime & Analytics Forecast")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current Market Regime", current_regime)
m2.metric("Predicted Bias Direction", direction)
m3.metric("Expected Next Close", f"${current_price:.2f} ➔ ${pred_price:.2f}")
m4.metric("Strategy Sharpe Ratio", f"{sharpe:.2f}")
m5.metric(f"{benchmark} Benchmark Sharpe", f"{bench_sharpe:.2f}")

st.info(f"""
🔒 **Statistical Confidence Spectrum (95% CI):** ${low:.2f} ———→ ${high:.2f}  |  
🎯 **Backtest Win Rate Metric:** {win_rate*100:.2f}%  |  
📊 **System Profit Factor:** {profit_factor:.2f}  |  
🩸 **Historical Peak Account Drawdown:** {max_dd*100:.2f}%
""")

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
        show_macd = st.checkbox("MACD Crossover Momentum Grid", value=False)
        show_rsi = st.checkbox("RSI Overbought/Oversold Oscillator", value=False)
        show_risk = st.checkbox("20-Day Rolling Risk Indicators (`vol_20`)", value=False)

# =========================================================
# UNIVERSAL PLOTLY STYLING HELPER
# =========================================================
def apply_clean_layout(fig, height=400, y_title=None, legend_title="Variables"):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=60, r=30, t=40, b=40),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
        showlegend=True,
        legend=dict(
            title_text=legend_title,
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(17,17,17,0.9)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(size=11, color="#FFFFFF")
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, title_text=y_title)

# =========================================================
# GRAPH CANVAS ARCHITECTURE RENDERS
# =========================================================

# 1. Equity Curves Plot
if show_equity:
    st.subheader("📈 Strategy Compounded Equity vs Benchmark Baseline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["equity"], name="Quant Strategy Returns Curve", line=dict(color="#00E676", width=2.5)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["benchmark_equity"], name=f"Passive Hold Matrix ({benchmark})", line=dict(color="#FFFFFF", dash="dash", width=1.5)))
    apply_clean_layout(fig, height=450, y_title="Growth Multiple", legend_title="Account Portfolios")
    st.plotly_chart(fig, width='stretch')

# 2. Peak-to-Trough Drawdown
if show_drawdown:
    st.subheader("🩸 Strategy Drawdown Profile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=drawdown * 100, fill="tozeroy", name="Strategy Historical Drawdown %", line=dict(color="#FF1744", width=1.5), fillcolor="rgba(255, 23, 68, 0.15)"))
    apply_clean_layout(fig, height=220, y_title="% Drop from Peak", legend_title="Risk Vector Metrics")
    st.plotly_chart(fig, width='stretch')

# 3. Signals Vector
if show_signals:
    st.subheader("⚡ Trading Signals Allocation States")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["signal"], line=dict(color="#FF9100", shape="hv", width=2), name="System Strategy Execution State (-1 vs +1)"))
    apply_clean_layout(fig, height=180, legend_title="System Outputs")
    fig.update_yaxes(tickvals=[-1, 1], ticktext=["Short (-1)", "Long (+1)"])
    st.plotly_chart(fig, width='stretch')

# 4. Pure Price Line
if show_price_base:
    st.subheader(f"📊 Raw {ticker} Price Close Topology")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name=f"Raw {ticker} Historical Closing Value", line=dict(color="#00E5FF", width=2)))
    apply_clean_layout(fig, height=350, y_title="Price ($)", legend_title="Asset Trackers")
    st.plotly_chart(fig, width='stretch')

# 5. Bollinger Bands Pricing
if show_bb:
    st.subheader(f"🛡️ Bollinger Bands Channels vs Basis Mid-Line")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_upper"], name="+2 StdDev Upper Volatility Boundary Band", line=dict(color="rgba(0, 229, 255, 0.35)", dash="dot")))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_lower"], name="-2 StdDev Lower Volatility Boundary Band", line=dict(color="rgba(0, 229, 255, 0.35)", dash="dot"), fill="tonexty", fillcolor="rgba(0, 229, 255, 0.07)"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name=f"{ticker} Underlying Spot Asset Close", line=dict(color="#FFFFFF", width=2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["bb_mid"], name="Basis Average Moving Channel (20 SMA)", line=dict(color="#FFD700", width=1.5)))
    apply_clean_layout(fig, height=450, y_title="Price ($)", legend_title="Bollinger Envelopes")
    st.plotly_chart(fig, width='stretch')

# 6. Price + Signal Multiplot Overlay
if show_price_sig:
    st.subheader(f"🎯 Integrated Price Action + Signal Entry Overlay Grid")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Underlying Raw Spot Asset Value Line", line=dict(color="#78909C", width=1.5)))
    
    longs = plot_df[plot_df["signal"] == 1]
    shorts = plot_df[plot_df["signal"] == -1]
    
    fig.add_trace(go.Scatter(x=longs.index, y=longs["Close"], mode="markers", name="Long Strategy Entry Allocations", marker=dict(color="#00E676", symbol="triangle-up", size=9, line=dict(color="#111111", width=1))))
    fig.add_trace(go.Scatter(x=shorts.index, y=shorts["Close"], mode="markers", name="Short Strategy Entry / Flip Actions", marker=dict(color="#FF1744", symbol="triangle-down", size=9, line=dict(color="#111111", width=1))))
    apply_clean_layout(fig, height=450, y_title="Price ($)", legend_title="Execution Layers")
    st.plotly_chart(fig, width='stretch')

# 7. Market Regime Crossover Chart
if show_regime_chart:
    st.subheader("🦁 Market Regime Trend Crossover Matrix (50 vs 200 SMA)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Close"], name="Asset Reference Underlier Spot Index", line=dict(color="#FFFFFF", width=1.5)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_50"], name="Fast Crossover Filter Vector (50 SMA)", line=dict(color="#00E5FF", width=2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["sma_200"], name="Slow Structural Trend Anchor Line (200 SMA)", line=dict(color="#E040FB", width=2.5)))
    
    regime_changes = plot_df["sma_50"] > plot_df["sma_200"]
    diff_blocks = regime_changes.astype(int).diff().fillna(0)
    switch_points = plot_df[diff_blocks != 0].index.tolist()
    boundaries = [plot_df.index[0]] + switch_points + [plot_df.index[-1]]
    
    # FIX: Used solid opaque colors for the legend identifiers so they display sharply on the dark background
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", color="#00E676", size=10), name="Structural Bull Zone Matrix State"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(symbol="square", color="#FF1744", size=10), name="Structural Bear Zone Matrix State"))

    for i in range(len(boundaries) - 1):
        mid_idx = plot_df.index[plot_df.index >= boundaries[i]][0]
        is_bull = plot_df.loc[mid_idx, "sma_50"] > plot_df.loc[mid_idx, "sma_200"]
        bg_color = "rgba(0, 230, 118, 0.05)" if is_bull else "rgba(255, 23, 68, 0.05)"
        
        fig.add_vrect(
            x0=boundaries[i], x1=boundaries[i+1],
            fillcolor=bg_color, opacity=1,
            layer="below", line_width=0
        )
        
    apply_clean_layout(fig, height=450, y_title="Price ($)", legend_title="Macro Regimes")
    st.plotly_chart(fig, width='stretch')

# 8. MACD Layout Engine
if show_macd:
    st.subheader("📉 Moving Average Convergence Divergence Momentum Grid")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd_line"], name="Fast Core Momentum MACD Vector (12/26)", line=dict(color="#00E5FF", width=2)))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["macd_signal"], name="Slow Baseline Signal Vector Line (9)", line=dict(color="#E040FB", width=1.5)))
    
    hist_colors = np.where(plot_df["macd_hist"] >= 0, "#00C853", "#D50000")
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["macd_hist"], name="Net Acceleration Spread Histogram", marker_color=hist_colors, opacity=0.6))
    apply_clean_layout(fig, height=280, legend_title="MACD Outputs")
    st.plotly_chart(fig, width='stretch')

# 9. RSI Profile
if show_rsi:
    st.subheader("⚡ Relative Strength Index Momentum Oscillator Profile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["rsi"], name="Calculated Relative Strength Index Vector", line=dict(color="#00E5FF", width=2)))
    apply_clean_layout(fig, height=250, legend_title="Oscillator Curves")
    
    fig.add_hline(y=70, line_dash="dash", line_color="#FF1744", opacity=0.7, annotation_text="Overbought Cutoff (70)", annotation_position="top left")
    fig.add_hline(y=50, line_dash="dot", line_color="#78909C", opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="#00E676", opacity=0.7, annotation_text="Oversold Cutoff (30)", annotation_position="bottom left")
    fig.update_yaxes(range=[10, 90])
    st.plotly_chart(fig, width='stretch')

# 10. Risk Volatility Variance Tracker
if show_risk:
    st.subheader("📊 20-Day Rolling Historic Volatility Footprint")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vol_20"] * 100, name="20-Day Statistical Pricing Sigma Deviation Vector", line=dict(color="#E040FB", width=2)))
    apply_clean_layout(fig, height=250, y_title="Volatility %", legend_title="Risk Metrics")
    st.plotly_chart(fig, width='stretch')