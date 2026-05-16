import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import BayesianRidge
import streamlit as st
import yfinance as yf

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Predictive Quant Engine", page_icon="📈", layout="wide"
)

# Custom CSS to handle dynamic metric and log styling
st.markdown(
    """
    <style>
    div.block-container {padding-top:2rem; padding-bottom:2rem;}
    section[data-testid="stSidebar"] {width: 320px !important;}
    
    /* Dynamic Metric Coloration */
    div[data-testid="stMetricValue"]:contains("Quiet Bull") { color: #00CC96 !important; font-weight: bold; }
    div[data-testid="stMetricValue"]:contains("Volatile Bull") { color: #FFA726 !important; font-weight: bold; }
    div[data-testid="stMetricValue"]:contains("Bear Market") { color: #EF553B !important; font-weight: bold; }
    div[data-testid="stMetricValue"]:contains("Consolidation") { color: #B0BEC5 !important; font-weight: bold; }
    </style>
""",
    unsafe_allow_html=True,
)

POPULAR_TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "AVGO", "TSLA", "NFLX",
    "AMD", "ADBE", "CRM", "CSCO", "QCOM", "INTC", "TXN", "AMAT", "IBM", "NOW",
    "MU", "LRCX", "ORCL", "ACN", "INTU", "ADP", "ADI", "PANW", "KLAC", "SNPS",
    "CDNS", "FTNT", "MSI", "HPQ", "HPE", "CTSH", "ANSS", "RCL", "WDAY", "VRSN",
    "ANET", "GEN", "COHR", "AKAM", "ON", "BRK.B", "JPM", "BAC", "WFC", "MS",
    "GS", "C", "BLK", "AXP", "SCHW", "SPGI", "MMC", "CB", "PNC", "USB",
    "CME", "ALL", "AFL", "MET", "AMT", "PLD", "CCI", "LLY", "UNH", "JNJ",
    "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN", "BMY", "GILD", "SYK",
    "MDT", "ISRG", "VRTX", "WMT", "PG", "KO", "PEP", "COST", "MCD", "HD",
    "NKE", "LOW", "DIS", "CMCSA", "XOM", "CVX", "UNP", "BA", "GE", "HON"
]
POPULAR_BENCHMARKS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "GLD"]

if "ticker_choice" not in st.session_state:
    st.session_state.ticker_choice = "AAPL"
if "benchmark_choice" not in st.session_state:
    st.session_state.benchmark_choice = "SPY"

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Configuration")

if st.sidebar.button("🧹 Reset Fields", use_container_width=True):
    st.session_state.ticker_choice = "AAPL"
    st.session_state.benchmark_choice = "SPY"
    st.rerun()

st.sidebar.markdown("---")

ticker = st.sidebar.selectbox("Select Ticker Symbol", options=POPULAR_TICKERS, index=POPULAR_TICKERS.index(st.session_state.ticker_choice)).upper().strip()
benchmark = st.sidebar.selectbox("Select Benchmark Symbol", options=POPULAR_BENCHMARKS, index=POPULAR_BENCHMARKS.index(st.session_state.benchmark_choice)).upper().strip()

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Date Range Selection")

current_year = datetime.datetime.now().year
start_year, end_year = st.sidebar.slider("Which years are you interested in?", min_value=2000, max_value=current_year, value=(2015, current_year), step=1)

FETCH_START = f"{start_year - 1}-01-01"
START_DATE = f"{start_year}-01-01"
END_DATE = f"{end_year}-12-31"

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Choose Indicators to Display")
show_regime = st.sidebar.checkbox("⚠️ Market Regime Detection Panel", value=True)
show_price = st.sidebar.checkbox("Price & Bollinger Bands (With ML Tracking)", value=True)
show_drawdown = st.sidebar.checkbox("Drawdown Matrix & Z-Score", value=False)
show_rsi = st.sidebar.checkbox("Relative Strength Index (RSI)", value=False)

# ==========================================
# DATA CORE ENGINE
# ==========================================
@st.cache_data(show_spinner="Fetching market data from Yahoo Finance...")
def load_and_clean_data(ticker, benchmark, start_date, end_date):
    stock_raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
    bench_raw = yf.download(benchmark, start=start_date, end=end_date, progress=False)
    if stock_raw.empty or bench_raw.empty: return pd.DataFrame(), pd.DataFrame()
    if isinstance(stock_raw.columns, pd.MultiIndex): stock_raw.columns = stock_raw.columns.get_level_values(0)
    if isinstance(bench_raw.columns, pd.MultiIndex): bench_raw.columns = bench_raw.columns.get_level_values(0)
    stock_raw.dropna(subset=["Close"], inplace=True)
    bench_raw.dropna(subset=["Close"], inplace=True)
    return stock_raw, bench_raw

def create_features(stock, benchmark_df, final_start_date):
    if stock.empty or benchmark_df.empty: return pd.DataFrame()
    df = stock.copy()
    df["ret_1d"] = df["Close"].pct_change()
    benchmark_df["spy_ret_1d"] = benchmark_df["Close"].pct_change()
    df = df.reset_index()
    df = pd.merge(df, benchmark_df.reset_index()[["Date", "spy_ret_1d"]], on="Date", how="inner").set_index("Date")

    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["drawdown"] = (df["Close"] / df["Close"].cummax()) - 1
    df["sma_20"] = df["Close"].rolling(20).mean()
    df["sma_50"] = df["Close"].rolling(50).mean()
    df["sma_200"] = df["Close"].rolling(200).mean()
    std_20 = df["Close"].rolling(20).std()
    df["bb_upper"] = df["sma_20"] + (2 * std_20)
    df["bb_lower"] = df["sma_20"] - (2 * std_20)

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + (gain / (loss + 1e-6))))

    df["vol_rolling_median"] = df["vol_20"].rolling(252).median()
    cond_bull = (df["Close"] > df["sma_200"]) & (df["vol_20"] <= df["vol_rolling_median"])
    cond_volatile_bull = (df["Close"] > df["sma_200"]) & (df["vol_20"] > df["vol_rolling_median"])
    cond_bear = (df["Close"] <= df["sma_200"]) & (df["vol_20"] > df["vol_rolling_median"])
    
    df["regime"] = "Consolidation"
    df.loc[cond_bull, "regime"] = "Quiet Bull"
    df.loc[cond_volatile_bull, "regime"] = "Volatile Bull"
    df.loc[cond_bear, "regime"] = "Bear Market"

    return df[df.index >= pd.to_datetime(final_start_date)].dropna()

# ==========================================
# MACHINE LEARNING & BACKTEST ENGINE
# ==========================================
def run_predictive_engine(df):
    predict_df = df[["Close", "ret_1d", "vol_20", "rsi"]].copy()
    predict_df["lag_1"] = predict_df["Close"].shift(1)
    predict_df["lag_2"] = predict_df["Close"].shift(2)
    predict_df["ret_lag"] = predict_df["ret_1d"].shift(1)
    predict_df.dropna(inplace=True)

    X = predict_df[["lag_1", "lag_2", "ret_lag", "vol_20", "rsi"]].values
    y = predict_df["Close"].values

    model = BayesianRidge()
    model.fit(X, y)

    # Next-Day Projections
    latest_row = df.iloc[-1]
    X_tomorrow = np.array([[latest_row["Close"], predict_df["lag_1"].iloc[-1], latest_row["ret_1d"], latest_row["vol_20"], latest_row["rsi"]]])
    pred_price, pred_std = model.predict(X_tomorrow, return_std=True)
    
    current_price = latest_row["Close"]
    predicted_value = pred_price[0]
    std_error = pred_std[0]

    expected_return = (predicted_value - current_price) / current_price
    direction = "UP 📈" if expected_return >= 0 else "DOWN 📉"
    
    z_score = (predicted_value - current_price) / (std_error + 1e-6)
    direction_confidence = min(99.9, max(50.0, 50 + (abs(z_score) * 15)))

    lower_95 = predicted_value - (1.96 * std_error)
    upper_95 = predicted_value + (1.96 * std_error)

    # Historical Rolling Backtest Strategy (Walk-Forward Validation)
    window = min(63, len(predict_df) - 10) # 63 trading days = 3 months
    correct_predictions = 0
    total_trades = 0
    
    # Track historical prediction coordinates to trace on the line chart
    history_dates = []
    history_preds = []

    for i in range(len(predict_df) - window, len(predict_df)):
        X_train = X[:i]
        y_train = y[:i]
        X_test = X[i].reshape(1, -1)
        
        bt_model = BayesianRidge()
        bt_model.fit(X_train, y_train)
        
        bt_pred = bt_model.predict(X_test)[0]
        prev_close = X_test[0][0]  # lag_1 close
        actual_close = y[i]
        
        pred_dir = 1 if (bt_pred >= prev_close) else 0
        actual_dir = 1 if (actual_close >= prev_close) else 0
        
        if pred_dir == actual_dir:
            correct_predictions += 1
        total_trades += 1
        
        # Log timeline coordinates
        history_dates.append(predict_df.index[i])
        history_preds.append(bt_pred)

    accuracy = (correct_predictions / total_trades) * 100 if total_trades > 0 else 0.0

    return {
        "current": current_price, "pred": predicted_value, "dir": direction,
        "conf": direction_confidence, "low": lower_95, "high": upper_95,
        "accuracy": accuracy, "total_days": total_trades,
        "history_dates": history_dates, "history_preds": history_preds
    }

# ==========================================
# MAIN INTERFACE DISPLAY
# ==========================================
st.title("📊 Quantitative Dashboard & ML Predictive Engine")

stock_data, bench_data = load_and_clean_data(ticker, benchmark, FETCH_START, END_DATE)

if stock_data.empty or bench_data.empty:
    st.error("Execution Error. Missing or dropped connection array data.")
else:
    quant_df = create_features(stock_data, bench_data, START_DATE)

    if quant_df.empty:
        st.warning("Insufficient structural data profile matrix timeline depth.")
    else:
        latest = quant_df.iloc[-1]
        ml_results = run_predictive_engine(quant_df)

        regime_meta = {
            "Quiet Bull": {"label": "Quiet Bull (Accumulation)", "prefix": "🟢"},
            "Volatile Bull": {"label": "Volatile Bull (Distribution)", "prefix": "🟡"},
            "Bear Market": {"label": "Bear Market (Panic Sells)", "prefix": "🔴"},
            "Consolidation": {"label": "Sideways / Consolidation", "prefix": "⚪"}
        }
        
        current_regime_str = latest["regime"]
        meta = regime_meta.get(current_regime_str, {"label": current_regime_str, "prefix": "🔷"})
        formatted_regime_metric = f"{meta['prefix']} {meta['label']}"

        # Metrics Header
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{ticker} Last Close", f"${latest['Close']:.2f}", f"{latest['ret_1d']*100:+.2f}%")
        m2.metric("Market Phase State", formatted_regime_metric)
        m3.metric("ML Target Direction (1D)", ml_results["dir"], f"Conf: {ml_results['conf']:.1f}%")
        m4.metric("95% Price Bound Interval", f"${ml_results['low']:.2f} - ${ml_results['high']:.2f}")

        st.markdown("---")

        # ==========================================
        # ADVANCED PREDICTIVE DISPLAY CONTAINER BLOCK
        # ==========================================
        st.subheader("🔮 24-Hour Predictive Probability Distribution")
        p1, p2, p3 = st.columns(3)
        p1.markdown(f"**Expected Target Price:** `${ml_results['pred']:.2f}`")
        p2.markdown(f"**Bearish Risk Boundary (95% floor):** `${ml_results['low']:.2f}`")
        p3.markdown(f"**Bullish Target Boundary (95% cap):** `${ml_results['high']:.2f}`")

        # ==========================================
        # HISTORICAL ACCURACY BACKTEST LOGS
        # ==========================================
        st.subheader("📋 Walk-Forward Historical Backtest Summary")
        
        acc = ml_results["accuracy"]
        acc_color = "#00CC96" if acc >= 53.0 else ("#FFA726" if acc >= 49.0 else "#EF553B")
        
        b1, b2, b3 = st.columns(3)
        b1.markdown(f"**Tested Window Depth:** `{ml_results['total_days']} Trading Days`")
        b2.markdown(f"**Directional Hit Rate (Accuracy):** <span style='color:{acc_color}; font-weight:bold;'>{acc:.2f}%</span>", unsafe_allow_html=True)
        b3.markdown(f"**Model Baseline Alpha Edge vs Chance:** `{acc - 50.0:+.2f}%`")
        
        st.markdown("---")

        # Dynamic Subplot Matrix Core Engine
        active_rows = []
        subplot_titles = []
        row_heights = []

        if show_regime:
            active_rows.append("regime"); subplot_titles.append("⚠️ Market Regime State Window"); row_heights.append(0.18)
        if show_price:
            active_rows.append("price"); subplot_titles.append(f"📈 {ticker} Price Channel & ML Historical Prediction Tracking"); row_heights.append(0.35)
        if show_drawdown:
            active_rows.append("drawdown"); subplot_titles.append("📉 Peak System Drawdowns"); row_heights.append(0.15)
        if show_rsi:
            active_rows.append("rsi"); subplot_titles.append("⚡ Momentum Trajectory (RSI 14)"); row_heights.append(0.15)

        if active_rows:
            total_height = sum(row_heights)
            row_heights = [h / total_height for h in row_heights]

            fig = make_subplots(rows=len(active_rows), cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights, subplot_titles=tuple(subplot_titles))
            current_row = 1

            # Regime Chart
            if "regime" in active_rows:
                regime_map = {"Quiet Bull": 3, "Volatile Bull": 2, "Consolidation": 1, "Bear Market": 0}
                fig.add_trace(go.Scatter(x=quant_df.index, y=quant_df["regime"].map(regime_map), mode="lines", line=dict(color="#B0BEC5", width=1.5), name="Regime Trace"), row=current_row, col=1)
                fig.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["Bear (🔴)", "Sideways (⚪)", "Vol Bull (🟡)", "Quiet Bull (🟢)"], row=current_row, col=1)
                current_row += 1

            # Price & ML Backtest Tracker Line
            if "price" in active_rows:
                fig.add_trace(go.Scatter(x=quant_df.index, y=quant_df["bb_upper"], name="BB Upper", line=dict(color="rgba(173,181,189,0.2)", width=1)), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=quant_df.index, y=quant_df["bb_lower"], name="BB Lower", line=dict(color="rgba(173,181,189,0.2)", width=1), fill="tonexty", fillcolor="rgba(173,181,189,0.02)"), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=quant_df.index, y=quant_df["Close"], name="Actual Close Price", line=dict(color="#55CCFF", width=2.5)), row=current_row, col=1)
                fig.add_trace(go.Scatter(x=quant_df.index, y=quant_df["sma_200"], name="SMA 200", line=dict(color="#FF5252", width=1.2, dash="dot")), row=current_row, col=1)
                
                # NEW PLOT ELEMENT: Historical walk-forward prediction tracking line overlay
                fig.add_trace(go.Scatter(
                    x=ml_results["history_dates"], 
                    y=ml_results["history_preds"], 
                    name="ML Historical Prediction Tracker", 
                    line=dict(color="#E040FB", width=1.5, dash="dash")
                ), row=current_row, col=1)
                
                current_row += 1

            # Drawdown Chart
            if "drawdown" in active_rows:
                fig.add_trace(go.Scatter(x=quant_df.index, y=quant_df["drawdown"] * 100, name="Drawdown %", fill="tozeroy", line=dict(color="#EF553B", width=1), fillcolor="rgba(239,85,59,0.12)"), row=current_row, col=1)
                current_row += 1

            # RSI Chart
            if "rsi" in active_rows:
                fig.add_trace(go.Scatter(x=quant_df.index, y=quant_df["rsi"], name="RSI", line=dict(color="#00CC96", width=1.5)), row=current_row, col=1)
                fig.add_hline(y=70, row=current_row, col=1, line_dash="dash", line_color="rgba(239,85,59,0.4)")
                fig.add_hline(y=30, row=current_row, col=1, line_dash="dash", line_color="rgba(0,204,150,0.4)")

            calculated_height = max(400, len(active_rows) * 250)
            fig.update_layout(template="plotly_dark", height=calculated_height, hovermode="x unified", margin=dict(l=15, r=15, t=50, b=15))
            st.plotly_chart(fig, use_container_width=True)
