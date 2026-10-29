"""
NIFTY 50 Pattern Lab — educational market dashboard
Run:
    pip install streamlit yfinance pandas numpy plotly
    streamlit run nifty_pattern_lab.py

This app is for learning and paper analysis only. It does not place trades.
Its pattern score is a simple heuristic, NOT a calibrated or validated forecast.
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="NIFTY 50 Pattern Lab", page_icon="📈", layout="wide")

st.title("📈 NIFTY 50 Pattern Lab")
st.caption("Educational market dashboard • historical data may be delayed • no trades are placed")

with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox(
        "Market",
        options=["^NSEI", "^NSEBANK", "RELIANCE.NS", "TCS.NS", "INFY.NS"],
        format_func=lambda x: {
            "^NSEI": "NIFTY 50",
            "^NSEBANK": "NIFTY BANK",
            "RELIANCE.NS": "Reliance Industries",
            "TCS.NS": "TCS",
            "INFY.NS": "Infosys",
        }[x],
    )
    period = st.selectbox("History", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Candle interval", ["1d", "1wk"], index=0)
    st.info("This is a learning / paper-analysis tool, not investment advice.")

@st.cache_data(ttl=300)
def load_data(symbol, selected_period, selected_interval):
    data = yf.download(
        symbol,
        period=selected_period,
        interval=selected_interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data = data.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required):
        return pd.DataFrame()
    return data[required].dropna()

def add_indicators(df):
    d = df.copy()
    close = d["Close"]
    d["SMA20"] = close.rolling(20).mean()
    d["SMA50"] = close.rolling(50).mean()
    d["EMA12"] = close.ewm(span=12, adjust=False).mean()
    d["EMA26"] = close.ewm(span=26, adjust=False).mean()
    d["MACD"] = d["EMA12"] - d["EMA26"]
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["RSI"] = 100 - (100 / (1 + rs))
    d["RSI"] = d["RSI"].fillna(50)
    d["Return"] = close.pct_change()
    d["Volatility20"] = d["Return"].rolling(20).std() * np.sqrt(252) * 100
    return d

def detect_patterns(d):
    """Simple, transparent candle patterns on the latest completed candle."""
    if len(d) < 3:
        return ["Not enough candles to check patterns."]
    last = d.iloc[-1]
    prev = d.iloc[-2]
    body = abs(last["Close"] - last["Open"])
    candle_range = max(last["High"] - last["Low"], 1e-9)
    upper_wick = last["High"] - max(last["Open"], last["Close"])
    lower_wick = min(last["Open"], last["Close"]) - last["Low"]
    patterns = []
    if body / candle_range < 0.10:
        patterns.append("Doji-like candle (indecision)")
    if lower_wick > 2 * max(body, 1e-9) and upper_wick < 1.2 * max(body, 1e-9):
        patterns.append("Hammer-like candle (possible bullish reversal; context matters)")
    if upper_wick > 2 * max(body, 1e-9) and lower_wick < 1.2 * max(body, 1e-9):
        patterns.append("Shooting-star-like candle (possible bearish reversal; context matters)")
    prev_bear = prev["Close"] < prev["Open"]
    prev_bull = prev["Close"] > prev["Open"]
    last_bull = last["Close"] > last["Open"]
    last_bear = last["Close"] < last["Open"]
    if prev_bear and last_bull and last["Open"] <= prev["Close"] and last["Close"] >= prev["Open"]:
        patterns.append("Bullish-engulfing-like candle")
    if prev_bull and last_bear and last["Open"] >= prev["Close"] and last["Close"] <= prev["Open"]:
        patterns.append("Bearish-engulfing-like candle")
    if not patterns:
        patterns.append("No basic candle pattern detected")
    return patterns

def estimate_direction(d):
    """
    Transparent heuristic score mapped to a probability-like number.
    This is NOT a statistically calibrated probability or a validated model.
    """
    last = d.iloc[-1]
    score = 0.0
    reasons = []
    if pd.notna(last["SMA20"]) and last["Close"] > last["SMA20"]:
        score += 1
        reasons.append("Close above 20-period average")
    elif pd.notna(last["SMA20"]):
        score -= 1
        reasons.append("Close below 20-period average")

    if pd.notna(last["SMA50"]) and last["Close"] > last["SMA50"]:
        score += 1
        reasons.append("Close above 50-period average")
    elif pd.notna(last["SMA50"]):
        score -= 1
        reasons.append("Close below 50-period average")

    if last["MACD"] > last["MACD_Signal"]:
        score += 1
        reasons.append("MACD above signal line")
    else:
        score -= 1
        reasons.append("MACD below signal line")

    if last["RSI"] >= 55:
        score += 0.5
        reasons.append("RSI momentum is positive")
    elif last["RSI"] <= 45:
        score -= 0.5
        reasons.append("RSI momentum is negative")
    else:
        reasons.append("RSI is in a middle range")

    # A deliberately conservative display score, capped away from certainty.
    probability_up = float(np.clip(50 + score * 7, 22, 78))
    if probability_up >= 57:
        label = "BULLISH BIAS (paper signal)"
    elif probability_up <= 43:
        label = "BEARISH BIAS (paper signal)"
    else:
        label = "NEUTRAL / WAIT"
    return probability_up, label, reasons, score

with st.spinner("Loading market history..."):
    raw = load_data(ticker, period, interval)

if raw.empty or len(raw) < 30:
    st.error("Market data could not be loaded or there are too few candles. Try another market or history range.")
    st.stop()

df = add_indicators(raw)
latest = df.iloc[-1]
prev_close = df["Close"].iloc[-2]
change = latest["Close"] - prev_close
change_pct = change / prev_close * 100
prob_up, signal, reasons, score = estimate_direction(df)
patterns = detect_patterns(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"{latest['Close']:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
c2.metric("RSI (14)", f"{latest['RSI']:.1f}")
c3.metric("Heuristic up-score", f"{prob_up:.0f}%", help="Rule-based score, not a calibrated probability.")
c4.metric("Paper signal", signal)

st.warning(
    "Important: the percentage above is a rule-based indicator score, not a proven chance of profit. "
    "It can be wrong, especially during news, gaps, and volatile markets. No real buy/sell orders are placed."
)

left, right = st.columns([2, 1])
with left:
    st.subheader("Price chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Candles"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA 20", mode="lines"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA 50", mode="lines"))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Momentum indicators")
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("RSI (14)", "MACD"))
    fig2.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"), row=1, col=1)
    fig2.add_hline(y=70, line_dash="dash", row=1, col=1)
    fig2.add_hline(y=30, line_dash="dash", row=1, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"), row=2, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"), row=2, col=1)
    fig2.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader("Pattern check")
    for pattern in patterns:
        st.write("• " + pattern)
    st.divider()
    st.subheader("Why this score?")
    for reason in reasons:
        st.write("• " + reason)
    st.caption(f"Internal rule score: {score:+.1f}. Signals use only these simple indicators.")
    st.subheader("Recent candles")
    view = df[["Open", "High", "Low", "Close", "Volume"]].tail(8).sort_index(ascending=False)
    st.dataframe(view.style.format({
        "Open": "{:,.2f}", "High": "{:,.2f}", "Low": "{:,.2f}",
        "Close": "{:,.2f}", "Volume": "{:,.0f}"
    }), use_container_width=True)

st.subheader("How to use this safely")
st.markdown(
    """
    - Treat **bullish / bearish / neutral** as educational labels, not instructions to trade.
    - Check the date and data source; free data can be delayed or incomplete.
    - Try different historical ranges and compare the signal with what happened next.
    - Do not assume a high displayed score means a high chance of profit. Proper probability estimates need
      out-of-sample testing, transaction costs, and calibration.
    """
)
st.caption("Data fetched via Yahoo Finance's public endpoints through yfinance; availability and terms may change.")
