import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta

st.set_page_config(page_title="Advanced Market Trend Predictor", page_icon="📈", layout="wide")

st.title("📈 Indian Market Trend Predictor (Pro + Probability)")
st.markdown("### Stocks | MCX | Nifty 50 | Sensex | RSI + MACD + EMA + Probability Score")

# ====================== SYMBOL MAPPING ======================
name_to_symbol = {
    "NIFTY": "^NSEI", "NIFTY 50": "^NSEI", "NIFTY50": "^NSEI", "NSEI": "^NSEI",
    "SENSEX": "^BSESN", "BSE SENSEX": "^BSESN", "BSESN": "^BSESN",
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "INFY": "INFY.NS", "INFOSYS": "INFY.NS",
    "SBIN": "SBIN.NS", "SBI": "SBIN.NS", "HDFCBANK": "HDFCBANK.NS", "HDFC BANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS", "ITC": "ITC.NS", "WIPRO": "WIPRO.NS", "LT": "LT.NS",
    "BAJFINANCE": "BAJFINANCE.NS", "MARUTI": "MARUTI.NS", "TATAMOTORS": "TATAMOTORS.NS",
    "AXISBANK": "AXISBANK.NS", "BHARTIARTL": "BHARTIARTL.NS", "AIRTEL": "BHARTIARTL.NS",
    "GOLD": "GC=F", "SILVER": "SI=F", "CRUDE": "CL=F", "CRUDE OIL": "CL=F", "CRUDEOIL": "CL=F",
    "NATURAL GAS": "NG=F", "NATGAS": "NG=F", "COPPER": "HG=F",
}

# ====================== INDICATOR FUNCTIONS ======================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

# ====================== UI ======================
user_input = st.text_input("Enter Name (Example: Nifty 50, Sensex, Gold, Reliance, TCS)", value="Nifty 50")
user_input_clean = user_input.upper().strip()

if st.button("Get Future Trend", type="primary") or user_input:
    try:
        if user_input_clean in name_to_symbol:
            ticker = name_to_symbol[user_input_clean]
        elif user_input_clean.endswith((".NS", ".BO", "=F")) or user_input_clean.startswith("^"):
            ticker = user_input_clean
        else:
            ticker = user_input_clean + ".NS"

        with st.spinner(f"Analyzing {user_input}..."):
            data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if data.empty or "Close" not in data.columns:
            st.error(f"❌ Could not find data for **{user_input}**")
            st.info("Try: Nifty 50, Sensex, Gold, Silver, Crude Oil, Reliance, TCS")
        else:
            data = data.dropna()
            close = data["Close"]
            current_price = float(close.iloc[-1])

            # ========== CALCULATE INDICATORS ==========
            data["RSI"] = calculate_rsi(close)
            data["EMA20"] = calculate_ema(close, 20)
            data["EMA50"] = calculate_ema(close, 50)
            data["EMA200"] = calculate_ema(close, 200)
            data["MACD"], data["MACD_Signal"] = calculate_macd(close)

            rsi = float(data["RSI"].iloc[-1])
            ema20 = float(data["EMA20"].iloc[-1])
            ema50 = float(data["EMA50"].iloc[-1])
            ema200 = float(data["EMA200"].iloc[-1])
            macd = float(data["MACD"].iloc[-1])
            macd_signal = float(data["MACD_Signal"].iloc[-1])

            # ========== SCORING SYSTEM ==========
            bullish_score = 0
            bearish_score = 0

            # RSI
            if rsi < 35:
                bullish_score += 2
            elif rsi < 45:
                bullish_score += 1
            elif rsi > 65:
                bearish_score += 2
            elif rsi > 55:
                bearish_score += 1

            # MACD
            if macd > macd_signal:
                bullish_score += 2
            else:
                bearish_score += 2

            # EMA Alignment
            if current_price > ema20 > ema50:
                bullish_score += 2
            elif current_price < ema20 < ema50:
                bearish_score += 2

            if current_price > ema200:
                bullish_score += 1
            else:
                bearish_score += 1

            total = bullish_score + bearish_score
            if total == 0:
                total = 1

            bullish_prob = int((bullish_score / total) * 100)
            bearish_prob = 100 - bullish_prob

            # Final Signal
            if bullish_prob >= 70:
                final_signal = "🟢 STRONG BUY"
            elif bullish_prob >= 55:
                final_signal = "🟡 BUY"
            elif bearish_prob >= 70:
                final_signal = "🔴 STRONG SELL"
            elif bearish_prob >= 55:
                final_signal = "🟠 SELL"
            else:
                final_signal = "⚪ WAIT / SIDEWAYS"

            # ========== DISPLAY ==========
            if ticker == "^NSEI":
                st.success("✅ Analyzing **Nifty 50**")
            elif ticker == "^BSESN":
                st.success("✅ Analyzing **Sensex**")
            elif ticker.endswith("=F"):
                st.success(f"✅ International Futures for **{user_input}**")
            else:
                st.success(f"✅ Found: **{ticker}**")

            # Main Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"{current_price:.2f}")
            col2.metric("Final Signal", final_signal)
            col3.metric("Bullish Probability", f"{bullish_prob}%")
            col4.metric("Bearish Probability", f"{bearish_prob}%")

            # Progress bar for probability
            st.progress(bullish_prob / 100)
            st.caption(f"Bullish Confidence: {bullish_prob}%  |  Bearish Confidence: {bearish_prob}%")

            # Indicators
            st.subheader("📊 Technical Indicators")
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("RSI (14)", f"{rsi:.1f}", "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral"))
            ic2.metric("MACD", f"{macd:.2f}", "Bullish" if macd > macd_signal else "Bearish")
            ic3.metric("EMA 20", f"{ema20:.2f}")
            ic4.metric("EMA 50", f"{ema50:.2f}")
            st.caption(f"EMA 200: {ema200:.2f} → Price is {'Above' if current_price > ema200 else 'Below'} EMA 200")

            # Support / Resistance + Risk
            st.subheader("🛡️ Support, Resistance & Risk Management")

            high = data["High"].iloc[-1]
            low = data["Low"].iloc[-1]
            pivot = (high + low + current_price) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high

            if "BUY" in final_signal:
                stop_loss = min(s1, current_price * 0.97)
                target = current_price + (current_price - stop_loss) * 1.8
            elif "SELL" in final_signal:
                stop_loss = max(r1, current_price * 1.03)
                target = current_price - (stop_loss - current_price) * 1.8
            else:
                stop_loss = s1
                target = r1

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Support (S1)", f"{s1:.2f}")
            rc2.metric("Resistance (R1)", f"{r1:.2f}")
            rc3.metric("Suggested Stop Loss", f"{stop_loss:.2f}")
            rc4.metric("Suggested Target", f"{target:.2f}")

            # Volatility
            st.subheader("📈 Volatility")
            returns = close.pct_change().dropna()
            vol_20 = returns.tail(20).std() * np.sqrt(252) * 100
            st.metric("20-Day Historical Volatility", f"{vol_20:.2f}%")

            if ticker == "^NSEI":
                try:
                    vix = yf.download("^INDIAVIX", period="5d", progress=False, auto_adjust=True)
                    if isinstance(vix.columns, pd.MultiIndex):
                        vix.columns = vix.columns.get_level_values(0)
                    if not vix.empty:
                        st.metric("India VIX", f"{float(vix['Close'].iloc[-1]):.2f}")
                except:
                    pass

            # Chart
            st.subheader("Price Chart + EMAs + Support/Resistance")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=close, name="Price", line=dict(color="#1f77b4", width=2)))
            fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], name="EMA 20", line=dict(color="orange", width=1)))
            fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], name="EMA 50", line=dict(color="green", width=1)))
            fig.add_trace(go.Scatter(x=data.index, y=data["EMA200"], name="EMA 200", line=dict(color="red", width=1.5)))
            fig.add_hline(y=s1, line_dash="dot", line_color="green", annotation_text="Support")
            fig.add_hline(y=r1, line_dash="dot", line_color="red", annotation_text="Resistance")
            fig.update_layout(title=f"{user_input} - Price + Indicators", height=520, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            st.warning("""
            **⚠️ Disclaimer**  
            This tool is for **educational purposes only**.  
            It is **NOT financial advice**.  
            Past performance does not guarantee future results.  
            Always manage your risk.
            """)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Try: Nifty 50, Sensex, Gold, Silver, Crude Oil, Reliance, TCS")
