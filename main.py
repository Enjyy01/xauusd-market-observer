
import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="XAUUSD Market Observer", page_icon="🥇", layout="wide")

st.title("🥇 XAUUSD Market Observer")
st.caption("Pengamat Gold untuk Long-Term / Position Trading — bukan jaminan profit.")

@st.cache_data(ttl=3600)
def load_data(period, interval):
    df = yf.download(
        "XAUUSD=X",
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def indicators(df):
    df = df.copy()
    df["SMA50"] = SMAIndicator(df["Close"], 50).sma_indicator()
    df["SMA200"] = SMAIndicator(df["Close"], 200).sma_indicator()

    m = MACD(df["Close"], 26, 12, 9)
    df["MACD"] = m.macd()
    df["MACD_SIGNAL"] = m.macd_signal()
    df["MACD_HIST"] = m.macd_diff()
    df["RSI"] = RSIIndicator(df["Close"], 14).rsi()

    df["GOLDEN"] = (df["SMA50"] > df["SMA200"]) & (df["SMA50"].shift(1) <= df["SMA200"].shift(1))
    df["DEATH"] = (df["SMA50"] < df["SMA200"]) & (df["SMA50"].shift(1) >= df["SMA200"].shift(1))
    return df.dropna()

def analyse(df):
    x = df.iloc[-1]
    score = 0
    reasons = []

    if x.Close > x.SMA200:
        score += 2
        reasons.append("Harga di atas SMA 200 → tren utama bullish.")
    else:
        score -= 2
        reasons.append("Harga di bawah SMA 200 → tren utama bearish.")

    if x.SMA50 > x.SMA200:
        score += 2
        reasons.append("SMA 50 di atas SMA 200 → struktur bullish.")
    else:
        score -= 2
        reasons.append("SMA 50 di bawah SMA 200 → struktur bearish.")

    if x.MACD > x.MACD_SIGNAL:
        score += 1
        reasons.append("MACD di atas signal → momentum bullish.")
    else:
        score -= 1
        reasons.append("MACD di bawah signal → momentum bearish.")

    if x.RSI >= 50:
        score += 1
        reasons.append(f"RSI {x.RSI:.1f} di atas 50 → momentum relatif bullish.")
    else:
        score -= 1
        reasons.append(f"RSI {x.RSI:.1f} di bawah 50 → momentum relatif bearish.")

    if x.GOLDEN:
        signal = "BUY"
        reasons.append("Terjadi Golden Cross.")
    elif x.DEATH:
        signal = "SELL"
        reasons.append("Terjadi Death Cross.")
    elif score >= 5:
        signal = "BUY"
    elif score <= -5:
        signal = "SELL"
    else:
        signal = "HOLD"

    if x.Close > x.SMA50 > x.SMA200:
        trend = "BULLISH / UPTREND"
    elif x.Close < x.SMA50 < x.SMA200:
        trend = "BEARISH / DOWNTREND"
    else:
        trend = "SIDEWAYS / TRANSISI"

    return trend, signal, score, reasons, x

period = st.sidebar.selectbox("Data historis", ["2y", "3y", "5y"], index=2)
interval = st.sidebar.selectbox("Timeframe", ["1d", "1wk"], index=0)
st.sidebar.info("Fokus aplikasi: Daily/Weekly. Sinyal dibuat konservatif agar tidak terlalu berisik.")

df = load_data(period, interval)

if df.empty:
    st.error("Data XAUUSD tidak tersedia. Coba refresh beberapa saat lagi.")
    st.stop()

df = indicators(df)
if len(df) < 200:
    st.error("Data belum cukup untuk menghitung SMA 200.")
    st.stop()

trend, signal, score, reasons, x = analyse(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Harga XAUUSD", f"{x.Close:,.2f}")
c2.metric("SMA 50", f"{x.SMA50:,.2f}")
c3.metric("SMA 200", f"{x.SMA200:,.2f}")
c4.metric("RSI 14", f"{x.RSI:.1f}")

st.subheader("Status Market")
if trend.startswith("BULLISH"):
    st.success("🟢 " + trend)
elif trend.startswith("BEARISH"):
    st.error("🔴 " + trend)
else:
    st.warning("🟡 " + trend)

st.subheader("Sinyal Jangka Panjang")
if signal == "BUY":
    st.success(f"🟢 BUY  |  Score {score:+d}/6")
elif signal == "SELL":
    st.error(f"🔴 SELL  |  Score {score:+d}/6")
else:
    st.warning(f"🟡 HOLD  |  Score {score:+d}/6")

st.write("**Alasan sistem:**")
for r in reasons:
    st.write("• " + r)

fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.60, 0.20, 0.20],
                    vertical_spacing=0.03,
                    subplot_titles=("XAUUSD + SMA 50/200", "MACD", "RSI 14"))

fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low,
                             close=df.Close, name="XAUUSD"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.SMA50, name="SMA 50"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.SMA200, name="SMA 200"), row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df.MACD, name="MACD"), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df.MACD_SIGNAL, name="Signal"), row=2, col=1)
fig.add_trace(go.Bar(x=df.index, y=df.MACD_HIST, name="Histogram"), row=2, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI 14"), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", row=3, col=1)
fig.add_hline(y=50, line_dash="dot", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", row=3, col=1)

fig.update_layout(height=900, xaxis_rangeslider_visible=False, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📖 Cara Membaca")
st.write("""
- **Bullish**: harga > SMA 50 > SMA 200.
- **Bearish**: harga < SMA 50 < SMA 200.
- **BUY**: skor bullish kuat atau terjadi Golden Cross.
- **SELL**: skor bearish kuat atau terjadi Death Cross.
- **HOLD**: kondisi belum cukup kuat; jangan memaksakan posisi.
- RSI 70/30 adalah area ekstrem, bukan tombol otomatis BUY/SELL.
""")

st.caption("Data: Yahoo Finance via yfinance. Gunakan sebagai alat bantu analisis, bukan nasihat investasi.")
