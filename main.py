import io
import urllib.request
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# KONFIGURASI APLIKASI
# ============================================================

st.set_page_config(
    page_title="XAUUSD Market Observer",
    page_icon="🥇",
    layout="wide",
)

st.title("🥇 XAUUSD Market Observer")
st.caption(
    "Pengamat Gold untuk Long-Term / Position Trading — bukan jaminan profit."
)


# ============================================================
# PENGATURAN DATA
# ============================================================

PERIOD_DAYS = {
    "2 tahun": 730,
    "3 tahun": 1095,
    "5 tahun": 1825,
}


# ============================================================
# NORMALISASI DATA
# ============================================================

def normalize_ohlc(df):
    """
    Menyamakan format data dari berbagai sumber
    menjadi Open, High, Low, Close.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Yahoo Finance versi tertentu dapat menghasilkan MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Rapikan nama kolom
    df.columns = [str(col).strip().title() for col in df.columns]

    required = ["Open", "High", "Low", "Close"]

    if not all(column in df.columns for column in required):
        return pd.DataFrame()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(subset=required)

    # Hilangkan candle duplikat
    df = df[~df.index.duplicated(keep="last")]

    # Urutkan tanggal
    df = df.sort_index()

    return df


# ============================================================
# SUMBER 1 — STOOQ XAUUSD SPOT
# ============================================================

def fetch_stooq(period_label):

    days = PERIOD_DAYS[period_label]

    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 20)

    url = (
        "https://stooq.com/q/d/l/"
        f"?s=xauusd"
        f"&d1={start.strftime('%Y%m%d')}"
        f"&d2={end.strftime('%Y%m%d')}"
        f"&i=d"
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=15
        ) as response:

            raw = response.read()

        df = pd.read_csv(
            io.BytesIO(raw)
        )

        if "Date" not in df.columns:
            return pd.DataFrame()

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["Date"]
        )

        df = df.set_index("Date")

        return normalize_ohlc(df)

    except Exception:
        return pd.DataFrame()


# ============================================================
# SUMBER 2 — YAHOO FINANCE XAUUSD SPOT
# ============================================================

def fetch_yahoo(period_label):

    period = {
        "2 tahun": "2y",
        "3 tahun": "3y",
        "5 tahun": "5y",
    }[period_label]

    try:

        df = yf.download(
            "XAUUSD=X",
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=15,
        )

        return normalize_ohlc(df)

    except Exception:
        return pd.DataFrame()


# ============================================================
# SUMBER 3 — GOLD FUTURES SEBAGAI FALLBACK
# ============================================================

def fetch_gold_futures(period_label):

    period = {
        "2 tahun": "2y",
        "3 tahun": "3y",
        "5 tahun": "5y",
    }[period_label]

    try:

        df = yf.download(
            "GC=F",
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=15,
        )

        return normalize_ohlc(df)

    except Exception:
        return pd.DataFrame()


# ============================================================
# LOAD DATA DENGAN SISTEM FALLBACK
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False
)
def load_daily_data(period_label):

    # ----------------------------------------
    # PRIORITAS 1
    # ----------------------------------------

    df = fetch_stooq(
        period_label
    )

    if not df.empty:

        return (
            df,
            "Stooq — XAUUSD spot"
        )

    # ----------------------------------------
    # PRIORITAS 2
    # ----------------------------------------

    df = fetch_yahoo(
        period_label
    )

    if not df.empty:

        return (
            df,
            "Yahoo Finance — XAUUSD spot"
        )

    # ----------------------------------------
    # PRIORITAS 3
    # ----------------------------------------

    df = fetch_gold_futures(
        period_label
    )

    if not df.empty:

        return (
            df,
            "Yahoo Finance — Gold Futures (GC=F), proxy"
        )

    # ----------------------------------------
    # SEMUA GAGAL
    # ----------------------------------------

    return (
        pd.DataFrame(),
        "Tidak ada sumber data"
    )


# ============================================================
# UBAH DAILY MENJADI WEEKLY
# ============================================================

def make_weekly(df):

    if df.empty:
        return df

    weekly = df.resample(
        "W-FRI"
    ).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
        }
    )

    if "Volume" in df.columns:

        weekly["Volume"] = (
            df["Volume"]
            .resample("W-FRI")
            .sum()
        )

    weekly = weekly.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    return weekly


# ============================================================
# INDIKATOR SMA
# ============================================================

def sma(series, period):

    return series.rolling(
        period
    ).mean()


# ============================================================
# INDIKATOR EMA
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# INDIKATOR RSI
# ============================================================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = (
        average_gain /
        average_loss.replace(
            0,
            pd.NA
        )
    )

    result = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    return result.fillna(50)


# ============================================================
# TAMBAHKAN SEMUA INDIKATOR
# ============================================================

def add_indicators(df):

    df = df.copy()

    # ----------------------------
    # SMA
    # ----------------------------

    df["SMA50"] = sma(
        df["Close"],
        50
    )

    df["SMA200"] = sma(
        df["Close"],
        200
    )

    # ----------------------------
    # MACD
    # ----------------------------

    df["EMA12"] = ema(
        df["Close"],
        12
    )

    df["EMA26"] = ema(
        df["Close"],
        26
    )

    df["MACD"] = (
        df["EMA12"] -
        df["EMA26"]
    )

    df["MACD_SIGNAL"] = ema(
        df["MACD"],
        9
    )

    df["MACD_HIST"] = (
        df["MACD"] -
        df["MACD_SIGNAL"]
    )

    # ----------------------------
    # RSI
    # ----------------------------

    df["RSI"] = rsi(
        df["Close"],
        14
    )

    # ----------------------------
    # GOLDEN CROSS
    # ----------------------------

    df["GOLDEN"] = (
        (
            df["SMA50"] >
            df["SMA200"]
        )
        &
        (
            df["SMA50"].shift(1)
            <=
            df["SMA200"].shift(1)
        )
    )

    # ----------------------------
    # DEATH CROSS
    # ----------------------------

    df["DEATH"] = (
        (
            df["SMA50"] <
            df["SMA200"]
        )
        &
        (
            df["SMA50"].shift(1)
            >=
            df["SMA200"].shift(1)
        )
    )

    return df


# ============================================================
# ANALISIS MARKET
# ============================================================

def analyze(df):

    current = df.iloc[-1]
    previous = df.iloc[-2]

    score = 0

    reasons = []

    # ========================================================
    # HARGA VS SMA 200
    # ========================================================

    if current["Close"] > current["SMA200"]:

        score += 2

        reasons.append(
            "Harga berada di atas SMA 200 → "
            "bias tren utama bullish."
        )

    else:

        score -= 2

        reasons.append(
            "Harga berada di bawah SMA 200 → "
            "bias tren utama bearish."
        )

    # ========================================================
    # SMA 50 VS SMA 200
    # ========================================================

    if current["SMA50"] > current["SMA200"]:

        score += 2

        reasons.append(
            "SMA 50 berada di atas SMA 200 → "
            "struktur jangka panjang bullish."
        )

    else:

        score -= 2

        reasons.append(
            "SMA 50 berada di bawah SMA 200 → "
            "struktur jangka panjang bearish."
        )

    # ========================================================
    # MACD
    # ========================================================

    if current["MACD"] > current["MACD_SIGNAL"]:

        score += 1

        reasons.append(
            "MACD di atas signal → "
            "momentum positif."
        )

    else:

        score -= 1

        reasons.append(
            "MACD di bawah signal → "
            "momentum negatif."
        )

    # ========================================================
    # RSI
    # ========================================================

    if current["RSI"] >= 50:

        score += 1

        reasons.append(
            f"RSI 14 = {current['RSI']:.1f} → "
            "momentum relatif positif."
        )

    else:

        score -= 1

        reasons.append(
            f"RSI 14 = {current['RSI']:.1f} → "
            "momentum relatif negatif."
        )

    # ========================================================
    # GOLDEN / DEATH CROSS
    # ========================================================

    if current["GOLDEN"]:

        reasons.append(
            "Golden Cross terdeteksi pada candle terakhir."
        )

    if current["DEATH"]:

        reasons.append(
            "Death Cross terdeteksi pada candle terakhir."
        )

    # ========================================================
    # TREND
    # ========================================================

    if (
        current["Close"] >
        current["SMA50"] >
        current["SMA200"]
    ):

        trend = "BULLISH / UPTREND"

    elif (
        current["Close"] <
        current["SMA50"] <
        current["SMA200"]
    ):

        trend = "BEARISH / DOWNTREND"

    else:

        trend = "SIDEWAYS / TRANSISI"

    # ========================================================
    # SIGNAL
    # ========================================================

    if current["GOLDEN"]:

        signal = "BUY"

    elif current["DEATH"]:

        signal = "SELL"

    elif score >= 5:

        signal = "BUY"

    elif score <= -5:

        signal = "SELL"

    else:

        signal = "HOLD"

    # ========================================================
    # PERUBAHAN HARGA
    # ========================================================

    price_change = (
        (
            current["Close"] -
            previous["Close"]
        )
        /
        previous["Close"]
    ) * 100

    return (
        trend,
        signal,
        score,
        reasons,
        current,
        price_change
    )


# ============================================================
# SUPPORT & RESISTANCE
# ============================================================

def support_resistance(
    df,
    window=20
):

    recent = df.tail(
        window
    )

    support = recent[
        "Low"
    ].min()

    resistance = recent[
        "High"
    ].max()

    return (
        support,
        resistance
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Pengaturan"
)

period_label = st.sidebar.selectbox(
    "Data historis",
    list(PERIOD_DAYS.keys()),
    index=2
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    [
        "Daily",
        "Weekly"
    ],
    index=0
)

st.sidebar.info(
    "Fokus aplikasi:\n\n"
    "🥇 XAUUSD\n"
    "📈 Long-Term / Position Trading\n"
    "📊 Daily / Weekly\n\n"
    "Sistem menggunakan SMA 50, "
    "SMA 200, MACD dan RSI 14."
)

if st.sidebar.button(
    "🔄 Refresh data"
):

    st.cache_data.clear()

    st.rerun()


# ============================================================
# AMBIL DATA
# ============================================================

daily_df, source = load_daily_data(
    period_label
)


# ============================================================
# CEK DATA
# ============================================================

if daily_df.empty:

    st.error(
        """
        ❌ Data XAUUSD belum berhasil diambil.

        Coba tekan tombol **Refresh data**
        beberapa saat lagi.
        """
    )

    st.stop()


# ============================================================
# WEEKLY ATAU DAILY
# ============================================================

if timeframe == "Weekly":

    df = make_weekly(
        daily_df
    )

else:

    df = daily_df.copy()


# ============================================================
# HITUNG INDIKATOR
# ============================================================

df = add_indicators(
    df
)

df = df.dropna(
    subset=[
        "SMA50",
        "SMA200",
        "MACD",
        "MACD_SIGNAL",
        "RSI"
    ]
)


# ============================================================
# CEK JUMLAH CANDLE
# ============================================================

if len(df) < 200:

    st.error(
        f"""
        Data yang tersedia hanya
        **{len(df)} candle** setelah pemrosesan.

        Belum cukup untuk menghitung SMA 200.
        """
    )

    st.stop()


# ============================================================
# ANALISIS
# ============================================================

(
    trend,
    signal,
    score,
    reasons,
    current,
    price_change
) = analyze(df)


support, resistance = (
    support_resistance(df)
)


# ============================================================
# INFORMASI SUMBER DATA
# ============================================================

if "proxy" in source.lower():

    st.warning(
        """
        ⚠️ **PERHATIAN**

        Sumber utama XAUUSD spot tidak tersedia.

        Aplikasi sedang menggunakan
        **Gold Futures (GC=F)** sebagai proxy.

        Harga futures dapat berbeda dari harga
        XAUUSD pada broker/MT5 kamu.
        """
    )

else:

    st.caption(
        f"📡 Sumber data: **{source}**"
    )


# ============================================================
# METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Harga XAUUSD",
    f"{current['Close']:,.2f}",
    f"{price_change:+.2f}%"
)


c2.metric(
    "SMA 50",
    f"{current['SMA50']:,.2f}"
)


c3.metric(
    "SMA 200",
    f"{current['SMA200']:,.2f}"
)


c4.metric(
    "RSI 14",
    f"{current['RSI']:.1f}"
)


# ============================================================
# STATUS MARKET
# ============================================================

st.subheader(
    "📊 Status Market"
)


if trend.startswith(
    "BULLISH"
):

    st.success(
        f"🟢 {trend}"
    )

elif trend.startswith(
    "BEARISH"
):

    st.error(
        f"🔴 {trend}"
    )

else:

    st.warning(
        f"🟡 {trend}"
    )


# ============================================================
# SIGNAL
# ============================================================

st.subheader(
    "🧭 Sinyal Jangka Panjang"
)


if signal == "BUY":

    st.success(
        f"🟢 BUY | Score {score:+d}/6"
    )

elif signal == "SELL":

    st.error(
        f"🔴 SELL | Score {score:+d}/6"
    )

else:

    st.warning(
        f"🟡 HOLD | Score {score:+d}/6"
    )


st.caption(
    "Sinyal merupakan hasil aturan indikator "
    "aplikasi dan bukan jaminan harga akan naik "
    "atau turun."
)


# ============================================================
# ALASAN SISTEM
# ============================================================

st.write(
    "**Alasan sistem:**"
)

for reason in reasons:

    st.write(
        "• " + reason
    )


# ============================================================
# SUPPORT & RESISTANCE
# ============================================================

st.subheader(
    "🎯 Area Harga"
)

l1, l2 = st.columns(2)


l1.metric(
    "Support ± 20 candle",
    f"{support:,.2f}"
)


l2.metric(
    "Resistance ± 20 candle",
    f"{resistance:,.2f}"
)


# ============================================================
# CHART
# ============================================================

st.subheader(
    f"📈 Chart XAUUSD — {timeframe}"
)


fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    row_heights=[
        0.60,
        0.20,
        0.20
    ],
    vertical_spacing=0.03,
    subplot_titles=(
        "XAUUSD + SMA 50/200",
        "MACD",
        "RSI 14"
    )
)


# ============================================================
# CANDLESTICK
# ============================================================

fig.add_trace(

    go.Candlestick(

        x=df.index,

        open=df["Open"],

        high=df["High"],

        low=df["Low"],

        close=df["Close"],

        name="XAUUSD"

    ),

    row=1,
    col=1
)


# ============================================================
# SMA 50
# ============================================================

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["SMA50"],

        name="SMA 50",

        mode="lines"

    ),

    row=1,
    col=1
)


# ============================================================
# SMA 200
# ============================================================

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["SMA200"],

        name="SMA 200",

        mode="lines"

    ),

    row=1,
    col=1
)


# ============================================================
# SUPPORT
# ============================================================

fig.add_hline(

    y=support,

    line_dash="dot",

    annotation_text="Support",

    row=1,
    col=1
)


# ============================================================
# RESISTANCE
# ============================================================

fig.add_hline(

    y=resistance,

    line_dash="dot",

    annotation_text="Resistance",

    row=1,
    col=1
)


# ============================================================
# MACD
# ============================================================

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["MACD"],

        name="MACD",

        mode="lines"

    ),

    row=2,
    col=1
)


# ============================================================
# MACD SIGNAL
# ============================================================

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["MACD_SIGNAL"],

        name="Signal",

        mode="lines"

    ),

    row=2,
    col=1
)


# ============================================================
# MACD HISTOGRAM
# ============================================================

fig.add_trace(

    go.Bar(

        x=df.index,

        y=df["MACD_HIST"],

        name="Histogram"

    ),

    row=2,
    col=1
)


# ============================================================
# RSI
# ============================================================

fig.add_trace(

    go.Scatter(

        x=df.index,

        y=df["RSI"],

        name="RSI 14",

        mode="lines"

    ),

    row=3,
    col=1
)


# ============================================================
# RSI LEVEL
# ============================================================

fig.add_hline(
    y=70,
    line_dash="dash",
    row=3,
    col=1
)

fig.add_hline(
    y=50,
    line_dash="dot",
    row=3,
    col=1
)

fig.add_hline(
    y=30,
    line_dash="dash",
    row=3,
    col=1
)


# ============================================================
# LAYOUT CHART
# ============================================================

fig.update_layout(

    height=950,

    xaxis_rangeslider_visible=False,

    hovermode="x unified",

    margin=dict(
        l=10,
        r=10,
        t=60,
        b=10
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# DATA CANDLE TERAKHIR
# ============================================================

st.subheader(
    "🕯️ Data Candle Terakhir"
)


display_df = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "SMA50",
        "SMA200",
        "MACD",
        "RSI"
    ]
].tail(10).copy()


st.dataframe(
    display_df.round(
        {
            "Open": 2,
            "High": 2,
            "Low": 2,
            "Close": 2,
            "SMA50": 2,
            "SMA200": 2,
            "MACD": 2,
            "RSI": 1
        }
    ),
    use_container_width=True
)


# ============================================================
# DOWNLOAD CSV
# ============================================================

csv_data = df.to_csv().encode(
    "utf-8"
)


st.download_button(

    "⬇️ Download data CSV",

    data=csv_data,

    file_name="xauusd_market_observer.csv",

    mime="text/csv"
)


# ============================================================
# CARA MEMBACA
# ============================================================

st.subheader(
    "📖 Cara Membaca"
)


st.markdown(
    """
### 🟢 BULLISH

Harga > SMA 50 > SMA 200.

Artinya struktur tren jangka panjang
sedang mengarah naik.

### 🔴 BEARISH

Harga < SMA 50 < SMA 200.

Artinya struktur tren jangka panjang
sedang mengarah turun.

### 🟡 SIDEWAYS / TRANSISI

SMA dan harga belum membentuk struktur
yang searah.

### BUY

Sistem memberikan BUY ketika kondisi
bullish cukup kuat atau terjadi Golden Cross.

### SELL

Sistem memberikan SELL ketika kondisi
bearish cukup kuat atau terjadi Death Cross.

### HOLD

Kondisi belum cukup kuat.

Jangan memaksakan posisi hanya karena
aplikasi belum memberikan BUY atau SELL.

### RSI

**RSI 70/30 bukan tombol otomatis BUY/SELL.**

RSI digunakan sebagai informasi momentum
dan kondisi ekstrem.

### Support / Resistance

Support dan resistance yang ditampilkan
merupakan area berdasarkan harga historis
terbaru, bukan level yang pasti memantulkan
harga.
"""
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Data pasar dapat berbeda dengan harga broker/MT5 "
    "karena sumber data dan waktu pembentukan candle "
    "dapat berbeda."
)

st.caption(
    "Gunakan aplikasi sebagai alat bantu analisis, "
    "bukan sebagai jaminan profit."
        )
