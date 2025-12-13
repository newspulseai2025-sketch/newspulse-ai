from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import pandas as pd
import os
import numpy as np

# If you have these modules
# from indicators.calculate_indicators import process_all_crypto_files
# from indicators.compute_signals import compute_confidence, decide_signal

app = FastAPI(title="Crypto Technical Indicators & Trading Signals API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
PROCESSED_FILE = "data/processed/crypto_tech_indicators_11.csv"
SIGNALS_FILE = "data/processed/crypto_trade_signals_d.csv"
TIMEFRAME = "1d"

# Ticker full names
TICKER_NAMES = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "USDT": "Tether",
    "BNB": "Binance Coin",
    "ADA": "Cardano",
    "SOL": "Solana",
    "XRP": "Ripple",
    "DOT": "Polkadot",
    "AVAX": "Avalanche",
    "MATIC": "Polygon",
    "ATOM": "Cosmos",
    "DAI": "Dai",
    "LTC": "Litecoin",
    "UNI": "Uniswap",
    "ALGO": "Algorand",
    "BCH": "Bitcoin Cash",
    "XLM": "Stellar",
    "XMR": "Monero",
    "LINK": "Chainlink",
    "SUI": "Sui",
    "TON": "Toncoin",
    "TRX": "Tron",
    "USDE": "Ethena USDe",
    "HBAR": "Hedera",
    "DOGE": "Dogecoin"
}

eps = 1e-9

# -------------------------
# Helpers
# -------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def clamp01(x):
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def safe_val(row, key, default=np.nan):
    return float(row.get(key, default) or default)

def compute_confidence(row):
    close = safe_val(row, "Close")
    ema20 = safe_val(row, "EMA_20")
    macd = safe_val(row, "MACD", 0.0)
    rsi = safe_val(row, "RSI_14", 50.0)
    vol_ratio = safe_val(row, "Vol_Ratio", 1.0)

    if np.isnan(close) or np.isnan(ema20):
        trend_score = 0.5
    else:
        rel = (close - ema20) / (ema20 + eps)
        trend_score = sigmoid(rel * 8.0)
        trend_score = clamp01(trend_score)

    macd_score = sigmoid(macd / (abs(close) * 0.0005 + eps))
    macd_score = clamp01(macd_score)

    rsi_score = 1.0 - (abs(rsi - 50.0) / 60.0)
    rsi_score = clamp01(rsi_score)

    vol_score = sigmoid(vol_ratio)
    vol_score = clamp01(vol_score)

    w_trend, w_macd, w_rsi, w_vol = 0.45, 0.25, 0.20, 0.10
    conf = w_trend * trend_score + w_macd * macd_score + w_rsi * rsi_score + w_vol * vol_score
    conf = 0.50 + 0.50 * conf
    return clamp01(conf)

def decide_signal(row):
    close = row.get("Close", np.nan)
    ema20 = row.get("EMA_20", np.nan)
    macd = row.get("MACD", np.nan)
    rsi = row.get("RSI_14", np.nan)

    if pd.isna(close) or pd.isna(ema20) or pd.isna(macd) or pd.isna(rsi):
        return "Hold"
    if (close > ema20) and (macd > 0) and (rsi < 70):
        return "Buy"
    if (close < ema20) and (macd < 0) and (rsi > 30):
        return "Sell"
    return "Hold"

# -------------------------
# Routes
# -------------------------
@app.get("/", tags=["Root"])
def root():
    return {"message": "Crypto Indicators & Signals API is running!"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    if not os.path.exists("templates/index.html"):
        return HTMLResponse(content="<h1>Dashboard HTML not found</h1>")
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/calculate-indicators", tags=["Indicators"])
def api_calculate_indicators():
    # Placeholder: replace with your actual indicator processing function
    if not os.path.exists(PROCESSED_FILE):
        return {"status": "failed", "detail": "Processed file not found. Implement process_all_crypto_files first."}
    df = pd.read_csv(PROCESSED_FILE)
    return {"status": "success", "rows": len(df), "file": PROCESSED_FILE}

@app.post("/compute-signals", tags=["Signals"])
def api_compute_signals():
    if not os.path.exists(PROCESSED_FILE):
        raise HTTPException(status_code=404, detail="Indicators file not found. Run /calculate-indicators first.")

    df = pd.read_csv(PROCESSED_FILE).sort_values(["Ticker", "Date"])
    last_rows = df.groupby("Ticker").last().reset_index()
    out_rows = []

    for _, r in last_rows.iterrows():
        ticker = r["Ticker"].strip()
        close = safe_val(r, "Close")
        confidence = compute_confidence(r)
        signal = decide_signal(r)
        atr = safe_val(r, "ATR", np.nan)
        atr_val = max(abs(close) * 0.01, eps) if pd.isna(atr) or atr == 0 else atr

        if signal == "Buy":
            sl = close - 2 * atr_val
            tp = close + 3 * atr_val
        elif signal == "Sell":
            sl = close + 2 * atr_val
            tp = close - 3 * atr_val
        else:  # Hold
            sl = close - atr_val
            tp = close + atr_val

        symbol = ticker.split("-")[0].upper()
        out_rows.append({
            "Ticker": ticker,
            "Coin_Name": TICKER_NAMES.get(symbol, symbol),
            "Live_Price": round(close, 2),
            "Confidence": round(confidence, 2),
            "Signal": signal,
            "SL": round(sl, 2),
            "TP": round(tp, 2)
        })

    os.makedirs(os.path.dirname(SIGNALS_FILE), exist_ok=True)
    pd.DataFrame(out_rows).to_csv(SIGNALS_FILE, index=False)

    return {"status": "success", "tickers": len(out_rows), "file": SIGNALS_FILE}

@app.get("/indicators/{ticker}", tags=["Indicators"])
def get_indicators(ticker: str):
    if not os.path.exists(PROCESSED_FILE):
        raise HTTPException(status_code=404, detail="Indicators file not found. Run /calculate-indicators first.")

    df = pd.read_csv(PROCESSED_FILE)
    df["Symbol"] = df["Ticker"].astype(str).str.split("-").str[0].str.upper()
    df["Coin_Name"] = df["Symbol"].map(TICKER_NAMES)
    ticker = ticker.strip().upper()
    df = df[df["Symbol"] == ticker]

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].round(2)
    return df.drop(columns=["Symbol"]).to_dict(orient="records")

@app.get("/signal/{ticker}", tags=["Signals"])
def get_signal(ticker: str):
    if not os.path.exists(SIGNALS_FILE):
        raise HTTPException(status_code=404, detail="Signals file not found. Run /compute-signals first.")

    df = pd.read_csv(SIGNALS_FILE)
    df["Symbol"] = df["Ticker"].astype(str).str.split("-").str[0].str.upper()
    df["Coin_Name"] = df["Symbol"].map(TICKER_NAMES)
    ticker = ticker.strip().upper()
    df = df[df["Symbol"] == ticker]

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No signal found for {ticker}")

    numeric_cols = ["Live_Price", "Confidence", "SL", "TP"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round(x, 2) if pd.notnull(x) else None)

    return df.drop(columns=["Symbol"]).to_dict(orient="records")
