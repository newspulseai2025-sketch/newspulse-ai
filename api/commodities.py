from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import numpy as np

app = FastAPI(title="Commodities Indicators & Signals API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
COM_PROCESSED_FILE = "data/processed/commodities_tech_indicators.csv"
COM_SIGNALS_FILE = "data/processed/commodities_trade_signals.csv"
eps = 1e-9

# Commodity names (must match CSV tickers)
COMMODITY_NAMES = {
    "BZF": "Brent Futures",
    "CLF": "Crude Oil Futures",
    "NGF": "Natural Gas Futures",
    "GCF": "Gold Futures",
    "SIF": "Silver Futures",
    "RBF": "Gasoline Futures",
    "HOF": "Heating Oil Futures"
}

# ----------------- Helpers -----------------
def sigmoid(x): return 1 / (1 + np.exp(-x))
def clamp01(x): return float(np.clip(x, 0.0, 1.0))
def safe_val(row, key, default=np.nan): return float(row.get(key, default) or default)

def compute_confidence(row):
    close = safe_val(row, "Close")
    ema20 = safe_val(row, "EMA_20")
    macd = safe_val(row, "MACD", 0.0)
    rsi = safe_val(row, "RSI_14", 50.0)
    vol_ratio = safe_val(row, "Vol_Ratio", 1.0)

    trend_score = 0.5 if np.isnan(close) or np.isnan(ema20) else clamp01(sigmoid((close - ema20) / (ema20 + eps) * 8))
    macd_score = clamp01(sigmoid(macd / (abs(close)*0.0005 + eps)))
    rsi_score = clamp01(1.0 - abs(rsi - 50)/60.0)
    vol_score = clamp01(sigmoid(vol_ratio))

    conf = 0.45*trend_score + 0.25*macd_score + 0.2*rsi_score + 0.1*vol_score
    return clamp01(0.5 + 0.5*conf)

def decide_signal(row):
    close, ema20, macd, rsi = safe_val(row, "Close"), safe_val(row, "EMA_20"), safe_val(row, "MACD"), safe_val(row, "RSI_14")
    if pd.isna(close) or pd.isna(ema20) or pd.isna(macd) or pd.isna(rsi): return "Hold"
    if (close > ema20) and (macd > 0) and (rsi < 70): return "Buy"
    if (close < ema20) and (macd < 0) and (rsi > 30): return "Sell"
    return "Hold"

def generate_signals(df, name_map):
    df = df.dropna(subset=["Ticker", "Close"], how="any")
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.upper()
    numeric_cols = ["Close", "EMA_20", "MACD", "RSI_14", "ATR", "Vol_Ratio"]
    for col in numeric_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values(["Ticker", "Date"])
    latest = df.groupby("Ticker").last().reset_index()

    rows = []
    for _, r in latest.iterrows():
        close = safe_val(r, "Close")
        atr = safe_val(r, "ATR", np.nan)
        atr_val = max(abs(close)*0.01, eps) if pd.isna(atr) or atr==0 else atr
        signal = decide_signal(r)

        if signal=="Buy": sl, tp = close - 2*atr_val, close + 3*atr_val
        elif signal=="Sell": sl, tp = close + 2*atr_val, close - 3*atr_val
        else: sl, tp = close - atr_val, close + atr_val

        rows.append({
            "Ticker": r["Ticker"],
            "Name": name_map.get(r["Ticker"], r["Ticker"]),
            "Live_Price": round(close,2),
            "Confidence": round(compute_confidence(r),2),
            "Signal": signal,
            "SL": round(sl,2),
            "TP": round(tp,2)
        })
    return pd.DataFrame(rows)

# ----------------- Routes -----------------
@app.get("/", tags=["Root"])
def root(): 
    return {"message": "Commodities API running!"}

@app.post("/commodities/calculate-indicators", tags=["Indicators"])
def calculate_commodities_indicators():
    if not os.path.exists(COM_PROCESSED_FILE):
        raise HTTPException(404, "Commodity indicators file not found")
    df = pd.read_csv(COM_PROCESSED_FILE)
    return {"status":"success","rows":len(df)}

@app.post("/commodities/compute-signals", tags=["Signals"])
def compute_commodities_signals():
    if not os.path.exists(COM_PROCESSED_FILE):
        raise HTTPException(404,"Commodity indicators missing")
    df = pd.read_csv(COM_PROCESSED_FILE)
    out = generate_signals(df, COMMODITY_NAMES)
    os.makedirs(os.path.dirname(COM_SIGNALS_FILE), exist_ok=True)
    out.to_csv(COM_SIGNALS_FILE,index=False)
    return {"status":"success","rows":len(out)}

@app.get("/commodities/indicators/{ticker}", tags=["Indicators"])
def get_commodity_indicators(ticker:str):
    if not os.path.exists(COM_PROCESSED_FILE):
        raise HTTPException(404,"Commodity indicators file missing")
    df = pd.read_csv(COM_PROCESSED_FILE)
    ticker = ticker.strip().upper()
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    df = df[df["Ticker"]==ticker]
    if df.empty:
        raise HTTPException(404,f"No data found for {ticker}")
    return df.round(2).to_dict("records")

@app.get("/commodities/signal/{ticker}", tags=["Signals"])
def get_commodity_signal(ticker:str):
    if not os.path.exists(COM_SIGNALS_FILE):
        raise HTTPException(404,"Run /commodities/compute-signals first")
    df = pd.read_csv(COM_SIGNALS_FILE)
    ticker = ticker.strip().upper()
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    df = df[df["Ticker"]==ticker]
    if df.empty:
        raise HTTPException(404,f"No signal found for {ticker}")
    return df.to_dict("records")
