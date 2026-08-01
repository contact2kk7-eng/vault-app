# ============================================================
# Daily automated scan -- runs after market close via GitHub Actions.
# Writes results to scan_results.json for the Vault app to display.
# Uses real Nifty 50 data via yfinance. No fabricated numbers.
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timezone
import warnings
warnings.filterwarnings("ignore")

NIFTY50_TICKERS = [
    "ADANIENT.NS","ADANIPORTS.NS","APOLLOHOSP.NS","ASIANPAINT.NS","AXISBANK.NS",
    "BAJAJ-AUTO.NS","BAJFINANCE.NS","BAJAJFINSV.NS","BPCL.NS","BHARTIARTL.NS",
    "BRITANNIA.NS","CIPLA.NS","COALINDIA.NS","DIVISLAB.NS","DRREDDY.NS",
    "EICHERMOT.NS","GRASIM.NS","HCLTECH.NS","HDFCBANK.NS","HDFCLIFE.NS",
    "HEROMOTOCO.NS","HINDALCO.NS","HINDUNILVR.NS","ICICIBANK.NS","ITC.NS",
    "INDUSINDBK.NS","INFY.NS","JSWSTEEL.NS","KOTAKBANK.NS","LT.NS",
    "M&M.NS","MARUTI.NS","NTPC.NS","NESTLEIND.NS","ONGC.NS",
    "POWERGRID.NS","RELIANCE.NS","SBILIFE.NS","SBIN.NS","SUNPHARMA.NS",
    "TCS.NS","TATACONSUM.NS","TATASTEEL.NS","TECHM.NS",
    "TITAN.NS","ULTRACEMCO.NS","UPL.NS","WIPRO.NS"
]

def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def scan_breakout_bullish(df):
    """6-month breakout + volume + RSI<70 -- the strategy we already tested and found underperforms."""
    prior_high = df["High"].shift(1).rolling(125).max()
    prior_vol_avg = df["Volume"].shift(1).rolling(125).mean()
    rsi = compute_rsi(df["Close"], 14)
    latest = df.iloc[-1]
    return bool(
        latest["Close"] > prior_high.iloc[-1] and
        latest["Volume"] > 2 * prior_vol_avg.iloc[-1] and
        rsi.iloc[-1] < 70
    ) if len(df) > 130 else False

def scan_breakdown_bearish(df):
    """Mirror version: 6-month breakdown + volume + RSI>30."""
    prior_low = df["Low"].shift(1).rolling(125).min()
    prior_vol_avg = df["Volume"].shift(1).rolling(125).mean()
    rsi = compute_rsi(df["Close"], 14)
    latest = df.iloc[-1]
    return bool(
        latest["Close"] < prior_low.iloc[-1] and
        latest["Volume"] > 2 * prior_vol_avg.iloc[-1] and
        rsi.iloc[-1] > 30
    ) if len(df) > 130 else False

def main():
    data = yf.download(NIFTY50_TICKERS, period="1y", group_by='ticker', progress=False, threads=True)

    bullish_hits = []
    bearish_hits = []
    errors = []

    for ticker in NIFTY50_TICKERS:
        try:
            df = data[ticker].dropna(subset=["Close", "Volume"]).copy()
            if scan_breakout_bullish(df):
                bullish_hits.append({"ticker": ticker, "close": round(float(df["Close"].iloc[-1]), 2)})
            if scan_breakdown_bearish(df):
                bearish_hits.append({"ticker": ticker, "close": round(float(df["Close"].iloc[-1]), 2)})
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe": "Nifty 50 (current constituent list)",
        "strategies": {
            "breakout_bullish": {
                "label": "Breakout + Volume + RSI<70",
                "known_performance": "Independently backtested: 5.2% CAGR, -53.5% max drawdown, underperformed Nifty 50 buy-and-hold (10.0% CAGR). Treat signals as informational, not a trading recommendation.",
                "hits": bullish_hits
            },
            "breakdown_bearish": {
                "label": "Breakdown + Volume + RSI>30",
                "known_performance": "Not yet independently backtested by us -- mirror logic of the bullish version above, same underlying caveats likely apply.",
                "hits": bearish_hits
            }
        },
        "errors": errors,
        "data_source": "yfinance (Yahoo Finance) -- free, delayed EOD data, not a licensed real-time feed"
    }

    with open("scan_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Scan complete. Bullish hits: {len(bullish_hits)}, Bearish hits: {len(bearish_hits)}, Errors: {len(errors)}")

if __name__ == "__main__":
    main()
