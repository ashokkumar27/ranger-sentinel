import requests
import pandas as pd
from configs.settings import settings

BASE = settings.drift_base_url.rstrip("/")

def _get(path: str, params: dict | None = None):
    r = requests.get(f"{BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_market_stats() -> pd.DataFrame:
    data = _get("/stats/markets")
    df = pd.DataFrame(data)
    if not df.empty:
        df["ts"] = pd.Timestamp.utcnow()
        df["funding_rate_daily"] = df.get("fundingRate24hAvg", 0.0)
        df["price_usd"] = df.get("oraclePrice")
        df["market"] = df.get("symbol")
        df["protocol"] = "drift"
    return df

def fetch_funding_rates(symbol: str) -> pd.DataFrame:
    data = _get("/fundingRates", {"symbol": symbol})
    df = pd.DataFrame(data)
    if not df.empty:
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True, errors="coerce")
        else:
            df["ts"] = pd.Timestamp.utcnow()
        df["protocol"] = "drift"
        df["market"] = df.get("symbol", symbol)
        df["funding_rate_daily"] = df.get("fundingRate", 0.0)
        df["price_usd"] = df.get("oraclePrice")
    return df
