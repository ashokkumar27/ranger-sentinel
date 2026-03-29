import requests
import pandas as pd
from configs.settings import settings

URL = "https://api.jup.ag/lend/v1/earn/tokens"

def fetch_earn_tokens() -> pd.DataFrame:
    if not settings.jupiter_api_key:
        return pd.DataFrame()
    headers = {"x-api-key": settings.jupiter_api_key}
    r = requests.get(URL, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if not df.empty:
        df["ts"] = pd.Timestamp.utcnow()
        df["protocol"] = "jupiter_lend"
        if "assetSymbol" in df.columns:
            df["asset"] = df["assetSymbol"]
        elif "symbol" in df.columns:
            df["asset"] = df["symbol"]
        if "apy" in df.columns:
            df["deposit_apy"] = df["apy"]
        elif "supplyApy" in df.columns:
            df["deposit_apy"] = df["supplyApy"]
        if "liquidityUsd" in df.columns:
            df["available_liquidity_usd"] = df["liquidityUsd"]
    return df
