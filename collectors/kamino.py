import requests
import pandas as pd
from configs.settings import settings

def fetch_borrow_and_staking_history() -> pd.DataFrame:
    if "YOUR_" in settings.kamino_market_pubkey or "YOUR_" in settings.kamino_reserve_pubkey:
        return pd.DataFrame()
    url = (
        f"https://api.kamino.finance/kamino-market/{settings.kamino_market_pubkey}"
        f"/reserves/{settings.kamino_reserve_pubkey}/borrow-and-staking-apys/history/median"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    if not df.empty:
        if "timestamp" in df.columns:
            df["ts"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
        elif "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        else:
            df["ts"] = pd.Timestamp.utcnow()
        df["protocol"] = "kamino"
        df["market"] = settings.kamino_market_pubkey
        df["asset"] = settings.kamino_reserve_pubkey
        if "borrowInterestApy" in df.columns:
            df["borrow_apy"] = df["borrowInterestApy"]
        if "stakingApyMedian" in df.columns:
            df["deposit_apy"] = df["stakingApyMedian"]
    return df
