import pandas as pd

CANONICAL_COLS = [
    "ts", "protocol", "venue", "market", "asset", "deposit_apy", "borrow_apy",
    "funding_rate_daily", "utilization", "available_liquidity_usd", "price_usd", "raw_json",
]

def canonicalize(df: pd.DataFrame, venue: str | None = None) -> pd.DataFrame:
    x = df.copy()
    if venue is not None:
        x["venue"] = venue
    for col in CANONICAL_COLS:
        if col not in x.columns:
            x[col] = None
    x["raw_json"] = x.apply(lambda r: r.to_dict(), axis=1)
    return x[CANONICAL_COLS]
