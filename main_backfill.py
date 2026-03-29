from collectors.drift import fetch_funding_rates, fetch_market_stats
from collectors.jupiter_lend import fetch_earn_tokens
from collectors.kamino import fetch_borrow_and_staking_history
from pipelines.normalize import canonicalize
from storage.db import init_db, SessionLocal
from storage.schemas import ProtocolSnapshot
import pandas as pd

def write_rows(df: pd.DataFrame):
    if df.empty:
        return
    session = SessionLocal()
    try:
        for row in df.to_dict(orient="records"):
            session.merge(ProtocolSnapshot(**row))
        session.commit()
    finally:
        session.close()

def main():
    init_db()
    frames = []

    try:
        drift_hist = fetch_funding_rates("SOL-PERP")
        if not drift_hist.empty:
            frames.append(canonicalize(drift_hist, venue="drift"))
    except Exception as e:
        print(f"[warn] drift funding backfill failed: {e}")

    try:
        drift_live = fetch_market_stats()
        if not drift_live.empty:
            frames.append(canonicalize(drift_live, venue="drift"))
    except Exception as e:
        print(f"[warn] drift live fetch failed: {e}")

    try:
        jup = fetch_earn_tokens()
        if not jup.empty:
            frames.append(canonicalize(jup, venue="jupiter_lend"))
    except Exception as e:
        print(f"[warn] jupiter lend fetch failed: {e}")

    try:
        kamino = fetch_borrow_and_staking_history()
        if not kamino.empty:
            frames.append(canonicalize(kamino, venue="kamino"))
    except Exception as e:
        print(f"[warn] kamino history fetch failed: {e}")

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        write_rows(all_df)
        print(f"Wrote {len(all_df)} rows")
    else:
        print("No rows fetched. Check your API key and selected market/reserve IDs.")

if __name__ == "__main__":
    main()
