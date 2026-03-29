from collectors.drift import fetch_funding_rates, fetch_market_stats
from collectors.jupiter_lend import fetch_earn_tokens
from collectors.kamino import fetch_borrow_and_staking_history
from pipelines.normalize import canonicalize
from storage.db import init_db, SessionLocal
from storage.schemas import ProtocolSnapshot
from sqlalchemy.dialects.sqlite import insert
import pandas as pd

PRIMARY_KEY_COLS = ["ts", "protocol", "market", "asset"]


def _fill_missing_key_parts(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for i, row in x.iterrows():
        raw = row.get("raw_json") if isinstance(row.get("raw_json"), dict) else {}

        if pd.isna(row.get("market")) or row.get("market") in ("", None):
            inferred_market = raw.get("symbol") or raw.get("market") or row.get("venue") or "unknown_market"
            x.at[i, "market"] = f"{inferred_market}:{i}"

        if pd.isna(row.get("asset")) or row.get("asset") in ("", None):
            inferred_asset = raw.get("asset") or raw.get("token") or raw.get("reserve") or "unknown_asset"
            x.at[i, "asset"] = inferred_asset

    return x


def _prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    if "ts" in x.columns:
        x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
    x = x.dropna(subset=["ts", "protocol"])
    x = _fill_missing_key_parts(x)
    x = x.drop_duplicates(subset=PRIMARY_KEY_COLS, keep="last")
    return x


def write_rows(df: pd.DataFrame) -> int:
    if df.empty:
        return 0

    cleaned = _prepare_rows(df)
    if cleaned.empty:
        return 0

    rows = cleaned.to_dict(orient="records")
    for row in rows:
        if isinstance(row.get("ts"), pd.Timestamp):
            row["ts"] = row["ts"].to_pydatetime()

    session = SessionLocal()
    try:
        stmt = insert(ProtocolSnapshot).values(rows)
        updatable_columns = {
            c.name: getattr(stmt.excluded, c.name)
            for c in ProtocolSnapshot.__table__.columns
            if c.name not in PRIMARY_KEY_COLS
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=PRIMARY_KEY_COLS,
            set_=updatable_columns,
        )
        session.execute(stmt)
        session.commit()
    finally:
        session.close()

    return len(rows)


def main():
    init_db()
    written = 0

    try:
        drift_hist = fetch_funding_rates("SOL-PERP")
        if not drift_hist.empty:
            written += write_rows(canonicalize(drift_hist, venue="drift"))
    except Exception as e:
        print(f"[warn] drift funding backfill failed: {e}")

    try:
        drift_live = fetch_market_stats()
        if not drift_live.empty:
            written += write_rows(canonicalize(drift_live, venue="drift"))
    except Exception as e:
        print(f"[warn] drift live fetch failed: {e}")

    try:
        jup = fetch_earn_tokens()
        if not jup.empty:
            written += write_rows(canonicalize(jup, venue="jupiter_lend"))
    except Exception as e:
        print(f"[warn] jupiter lend fetch failed: {e}")

    try:
        kamino = fetch_borrow_and_staking_history()
        if not kamino.empty:
            written += write_rows(canonicalize(kamino, venue="kamino"))
    except Exception as e:
        print(f"[warn] kamino history fetch failed: {e}")

    if written:
        print(f"Wrote {written} rows")
    else:
        print("No rows fetched. Check your API key and selected market/reserve IDs.")


if __name__ == "__main__":
    main()
