from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict
from typing import Callable

import pandas as pd
from sqlalchemy.dialects.sqlite import insert

from collectors.drift import fetch_funding_rates, fetch_market_stats
from collectors.jupiter_lend import fetch_earn_tokens
from collectors.kamino import fetch_borrow_and_staking_history
from pipelines.normalize import canonicalize
from storage.db import SessionLocal, init_db
from storage.schemas import ProtocolSnapshot

PRIMARY_KEY_COLS = ["ts", "protocol", "market", "asset"]
NUMERIC_COLS = [
    "deposit_apy",
    "borrow_apy",
    "funding_rate_daily",
    "utilization",
    "available_liquidity_usd",
    "price_usd",
]
CANONICAL_REQUIRED_COLS = ["ts", "protocol", "venue", "market", "asset", "raw_json"]


class DataQualityError(RuntimeError):
    """Raised when a critical data source is missing or malformed."""


@dataclass
class SourceHealth:
    source: str
    rows_raw: int = 0
    rows_canonical: int = 0
    rows_ready_to_write: int = 0
    non_null_ts: int = 0
    non_null_market: int = 0
    non_null_asset: int = 0
    non_null_deposit_apy: int = 0
    non_null_borrow_apy: int = 0
    non_null_funding_rate_daily: int = 0
    non_null_available_liquidity_usd: int = 0
    min_ts: str | None = None
    max_ts: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _fill_missing_key_parts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for idx, row in out.iterrows():
        raw = row.get("raw_json") if isinstance(row.get("raw_json"), dict) else {}

        market = row.get("market")
        if pd.isna(market) or market in ("", None):
            inferred_market = (
                raw.get("symbol")
                or raw.get("market")
                or raw.get("venue")
                or row.get("venue")
                or "unknown_market"
            )
            out.at[idx, "market"] = f"{inferred_market}:{idx}"

        asset = row.get("asset")
        if pd.isna(asset) or asset in ("", None):
            inferred_asset = (
                raw.get("asset")
                or raw.get("token")
                or raw.get("reserve")
                or raw.get("assetSymbol")
                or raw.get("symbol")
                or "unknown_asset"
            )
            out.at[idx, "asset"] = inferred_asset

    return out


def _prepare_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")

    out = out.dropna(subset=["ts", "protocol"])
    out = _fill_missing_key_parts(out)

    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.drop_duplicates(subset=PRIMARY_KEY_COLS, keep="last")
    return out


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
        update_cols = {
            c.name: getattr(stmt.excluded, c.name)
            for c in ProtocolSnapshot.__table__.columns
            if c.name not in PRIMARY_KEY_COLS
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=PRIMARY_KEY_COLS,
            set_=update_cols,
        )
        session.execute(stmt)
        session.commit()
    finally:
        session.close()

    return len(rows)


def _iso_or_none(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def summarize_health(source: str, raw_df: pd.DataFrame, canonical_df: pd.DataFrame) -> SourceHealth:
    health = SourceHealth(source=source, rows_raw=len(raw_df), rows_canonical=len(canonical_df))

    if canonical_df.empty:
        return health

    prepared = _prepare_rows(canonical_df)
    health.rows_ready_to_write = len(prepared)

    if "ts" in prepared.columns:
        ts_series = pd.to_datetime(prepared["ts"], utc=True, errors="coerce")
        health.non_null_ts = int(ts_series.notna().sum())
        if health.non_null_ts:
            health.min_ts = _iso_or_none(ts_series.min())
            health.max_ts = _iso_or_none(ts_series.max())

    for col in ["market", "asset", "deposit_apy", "borrow_apy", "funding_rate_daily", "available_liquidity_usd"]:
        if col in prepared.columns:
            value = int(prepared[col].notna().sum())
            setattr(health, f"non_null_{col}", value)

    return health


def _require_columns(df: pd.DataFrame, required: list[str], source: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataQualityError(f"{source}: missing canonical columns: {missing}")


def _require_min_rows(health: SourceHealth, minimum: int) -> None:
    if health.rows_ready_to_write < minimum:
        raise DataQualityError(
            f"{health.source}: only {health.rows_ready_to_write} prepared rows, need at least {minimum}"
        )


def _require_non_null(health: SourceHealth, field: str, minimum: int = 1) -> None:
    attr = f"non_null_{field}"
    count = getattr(health, attr, 0)
    if count < minimum:
        raise DataQualityError(
            f"{health.source}: field '{field}' has only {count} usable rows, need at least {minimum}"
        )


def _require_freshness(health: SourceHealth, max_age_days: int) -> None:
    if not health.max_ts:
        raise DataQualityError(f"{health.source}: missing max timestamp; cannot validate freshness")

    max_ts = pd.to_datetime(health.max_ts, utc=True, errors="coerce")
    now = pd.Timestamp.utcnow()
    age_days = (now - max_ts).total_seconds() / 86400.0
    if age_days > max_age_days:
        raise DataQualityError(
            f"{health.source}: latest row is too old ({age_days:.2f} days > {max_age_days} days)"
        )


def validate_source(source: str, canonical_df: pd.DataFrame, health: SourceHealth) -> None:
    _require_columns(canonical_df, CANONICAL_REQUIRED_COLS, source)
    _require_min_rows(health, 1)

    if source == "drift_hist":
        _require_non_null(health, "funding_rate_daily", 1)
        _require_non_null(health, "market", 1)
        _require_freshness(health, max_age_days=14)

    elif source == "drift_live":
        _require_non_null(health, "funding_rate_daily", 1)
        _require_non_null(health, "market", 1)
        _require_non_null(health, "asset", 1)

    elif source == "jupiter_lend":
        _require_non_null(health, "deposit_apy", 1)
        _require_non_null(health, "asset", 1)
        _require_freshness(health, max_age_days=3)

    elif source == "kamino":
        if health.non_null_deposit_apy < 1 and health.non_null_borrow_apy < 1:
            raise DataQualityError(
                "kamino: both deposit_apy and borrow_apy are unusable"
            )
        _require_non_null(health, "market", 1)
        _require_non_null(health, "asset", 1)
        _require_freshness(health, max_age_days=30)

    else:
        raise DataQualityError(f"Unknown source validator requested: {source}")


def fetch_validate_and_canonicalize(
    source: str,
    fetcher: Callable[[], pd.DataFrame],
    venue: str,
) -> tuple[pd.DataFrame, SourceHealth]:
    raw_df = fetcher()
    if raw_df is None:
        raw_df = pd.DataFrame()

    canonical_df = canonicalize(raw_df, venue=venue)
    health = summarize_health(source=source, raw_df=raw_df, canonical_df=canonical_df)
    validate_source(source=source, canonical_df=canonical_df, health=health)
    return canonical_df, health


def print_health_report(health_rows: list[SourceHealth]) -> None:
    if not health_rows:
        print("No source health rows to report.")
        return

    report = pd.DataFrame([h.to_dict() for h in health_rows])
    report = report.sort_values("source").reset_index(drop=True)
    print("\n=== Source Health Report ===")
    print(report.to_string(index=False))


def assert_cross_venue_coverage(combined: pd.DataFrame) -> None:
    prepared = _prepare_rows(combined)
    if prepared.empty:
        raise DataQualityError("No prepared rows after combining all sources")

    venues = sorted(v for v in prepared["venue"].dropna().unique().tolist() if v)
    protocols = sorted(v for v in prepared["protocol"].dropna().unique().tolist() if v)

    if len(venues) < 2:
        raise DataQualityError(
            f"Cross-venue coverage too weak: found venues={venues}, need at least 2 venues"
        )

    if len(protocols) < 3:
        raise DataQualityError(
            f"Protocol coverage too weak: found protocols={protocols}, need at least 3 protocols"
        )

    if prepared["asset"].dropna().nunique() < 2:
        raise DataQualityError(
            "Asset coverage too weak: need at least 2 distinct assets across sources"
        )


def main() -> None:
    init_db()

    health_rows: list[SourceHealth] = []
    canonical_parts: list[pd.DataFrame] = []

    source_specs = [
        ("drift_live", fetch_market_stats, "drift"),
        ("jupiter_lend", fetch_earn_tokens, "jupiter_lend"),
        ("kamino", fetch_borrow_and_staking_history, "kamino"),
    ]

    for source_name, fetcher, venue in source_specs:
        canonical_df, health = fetch_validate_and_canonicalize(
            source=source_name,
            fetcher=fetcher,
            venue=venue,
        )
        canonical_parts.append(canonical_df)
        health_rows.append(health)

    print_health_report(health_rows)

    combined = pd.concat(canonical_parts, ignore_index=True)
    assert_cross_venue_coverage(combined)

    written = write_rows(combined)
    if written < 10:
        raise DataQualityError(f"Too few rows written: {written}")

    print(f"\nWrote {written} rows successfully.")


if __name__ == "__main__":
    main()
