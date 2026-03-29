import pandas as pd
import pytest

from main_backfill import (
    DataQualityError,
    _prepare_rows,
    summarize_health,
    validate_source,
    write_rows,
)


def test_write_rows_returns_zero_for_empty_df():
    assert write_rows(pd.DataFrame()) == 0


def test_prepare_rows_drops_bad_ts_and_deduplicates():
    df = pd.DataFrame(
        [
            {
                "ts": "2026-03-29T00:00:00Z",
                "protocol": "kamino",
                "venue": "kamino",
                "market": "m1",
                "asset": "a1",
                "deposit_apy": 0.12,
                "raw_json": {},
            },
            {
                "ts": "bad-ts",
                "protocol": "kamino",
                "venue": "kamino",
                "market": "m1",
                "asset": "a1",
                "deposit_apy": 0.13,
                "raw_json": {},
            },
            {
                "ts": "2026-03-29T00:00:00Z",
                "protocol": "kamino",
                "venue": "kamino",
                "market": "m1",
                "asset": "a1",
                "deposit_apy": 0.14,
                "raw_json": {},
            },
        ]
    )

    out = _prepare_rows(df)
    assert len(out) == 1
    assert float(out.iloc[0]["deposit_apy"]) == 0.14


def test_validate_jupiter_fails_when_deposit_apy_missing():
    raw_df = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp.utcnow(),
                "protocol": "jupiter_lend",
                "venue": "jupiter_lend",
                "market": "earn",
                "asset": "USDC",
                "deposit_apy": None,
                "borrow_apy": None,
                "funding_rate_daily": None,
                "available_liquidity_usd": 1000,
                "price_usd": 1.0,
                "raw_json": {"assetSymbol": "USDC"},
            }
        ]
    )

    health = summarize_health("jupiter_lend", raw_df, raw_df)

    with pytest.raises(DataQualityError, match="deposit_apy"):
        validate_source("jupiter_lend", raw_df, health)


def test_validate_drift_hist_fails_when_funding_missing():
    raw_df = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp.utcnow(),
                "protocol": "drift",
                "venue": "drift",
                "market": "SOL-PERP",
                "asset": "SOL",
                "deposit_apy": None,
                "borrow_apy": None,
                "funding_rate_daily": None,
                "available_liquidity_usd": None,
                "price_usd": 140.0,
                "raw_json": {"symbol": "SOL-PERP"},
            }
        ]
    )

    health = summarize_health("drift_hist", raw_df, raw_df)

    with pytest.raises(DataQualityError, match="funding_rate_daily"):
        validate_source("drift_hist", raw_df, health)


def test_validate_kamino_fails_when_both_apys_missing():
    raw_df = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp.utcnow(),
                "protocol": "kamino",
                "venue": "kamino",
                "market": "market_pubkey",
                "asset": "reserve_pubkey",
                "deposit_apy": None,
                "borrow_apy": None,
                "funding_rate_daily": None,
                "available_liquidity_usd": None,
                "price_usd": None,
                "raw_json": {"reserve": "reserve_pubkey"},
            }
        ]
    )

    health = summarize_health("kamino", raw_df, raw_df)

    with pytest.raises(DataQualityError, match="both deposit_apy and borrow_apy"):
        validate_source("kamino", raw_df, health)


def test_validate_jupiter_passes_when_minimum_fields_exist():
    raw_df = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp.utcnow(),
                "protocol": "jupiter_lend",
                "venue": "jupiter_lend",
                "market": "earn",
                "asset": "USDC",
                "deposit_apy": 0.08,
                "borrow_apy": None,
                "funding_rate_daily": None,
                "available_liquidity_usd": 500000,
                "price_usd": 1.0,
                "raw_json": {"assetSymbol": "USDC", "apy": 0.08},
            }
        ]
    )

    health = summarize_health("jupiter_lend", raw_df, raw_df)
    validate_source("jupiter_lend", raw_df, health)
