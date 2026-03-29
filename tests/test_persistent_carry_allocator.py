from __future__ import annotations

import pandas as pd

from engine.allocator_persistent_carry_v1 import allocate_daily_decisions
from engine.opportunity_builder import build_opportunity_table
from pipelines.feature_model_v2 import build_feature_model_v2


def _sample_snapshots() -> pd.DataFrame:
    rows = []
    for day in pd.date_range("2026-01-01", periods=35, tz="UTC"):
        rows.extend(
            [
                {
                    "ts": day,
                    "protocol": "jupiter_lend",
                    "venue": "jupiter",
                    "market": "earn-usdc",
                    "asset": "USDC",
                    "deposit_apy": 0.12,
                    "borrow_apy": None,
                    "funding_rate_daily": None,
                    "utilization": 0.52,
                    "available_liquidity_usd": 4_000_000,
                    "price_usd": 1.0,
                    "effective_earn_apy": 0.12,
                },
                {
                    "ts": day,
                    "protocol": "kamino",
                    "venue": "kamino",
                    "market": "lend-usdc",
                    "asset": "USDC",
                    "deposit_apy": 0.145,
                    "borrow_apy": 0.07,
                    "funding_rate_daily": None,
                    "utilization": 0.58,
                    "available_liquidity_usd": 2_800_000,
                    "price_usd": 1.0,
                    "effective_earn_apy": 0.145,
                },
                {
                    "ts": day,
                    "protocol": "drift",
                    "venue": "drift",
                    "market": "SOL-PERP",
                    "asset": "SOL",
                    "deposit_apy": 0.05,
                    "borrow_apy": 0.02,
                    "funding_rate_daily": 0.18,
                    "utilization": 0.48,
                    "available_liquidity_usd": 1_700_000,
                    "price_usd": 130 + (day.day % 5),
                    "effective_earn_apy": 0.05,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_daily_decisions_shape() -> None:
    snapshots = _sample_snapshots()
    opps = build_opportunity_table(snapshots)
    feats = build_feature_model_v2(opps)
    decisions, allocations = allocate_daily_decisions(feats)
    assert not decisions.empty
    assert not allocations.empty
    assert {"base_weight", "alpha_weight", "reserve_weight", "alpha_gate_pass"}.issubset(decisions.columns)


def test_weights_sum_to_one() -> None:
    snapshots = _sample_snapshots()
    opps = build_opportunity_table(snapshots)
    feats = build_feature_model_v2(opps)
    decisions, _ = allocate_daily_decisions(feats)
    total = decisions[["base_weight", "alpha_weight", "reserve_weight"]].sum(axis=1)
    assert ((total - 1.0).abs() < 1e-9).all()


def test_policy_can_activate_alpha() -> None:
    snapshots = _sample_snapshots()
    opps = build_opportunity_table(snapshots)
    feats = build_feature_model_v2(opps)
    decisions, allocations = allocate_daily_decisions(feats)
    assert decisions["alpha_gate_pass"].any()
    assert (allocations[allocations["decision_bucket"] == "alpha"]["target_weight"] > 0).any()
