from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class OpportunityConfig:
    reserve_target_weight: float = 0.10
    risk_free_reserve_apy: float = 0.0
    default_turnover_cost_bps: float = 15.0
    default_withdraw_friction_score: float = 0.25
    default_oracle_stress_score: float = 0.10
    utilization_soft_cap: float = 0.85
    liquidity_soft_floor_usd: float = 500_000.0
    min_effective_base_apy: float = 0.0
    spread_alpha_weight: float = 0.60
    base_net_haircut_bps: float = 10.0
    funding_stress_haircut_bps: float = 40.0



def _safe_num(series: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)



def _norm_high(x: pd.Series, floor: float, cap: float) -> pd.Series:
    denom = max(cap - floor, 1e-9)
    return ((x - floor) / denom).clip(lower=0.0, upper=1.0)



def _norm_low(x: pd.Series, floor: float, cap: float) -> pd.Series:
    denom = max(cap - floor, 1e-9)
    return (1.0 - ((x - floor) / denom)).clip(lower=0.0, upper=1.0)



def _effective_base_apy(row: pd.Series) -> float:
    candidates = [row.get("deposit_apy"), row.get("effective_earn_apy")]
    vals = [float(v) for v in candidates if v is not None and not pd.isna(v)]
    return max(vals) if vals else 0.0



def build_opportunity_table(snapshots: pd.DataFrame, cfg: OpportunityConfig | None = None) -> pd.DataFrame:
    cfg = cfg or OpportunityConfig()
    x = snapshots.copy()
    if x.empty:
        return x

    x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
    x = x.dropna(subset=["ts"]).sort_values(["ts", "protocol", "market", "asset"])

    for c in [
        "deposit_apy",
        "borrow_apy",
        "funding_rate_daily",
        "available_liquidity_usd",
        "utilization",
        "price_usd",
        "effective_earn_apy",
    ]:
        if c not in x.columns:
            x[c] = np.nan
        x[c] = _safe_num(x[c])

    x["date"] = x["ts"].dt.floor("D")
    x["asset"] = x["asset"].fillna("UNKNOWN").astype(str).str.upper()
    x["venue"] = x["venue"].fillna(x["protocol"]).astype(str)

    x["effective_earn_apy"] = x.apply(_effective_base_apy, axis=1)
    x["strategy_type"] = "base_lend"
    x.loc[x["funding_rate_daily"].notna() & (x["funding_rate_daily"].abs() > 0), "strategy_type"] = "funding_alpha"
    x.loc[
        x["deposit_apy"].notna() & x["borrow_apy"].notna() & (x["deposit_apy"] > x["borrow_apy"]),
        "strategy_type",
    ] = "spread_alpha"

    x["supply_apy"] = x["deposit_apy"]
    x["turnover_cost_bps"] = cfg.default_turnover_cost_bps
    x.loc[x["strategy_type"] == "funding_alpha", "turnover_cost_bps"] = cfg.default_turnover_cost_bps + 15.0
    x.loc[x["strategy_type"] == "spread_alpha", "turnover_cost_bps"] = cfg.default_turnover_cost_bps + 5.0

    x["withdraw_friction_score"] = cfg.default_withdraw_friction_score
    if x["available_liquidity_usd"].notna().any():
        x["withdraw_friction_score"] = 1.0 - _norm_high(
            x["available_liquidity_usd"],
            floor=cfg.liquidity_soft_floor_usd,
            cap=max(cfg.liquidity_soft_floor_usd * 10.0, float(x["available_liquidity_usd"].max())),
        )
    x["withdraw_friction_score"] = x["withdraw_friction_score"].clip(0.0, 1.0)

    x["oracle_stress_score"] = cfg.default_oracle_stress_score
    if x["price_usd"].notna().any():
        realized = (
            x.groupby(["protocol", "market", "asset"], dropna=False)["price_usd"]
            .pct_change(fill_method=None)
            .abs()
            .fillna(0.0)
        )
        x["oracle_stress_score"] = _norm_high(realized, floor=0.0, cap=0.08)

    x["data_quality_pass"] = True
    x.loc[x["effective_earn_apy"].isna(), "data_quality_pass"] = False
    x.loc[x["available_liquidity_usd"].isna(), "data_quality_pass"] = False

    x["banned_exposure_flag"] = False
    x.loc[x["strategy_type"].eq("dex_lp"), "banned_exposure_flag"] = True

    x["market_status"] = "active"

    x["base_net_apy"] = (
        x["effective_earn_apy"].fillna(0.0)
        - (x["turnover_cost_bps"] / 10_000.0)
        - (cfg.base_net_haircut_bps / 10_000.0)
    )

    spread = (x["deposit_apy"] - x["borrow_apy"]).fillna(0.0)
    x["spread_gross_apy"] = spread.clip(lower=0.0)
    x["spread_net_apy"] = x["spread_gross_apy"] - (x["turnover_cost_bps"] / 10_000.0)

    funding_alpha = x["funding_rate_daily"].fillna(0.0)
    funding_haircut = (cfg.funding_stress_haircut_bps / 10_000.0) + 0.20 * x["oracle_stress_score"]
    x["funding_net_apy"] = funding_alpha - (x["turnover_cost_bps"] / 10_000.0) - funding_haircut

    x["expected_gross_apy"] = x["effective_earn_apy"].fillna(0.0)
    x["expected_net_apy"] = x["base_net_apy"]
    x.loc[x["strategy_type"] == "spread_alpha", "expected_gross_apy"] = x["spread_gross_apy"]
    x.loc[x["strategy_type"] == "spread_alpha", "expected_net_apy"] = x["spread_net_apy"]
    x.loc[x["strategy_type"] == "funding_alpha", "expected_gross_apy"] = x["funding_rate_daily"].fillna(0.0)
    x.loc[x["strategy_type"] == "funding_alpha", "expected_net_apy"] = x["funding_net_apy"]

    x["reserve_candidate_apy"] = cfg.risk_free_reserve_apy
    x.loc[x["strategy_type"] == "base_lend", "reserve_candidate_apy"] = x["base_net_apy"].clip(lower=0.0)

    asset_peer_daily = (
        x.groupby(["date", "asset"], dropna=False)["effective_earn_apy"].median().rename("asset_peer_median_apy")
    )
    x = x.merge(asset_peer_daily.reset_index(), on=["date", "asset"], how="left")
    x["richness_apy"] = (x["effective_earn_apy"] - x["asset_peer_median_apy"]).fillna(0.0)

    util_penalty = _norm_high(x["utilization"].fillna(0.0), floor=cfg.utilization_soft_cap, cap=1.0)
    liq_score = _norm_high(
        x["available_liquidity_usd"].fillna(0.0),
        floor=cfg.liquidity_soft_floor_usd,
        cap=max(cfg.liquidity_soft_floor_usd * 10.0, float(x["available_liquidity_usd"].fillna(0).max() or cfg.liquidity_soft_floor_usd * 10.0)),
    )
    x["exit_quality_base"] = (0.55 * liq_score + 0.25 * (1.0 - util_penalty) + 0.20 * (1.0 - x["withdraw_friction_score"]))
    x["exit_quality_base"] = x["exit_quality_base"].clip(0.0, 1.0)

    cols = [
        "ts",
        "date",
        "protocol",
        "venue",
        "market",
        "asset",
        "strategy_type",
        "supply_apy",
        "borrow_apy",
        "effective_earn_apy",
        "funding_rate_daily",
        "utilization",
        "available_liquidity_usd",
        "price_usd",
        "expected_gross_apy",
        "expected_net_apy",
        "turnover_cost_bps",
        "withdraw_friction_score",
        "oracle_stress_score",
        "market_status",
        "data_quality_pass",
        "banned_exposure_flag",
        "asset_peer_median_apy",
        "richness_apy",
        "exit_quality_base",
        "reserve_candidate_apy",
    ]
    return x[cols].sort_values(["date", "asset", "strategy_type", "protocol"]).reset_index(drop=True)
