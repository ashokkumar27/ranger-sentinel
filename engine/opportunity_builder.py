from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class OpportunityConfig:
    research_mode: bool = True
    annual_turnover_cost_bps: float = 15.0

    # Soft defaults for sparse historical data
    default_utilization: float = 0.55
    default_available_liquidity_usd: float = 250_000.0
    default_price_usd: float = 1.0

    # Haircuts / adjustments
    base_reserve_drag: float = 0.0
    spread_alpha_haircut: float = 0.01
    funding_alpha_haircut: float = 0.005

    # Filters
    min_base_apy: float = 0.0
    min_spread_alpha: float = 0.0
    min_abs_funding_alpha: float = 0.0001  # annualized-like field in your current model path

    # Drift-specific research filters
    allow_settlement_markets_in_research: bool = False
    drop_unknown_asset_drift: bool = True


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def _normalize_text(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _parse_raw_json(x: Any) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            return json.loads(s)
        except Exception:
            return {}
    return {}


def _nested_get(d: dict[str, Any], path: list[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _coerce_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    out["date"] = out["ts"].dt.floor("D")
    return out


def _extract_drift_fields(row: pd.Series) -> dict[str, Any]:
    raw = _parse_raw_json(row.get("raw_json"))
    markets = raw.get("markets", {}) if isinstance(raw, dict) else {}

    symbol = _normalize_text(_nested_get(raw, ["markets", "symbol"])) or _normalize_text(row.get("market"))
    base_asset = _normalize_text(_nested_get(raw, ["markets", "baseAsset"]))
    quote_asset = _normalize_text(_nested_get(raw, ["markets", "quoteAsset"]))
    status = _normalize_text(_nested_get(raw, ["markets", "status"])) or "active"
    ui_status = _normalize_text(_nested_get(raw, ["markets", "uiStatus"]))

    oracle_price = _safe_float(_nested_get(raw, ["markets", "oraclePrice"]))
    price = _safe_float(_nested_get(raw, ["markets", "price"]))
    mark_price = _safe_float(_nested_get(raw, ["markets", "markPrice"]))

    funding_24h = _safe_float(_nested_get(raw, ["markets", "fundingRate24h"]))
    funding_long = _safe_float(_nested_get(raw, ["markets", "fundingRate", "long"]))
    funding_short = _safe_float(_nested_get(raw, ["markets", "fundingRate", "short"]))

    # Prefer richer 24h field if available, else fallback to long-side funding
    funding = funding_24h
    if funding is None:
        funding = funding_long
    if funding is None:
        funding = _safe_float(row.get("funding_rate_daily"))

    market = symbol or _normalize_text(row.get("market")) or "unknown_market"
    asset = base_asset or _normalize_text(row.get("asset")) or "UNKNOWN_ASSET"

    if asset == "unknown_asset":
        asset = "UNKNOWN_ASSET"

    out = {
        "market": market,
        "asset": asset,
        "quote_asset": quote_asset or "USDC",
        "market_status": status or "active",
        "ui_status": ui_status,
        "price_usd": oracle_price or price or mark_price,
        "funding_rate_daily": funding if funding is not None else 0.0,
        "oracle_price": oracle_price,
        "mark_price": mark_price,
        "raw_json_parsed": raw,
    }
    return out


def _backfill_common_fields(row: pd.Series, cfg: OpportunityConfig) -> dict[str, Any]:
    protocol = _normalize_text(row.get("protocol")) or "unknown_protocol"
    venue = _normalize_text(row.get("venue")) or protocol
    market = _normalize_text(row.get("market")) or "unknown_market"
    asset = _normalize_text(row.get("asset")) or "UNKNOWN_ASSET"

    price_usd = _safe_float(row.get("price_usd"))
    utilization = _safe_float(row.get("utilization"))
    available_liquidity_usd = _safe_float(row.get("available_liquidity_usd"))

    if utilization is None:
        utilization = cfg.default_utilization if cfg.research_mode else None
    if available_liquidity_usd is None:
        available_liquidity_usd = cfg.default_available_liquidity_usd if cfg.research_mode else None
    if price_usd is None:
        price_usd = cfg.default_price_usd if cfg.research_mode else None

    return {
        "protocol": protocol,
        "venue": venue,
        "market": market,
        "asset": asset,
        "price_usd": price_usd,
        "utilization": utilization,
        "available_liquidity_usd": available_liquidity_usd,
    }


def _data_quality_pass(
    strategy_type: str,
    supply_apy: float | None,
    borrow_apy: float | None,
    effective_earn_apy: float | None,
    funding_rate_daily: float | None,
    asset: str | None,
) -> bool:
    if strategy_type == "base_lend":
        return (effective_earn_apy is not None and effective_earn_apy > 0) or (supply_apy is not None and supply_apy > 0)
    if strategy_type == "spread_alpha":
        return supply_apy is not None and borrow_apy is not None and supply_apy > borrow_apy
    if strategy_type == "funding_alpha":
        return asset not in (None, "", "UNKNOWN_ASSET") and funding_rate_daily is not None and abs(funding_rate_daily) > 0
    return False


def _withdraw_friction_score(liquidity_usd: float | None, cfg: OpportunityConfig) -> float:
    if liquidity_usd is None:
        return 0.45 if cfg.research_mode else 0.8
    if liquidity_usd >= 10_000_000:
        return 0.05
    if liquidity_usd >= 1_000_000:
        return 0.15
    if liquidity_usd >= 250_000:
        return 0.30
    if liquidity_usd >= 50_000:
        return 0.50
    return 0.75


def _oracle_stress_score(price_usd: float | None, raw_json_parsed: dict[str, Any] | None = None) -> float:
    if raw_json_parsed:
        oracle_price = _safe_float(_nested_get(raw_json_parsed, ["markets", "oraclePrice"]))
        mark_price = _safe_float(_nested_get(raw_json_parsed, ["markets", "markPrice"]))
        if oracle_price and mark_price and oracle_price > 0:
            return min(abs(mark_price - oracle_price) / oracle_price, 1.0)
    if price_usd is None:
        return 0.35
    return 0.0


def _expected_base_net(effective_earn_apy: float, cfg: OpportunityConfig) -> float:
    return float(effective_earn_apy) - (cfg.annual_turnover_cost_bps / 10_000.0) - cfg.base_reserve_drag


def _expected_spread_net(supply_apy: float, borrow_apy: float, cfg: OpportunityConfig) -> float:
    spread = float(supply_apy) - float(borrow_apy)
    return spread - cfg.spread_alpha_haircut - (cfg.annual_turnover_cost_bps / 10_000.0)


def _expected_funding_net(funding_rate_daily: float, cfg: OpportunityConfig) -> float:
    # Current data path behaves like an annualized-ish funding input in your feature model.
    # Keep it simple and conservative.
    haircut = cfg.funding_alpha_haircut
    turnover = cfg.annual_turnover_cost_bps / 10_000.0
    return abs(float(funding_rate_daily)) - haircut - turnover


def _make_opportunity_row(
    *,
    row: pd.Series,
    cfg: OpportunityConfig,
    strategy_type: str,
    protocol: str,
    venue: str,
    market: str,
    asset: str,
    supply_apy: float | None,
    borrow_apy: float | None,
    effective_earn_apy: float | None,
    funding_rate_daily: float | None,
    utilization: float | None,
    available_liquidity_usd: float | None,
    price_usd: float | None,
    market_status: str,
    raw_json_parsed: dict[str, Any] | None,
) -> dict[str, Any]:
    turnover_cost_bps = cfg.annual_turnover_cost_bps
    withdraw_friction = _withdraw_friction_score(available_liquidity_usd, cfg)
    oracle_stress = _oracle_stress_score(price_usd, raw_json_parsed)

    if strategy_type == "base_lend":
        gross = float(effective_earn_apy or supply_apy or 0.0)
        net = _expected_base_net(gross, cfg)
        reserve_candidate_apy = gross
    elif strategy_type == "spread_alpha":
        gross = float((supply_apy or 0.0) - (borrow_apy or 0.0))
        net = _expected_spread_net(float(supply_apy or 0.0), float(borrow_apy or 0.0), cfg)
        reserve_candidate_apy = 0.0
    elif strategy_type == "funding_alpha":
        gross = abs(float(funding_rate_daily or 0.0))
        net = _expected_funding_net(float(funding_rate_daily or 0.0), cfg)
        reserve_candidate_apy = 0.0
    else:
        gross = 0.0
        net = -(turnover_cost_bps / 10_000.0)
        reserve_candidate_apy = 0.0

    data_quality = _data_quality_pass(
        strategy_type=strategy_type,
        supply_apy=supply_apy,
        borrow_apy=borrow_apy,
        effective_earn_apy=effective_earn_apy,
        funding_rate_daily=funding_rate_daily,
        asset=asset,
    )

    return {
        "ts": row["ts"],
        "date": row["date"],
        "protocol": protocol,
        "venue": venue,
        "market": market,
        "asset": asset,
        "strategy_type": strategy_type,
        "supply_apy": supply_apy,
        "borrow_apy": borrow_apy,
        "effective_earn_apy": effective_earn_apy,
        "funding_rate_daily": funding_rate_daily,
        "utilization": utilization,
        "available_liquidity_usd": available_liquidity_usd,
        "price_usd": price_usd,
        "expected_gross_apy": gross,
        "expected_net_apy": net,
        "turnover_cost_bps": turnover_cost_bps,
        "withdraw_friction_score": withdraw_friction,
        "oracle_stress_score": oracle_stress,
        "market_status": market_status,
        "data_quality_pass": data_quality,
        "banned_exposure_flag": False,
        "asset_peer_median_apy": np.nan,  # backfilled later
        "richness_apy": np.nan,           # backfilled later
        "exit_quality_base": np.nan,      # backfilled later
        "reserve_candidate_apy": reserve_candidate_apy,
    }


def build_opportunity_table(
    snapshots: pd.DataFrame,
    cfg: OpportunityConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or OpportunityConfig()

    x = snapshots.copy()
    if x.empty:
        return pd.DataFrame()

    x = _coerce_ts(x)

    rows: list[dict[str, Any]] = []

    for _, row in x.iterrows():
        protocol = _normalize_text(row.get("protocol")) or "unknown_protocol"

        common = _backfill_common_fields(row, cfg)
        market_status = "active"
        raw_json_parsed: dict[str, Any] | None = _parse_raw_json(row.get("raw_json"))

        protocol_out = common["protocol"]
        venue_out = common["venue"]
        market_out = common["market"]
        asset_out = common["asset"]
        price_usd = common["price_usd"]
        utilization = common["utilization"]
        available_liquidity_usd = common["available_liquidity_usd"]

        supply_apy = _safe_float(row.get("deposit_apy"))
        borrow_apy = _safe_float(row.get("borrow_apy"))
        effective_earn_apy = _safe_float(row.get("effective_earn_apy"))
        funding_rate_daily = _safe_float(row.get("funding_rate_daily"))

        if effective_earn_apy is None:
            effective_earn_apy = supply_apy if supply_apy is not None else 0.0

        if protocol == "drift":
            d = _extract_drift_fields(row)
            market_out = d["market"] or market_out
            asset_out = d["asset"] or asset_out
            market_status = d["market_status"] or "active"
            funding_rate_daily = d["funding_rate_daily"]
            price_usd = d["price_usd"] if d["price_usd"] is not None else price_usd
            raw_json_parsed = d["raw_json_parsed"]

            if cfg.drop_unknown_asset_drift and asset_out == "UNKNOWN_ASSET":
                continue

            if not cfg.allow_settlement_markets_in_research and market_status == "settlement":
                continue

        # 1) base lend opportunity
        if (effective_earn_apy or 0.0) > cfg.min_base_apy:
            rows.append(
                _make_opportunity_row(
                    row=row,
                    cfg=cfg,
                    strategy_type="base_lend",
                    protocol=protocol_out,
                    venue=venue_out,
                    market=market_out,
                    asset=asset_out,
                    supply_apy=supply_apy,
                    borrow_apy=borrow_apy,
                    effective_earn_apy=effective_earn_apy,
                    funding_rate_daily=funding_rate_daily,
                    utilization=utilization,
                    available_liquidity_usd=available_liquidity_usd,
                    price_usd=price_usd,
                    market_status=market_status,
                    raw_json_parsed=raw_json_parsed,
                )
            )

        # 2) spread alpha opportunity
        if supply_apy is not None and borrow_apy is not None and (supply_apy - borrow_apy) > cfg.min_spread_alpha:
            rows.append(
                _make_opportunity_row(
                    row=row,
                    cfg=cfg,
                    strategy_type="spread_alpha",
                    protocol=protocol_out,
                    venue=venue_out,
                    market=market_out,
                    asset=asset_out,
                    supply_apy=supply_apy,
                    borrow_apy=borrow_apy,
                    effective_earn_apy=effective_earn_apy,
                    funding_rate_daily=funding_rate_daily,
                    utilization=utilization,
                    available_liquidity_usd=available_liquidity_usd,
                    price_usd=price_usd,
                    market_status=market_status,
                    raw_json_parsed=raw_json_parsed,
                )
            )

        # 3) funding alpha opportunity
        if protocol == "drift":
            if market_status == "active" and funding_rate_daily is not None and abs(funding_rate_daily) > cfg.min_abs_funding_alpha:
                rows.append(
                    _make_opportunity_row(
                        row=row,
                        cfg=cfg,
                        strategy_type="funding_alpha",
                        protocol=protocol_out,
                        venue=venue_out,
                        market=market_out,
                        asset=asset_out,
                        supply_apy=supply_apy,
                        borrow_apy=borrow_apy,
                        effective_earn_apy=effective_earn_apy,
                        funding_rate_daily=funding_rate_daily,
                        utilization=utilization,
                        available_liquidity_usd=available_liquidity_usd,
                        price_usd=price_usd,
                        market_status=market_status,
                        raw_json_parsed=raw_json_parsed,
                    )
                )

    opps = pd.DataFrame(rows)
    if opps.empty:
        return opps

    # Fill peer richness and exit-quality helpers
    opps["asset_peer_median_apy"] = opps.groupby(["date", "asset"])["expected_gross_apy"].transform("median")
    opps["asset_peer_median_apy"] = opps["asset_peer_median_apy"].fillna(0.0)
    opps["richness_apy"] = opps["expected_gross_apy"] - opps["asset_peer_median_apy"]

    liq_component = np.clip(np.log1p(opps["available_liquidity_usd"].fillna(cfg.default_available_liquidity_usd)) / np.log(10_000_001), 0.0, 1.0)
    util_component = 1.0 - np.clip(opps["utilization"].fillna(cfg.default_utilization), 0.0, 1.0)
    friction_component = 1.0 - np.clip(opps["withdraw_friction_score"].fillna(0.5), 0.0, 1.0)

    opps["exit_quality_base"] = 0.45 * liq_component + 0.35 * util_component + 0.20 * friction_component

    opps = opps.sort_values(["date", "protocol", "venue", "market", "asset", "strategy_type", "ts"]).reset_index(drop=True)
    return opps
