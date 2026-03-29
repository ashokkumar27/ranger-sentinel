from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import math
import pandas as pd


REQUIRED_COLUMNS = {
    "ts",
    "protocol",
    "venue",
    "market",
    "asset",
    "strategy_type",
    "expected_net_apy",
    "available_liquidity_usd",
}


@dataclass(slots=True)
class FirewallConfig:
    min_liquidity_usd: float = 100_000.0
    max_utilization: float = 0.92
    max_oracle_stress_score: float = 0.60
    max_withdraw_friction_score: float = 0.70
    max_turnover_cost_bps: float = 150.0
    allow_missing_borrow_for_base_lend: bool = True
    banned_strategy_types: tuple[str, ...] = (
        "recursive_ybs",
        "junior_tranche",
        "insurance_pool",
        "dex_lp",
        "unsafe_loop",
    )
    banned_protocol_keywords: tuple[str, ...] = (
        "lp",
        "pool",
        "junior",
        "tranche",
    )
    approved_strategy_types: tuple[str, ...] = (
        "base_lend",
        "spread_alpha",
        "funding_alpha",
        "reserve",
    )


@dataclass(slots=True)
class FirewallDecision:
    passed: bool
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "reasons": self.reasons}



def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False



def _contains_banned_keyword(text: str | None, keywords: Iterable[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(k in lowered for k in keywords)



def evaluate_row(row: pd.Series, cfg: FirewallConfig | None = None) -> FirewallDecision:
    cfg = cfg or FirewallConfig()
    reasons: list[str] = []

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in row.index]
    if missing_columns:
        return FirewallDecision(False, [f"missing_columns:{','.join(sorted(missing_columns))}"])

    strategy_type = str(row.get("strategy_type") or "")
    protocol = str(row.get("protocol") or "")
    venue = str(row.get("venue") or "")

    if strategy_type not in cfg.approved_strategy_types:
        reasons.append(f"unsupported_strategy_type:{strategy_type}")

    if strategy_type in cfg.banned_strategy_types:
        reasons.append(f"banned_strategy_type:{strategy_type}")

    if bool(row.get("banned_exposure_flag", False)):
        reasons.append("banned_exposure_flag")

    if _contains_banned_keyword(protocol, cfg.banned_protocol_keywords) or _contains_banned_keyword(
        venue, cfg.banned_protocol_keywords
    ):
        reasons.append("banned_protocol_keyword")

    market_status = str(row.get("market_status") or "active").lower()
    if market_status not in {"active", "ok", "open", "live", "unknown"}:
        reasons.append(f"market_status:{market_status}")

    data_quality_pass = row.get("data_quality_pass")
    if data_quality_pass is False:
        reasons.append("data_quality_fail")

    available_liquidity_usd = row.get("available_liquidity_usd")
    if _is_missing(available_liquidity_usd) or float(available_liquidity_usd) < cfg.min_liquidity_usd:
        reasons.append("liquidity_below_floor")

    utilization = row.get("utilization")
    if not _is_missing(utilization) and float(utilization) > cfg.max_utilization:
        reasons.append("utilization_too_high")

    oracle_stress_score = row.get("oracle_stress_score")
    if not _is_missing(oracle_stress_score) and float(oracle_stress_score) > cfg.max_oracle_stress_score:
        reasons.append("oracle_stress_too_high")

    withdraw_friction_score = row.get("withdraw_friction_score")
    if not _is_missing(withdraw_friction_score) and float(withdraw_friction_score) > cfg.max_withdraw_friction_score:
        reasons.append("withdraw_friction_too_high")

    turnover_cost_bps = row.get("turnover_cost_bps")
    if not _is_missing(turnover_cost_bps) and float(turnover_cost_bps) > cfg.max_turnover_cost_bps:
        reasons.append("turnover_cost_too_high")

    expected_net_apy = row.get("expected_net_apy")
    if _is_missing(expected_net_apy) or not math.isfinite(float(expected_net_apy)):
        reasons.append("invalid_expected_net_apy")

    if strategy_type in {"spread_alpha", "funding_alpha"}:
        if _is_missing(row.get("turnover_cost_bps")):
            reasons.append("missing_turnover_cost")

    if strategy_type == "spread_alpha":
        if _is_missing(row.get("borrow_apy")):
            reasons.append("missing_borrow_apy")
        if _is_missing(row.get("supply_apy")):
            reasons.append("missing_supply_apy")

    if strategy_type == "base_lend" and not cfg.allow_missing_borrow_for_base_lend:
        if _is_missing(row.get("borrow_apy")):
            reasons.append("missing_borrow_apy")

    if strategy_type == "funding_alpha":
        if _is_missing(row.get("funding_rate_daily")):
            reasons.append("missing_funding_rate")

    return FirewallDecision(passed=not reasons, reasons=reasons)



def apply_firewall(df: pd.DataFrame, cfg: FirewallConfig | None = None) -> pd.DataFrame:
    cfg = cfg or FirewallConfig()
    x = df.copy()
    decisions = x.apply(lambda row: evaluate_row(row, cfg), axis=1)
    x["policy_pass"] = decisions.map(lambda d: d.passed)
    x["policy_fail_reasons"] = decisions.map(lambda d: ";".join(d.reasons))
    return x
