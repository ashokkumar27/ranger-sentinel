from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.policy_firewall import apply_firewall, FirewallConfig


@dataclass(slots=True)
class AllocationConfig:
    base_weight_min: float = 0.70
    base_weight_max: float = 0.85
    alpha_weight_min: float = 0.10
    alpha_weight_max: float = 0.25
    reserve_weight_min: float = 0.05
    reserve_weight_max: float = 0.10
    reserve_weight_default: float = 0.10
    expected_net_hurdle: float = 0.05
    persistence_floor: float = 0.62
    exit_quality_floor: float = 0.55
    turnover_edge_buffer: float = 0.015
    single_base_cap: float = 0.40
    single_alpha_cap: float = 0.10
    funding_alpha_total_cap: float = 0.15


def _normalize_weights(df: pd.DataFrame, total_weight: float, cap: float, score_col: str) -> pd.DataFrame:
    if df.empty or total_weight <= 0:
        return df.assign(target_weight=0.0)
    x = df.copy()
    raw = x[score_col].clip(lower=0.0).fillna(0.0)
    if raw.sum() <= 0:
        raw = pd.Series(np.ones(len(x)), index=x.index, dtype=float)
    x["target_weight"] = raw / raw.sum() * total_weight
    x["target_weight"] = x["target_weight"].clip(upper=cap)
    if x["target_weight"].sum() > 0 and abs(x["target_weight"].sum() - total_weight) > 1e-9:
        x["target_weight"] *= total_weight / x["target_weight"].sum()
    return x


def _alpha_weight_from_conviction(conviction: float, cfg: AllocationConfig) -> float:
    if conviction <= 0:
        return 0.0
    if conviction < 0.70:
        return cfg.alpha_weight_min
    if conviction < 0.78:
        return 0.15
    if conviction < 0.86:
        return 0.20
    return cfg.alpha_weight_max


def _gate_alpha(candidates: pd.DataFrame, cfg: AllocationConfig) -> tuple[bool, str, float, float, pd.DataFrame]:
    alpha = candidates.copy()

    if alpha.empty:
        return False, "no_alpha_candidates", 0.0, 0.0, alpha

    alpha = alpha[
        alpha["policy_pass"].fillna(False)
        & (~alpha["stress_flag"].fillna(False))
        & (alpha["expected_net_apy"].fillna(0.0) > 0)
    ].copy()

    if alpha.empty:
        return False, "no_alpha_candidates", 0.0, 0.0, alpha

    alpha = alpha.sort_values(
        ["expected_net_apy", "conviction_score"],
        ascending=False,
    )

    top_alpha = alpha.head(3).copy()

    best = top_alpha.iloc[0]
    
    best_expected_net = float(best["expected_net_apy"])
    best_conviction = float(best["conviction_score"])
    turnover_cost = float(best.get("turnover_cost_bps", 0.0)) / 10_000.0
    edge_after_cost = best_expected_net - turnover_cost
    
    if best_expected_net <= cfg.expected_net_hurdle:
        return False, "net_carry_below_hurdle", best_conviction, edge_after_cost, top_alpha
    if float(best["persistence_score"]) <= cfg.persistence_floor:
        return False, "persistence_below_floor", blended_conviction, edge_after_cost, top_alpha
    if float(best["exit_quality_score"]) <= cfg.exit_quality_floor:
        return False, "exit_quality_below_floor", blended_conviction, edge_after_cost, top_alpha
    if edge_after_cost <= cfg.turnover_edge_buffer:
        return False, "edge_not_above_cost", blended_conviction, edge_after_cost, top_alpha

    return True, "pass", best_conviction, edge_after_cost, top_alpha


def allocate_daily_decisions(features: pd.DataFrame, cfg: AllocationConfig | None = None, firewall_cfg: FirewallConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or AllocationConfig()
    firewall_cfg = firewall_cfg or FirewallConfig()
    x = apply_firewall(features, firewall_cfg)
    if x.empty:
        return pd.DataFrame(), x

    decisions: list[dict] = []
    allocations: list[pd.DataFrame] = []

    for date, day in x.groupby("date", dropna=False):
        base_candidates = day[(day["strategy_type"] == "base_lend") & (day["policy_pass"])]
        alpha_candidates = day[(day["strategy_type"].isin(["spread_alpha", "funding_alpha"])) & (day["policy_pass"])]

        alpha_pass, gate_reason, conviction_score, edge_after_cost, selected_alpha = _gate_alpha(alpha_candidates, cfg)
        alpha_weight = _alpha_weight_from_conviction(conviction_score, cfg) if alpha_pass else 0.0

        reserve_weight = cfg.reserve_weight_default
        if alpha_pass and day["exit_quality_score"].max() > 0.80:
            reserve_weight = cfg.reserve_weight_min
        reserve_weight = min(max(reserve_weight, cfg.reserve_weight_min), cfg.reserve_weight_max)

        base_weight = 1.0 - alpha_weight - reserve_weight
        base_weight = min(max(base_weight, cfg.base_weight_min), cfg.base_weight_max)
        total = base_weight + alpha_weight + reserve_weight
        if abs(total - 1.0) > 1e-9:
            base_weight += 1.0 - total

        base_alloc = _normalize_weights(base_candidates, base_weight, cfg.single_base_cap, "expected_net_apy")
        alpha_alloc = _normalize_weights(
            selected_alpha if alpha_pass else alpha_candidates.iloc[0:0],
            alpha_weight,
            cfg.single_alpha_cap,
            "expected_net_apy",
        )

        if not alpha_alloc.empty:
            funding_mask = alpha_alloc["strategy_type"].eq("funding_alpha")
            if funding_mask.any() and alpha_alloc.loc[funding_mask, "target_weight"].sum() > cfg.funding_alpha_total_cap:
                overflow = alpha_alloc.loc[funding_mask, "target_weight"].sum() - cfg.funding_alpha_total_cap
                alpha_alloc.loc[funding_mask, "target_weight"] *= cfg.funding_alpha_total_cap / alpha_alloc.loc[funding_mask, "target_weight"].sum()
                non_funding = alpha_alloc.loc[~funding_mask, "target_weight"].sum()
                if non_funding > 0:
                    alpha_alloc.loc[~funding_mask, "target_weight"] *= (non_funding + overflow) / non_funding

        reserve_row = pd.DataFrame(
            [{
                "date": date,
                "ts": pd.Timestamp(date),
                "protocol": "reserve",
                "venue": "reserve",
                "market": "reserve",
                "asset": "USD",
                "strategy_type": "reserve",
                "target_weight": reserve_weight,
                "expected_net_apy": 0.0,
                "conviction_score": 0.0,
                "policy_pass": True,
                "policy_fail_reasons": "",
            }]
        )

        if not base_alloc.empty:
            allocations.append(base_alloc.assign(decision_bucket="base"))
        if not alpha_alloc.empty:
            allocations.append(alpha_alloc.assign(decision_bucket="alpha"))
        allocations.append(reserve_row.assign(decision_bucket="reserve"))

        expected_net_apy = 0.0
        if not base_alloc.empty:
            expected_net_apy += float((base_alloc["expected_net_apy"] * base_alloc["target_weight"]).sum())
        if not alpha_alloc.empty:
            expected_net_apy += float((alpha_alloc["expected_net_apy"] * alpha_alloc["target_weight"]).sum())

        decisions.append(
            {
                "date": date,
                "alpha_gate_pass": alpha_pass,
                "gate_fail_reason": gate_reason if not alpha_pass else "",
                "base_weight": base_weight,
                "alpha_weight": alpha_weight,
                "reserve_weight": reserve_weight,
                "expected_net_apy": float(edge_after_cost),
                "persistence_score": float(selected_alpha["persistence_score"].max()) if not selected_alpha.empty else 0.0,
                "exit_quality_score": float(day["exit_quality_score"].max()) if not day.empty else 0.0,
                "funding_quality_score": float(selected_alpha["funding_quality_score"].max()) if not selected_alpha.empty else 0.0,
                "conviction_score": conviction_score,
                "edge_after_cost": edge_after_cost,
                "stress_flags": int(day["stress_flag"].fillna(False).sum()),
                "policy_failures": int((~day["policy_pass"].fillna(False)).sum()),
            }
        )

    decisions_df = pd.DataFrame(decisions).sort_values("date").reset_index(drop=True)
    allocations_df = pd.concat(allocations, ignore_index=True).sort_values(["date", "decision_bucket", "target_weight"], ascending=[True, True, False]) if allocations else pd.DataFrame()
    return decisions_df, allocations_df
