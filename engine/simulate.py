import pandas as pd
from engine.strategy import target_weights, should_rebalance

def run_daily_replay(signals_df: pd.DataFrame, initial_nav: float = 100_000.0, policy_version: str = "adaptive_v1") -> pd.DataFrame:
    nav = initial_nav
    current = {"base_yield": 0.70, "carry": 0.20, "reserve": 0.10}
    rows = []
    if signals_df.empty:
        return pd.DataFrame(rows)

    for _, row in signals_df.sort_values("ts").iterrows():
        signal = {
            "carry_quality_score": float(row["carry_quality_score"]),
            "liquidity_score": float(row["liquidity_score"]),
            "volatility_score": float(row["volatility_score"]),
        }
        target = target_weights(signal)
        rebalance = should_rebalance(current, target)
        if len(rows) < 10:
            print("DEBUG", row["ts"], signal["carry_quality_score"], current, target, rebalance)

        rebalance_cost = 0.0
        if rebalance:
            shift = sum(abs(current[k] - target[k]) for k in current)
            rebalance_cost = nav * shift * 0.0005
            current = target

        base_r = float(row["base_return_daily"])
        carry_r = float(row["carry_return_daily"])
        gross_r = current["base_yield"] * base_r + current["carry"] * carry_r

        nav_start = nav
        nav = nav * (1 + gross_r) - rebalance_cost
        rows.append({
            "ts": row["ts"],
            "policy_version": policy_version,
            "nav_start": nav_start,
            "nav_end": nav,
            "base_weight": current["base_yield"],
            "carry_weight": current["carry"],
            "reserve_weight": current["reserve"],
            "rebalance_cost_usd": rebalance_cost,
            "gross_return": gross_r,
            "net_return": (nav / nav_start) - 1,
        })
    return pd.DataFrame(rows)

def run_static_baseline(signals_df: pd.DataFrame, initial_nav: float = 100_000.0, policy_version: str = "static_70_20_10") -> pd.DataFrame:
    nav = initial_nav
    current = {"base_yield": 0.70, "carry": 0.20, "reserve": 0.10}
    rows = []
    if signals_df.empty:
        return pd.DataFrame(rows)

    for _, row in signals_df.sort_values("ts").iterrows():
        base_r = float(row["base_return_daily"])
        carry_r = float(row["carry_return_daily"])
        gross_r = current["base_yield"] * base_r + current["carry"] * carry_r
        nav_start = nav
        nav = nav * (1 + gross_r)

        rows.append({
            "ts": row["ts"],
            "policy_version": policy_version,
            "nav_start": nav_start,
            "nav_end": nav,
            "base_weight": current["base_yield"],
            "carry_weight": current["carry"],
            "reserve_weight": current["reserve"],
            "rebalance_cost_usd": 0.0,
            "gross_return": gross_r,
            "net_return": (nav / nav_start) - 1,
        })
    return pd.DataFrame(rows)
