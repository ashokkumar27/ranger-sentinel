from __future__ import annotations

import numpy as np
import pandas as pd



def apy_to_daily(apy: float) -> float:
    if apy is None or pd.isna(apy):
        return 0.0
    return (1.0 + max(float(apy), -0.9999)) ** (1.0 / 365.0) - 1.0



def replay_from_decisions(decisions: pd.DataFrame, allocations: pd.DataFrame, initial_nav: float = 100_000.0) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame()

    nav = initial_nav
    rows: list[dict] = []
    allocations = allocations.copy()
    allocations["date"] = pd.to_datetime(allocations["date"], utc=True, errors="coerce")

    for _, d in decisions.sort_values("date").iterrows():
        day_allocs = allocations[allocations["date"] == pd.to_datetime(d["date"], utc=True)]
        day_return = 0.0
        turnover_proxy = 0.0
        sleeve_attr = {"base": 0.0, "alpha": 0.0, "reserve": 0.0}
        for _, a in day_allocs.iterrows():
            daily_r = apy_to_daily(float(a.get("expected_net_apy", 0.0)))
            contrib = float(a.get("target_weight", 0.0)) * daily_r
            day_return += contrib
            sleeve_attr[str(a.get("decision_bucket", "reserve"))] += contrib
            turnover_proxy += float(a.get("target_weight", 0.0)) * float(a.get("conviction_score", 0.0)) * 0.00005

        nav_start = nav
        nav = nav * (1.0 + day_return - turnover_proxy)
        rows.append(
            {
                "date": pd.to_datetime(d["date"], utc=True),
                "nav_start": nav_start,
                "nav_end": nav,
                "gross_return": day_return,
                "turnover_proxy": turnover_proxy,
                "net_return": (nav / nav_start) - 1.0,
                "base_attr": sleeve_attr["base"],
                "alpha_attr": sleeve_attr["alpha"],
                "reserve_attr": sleeve_attr["reserve"],
                "alpha_gate_pass": bool(d["alpha_gate_pass"]),
                "gate_fail_reason": d.get("gate_fail_reason", ""),
                "expected_net_apy": float(d["expected_net_apy"]),
            }
        )
    return pd.DataFrame(rows)



def summarize_performance(replay: pd.DataFrame, hurdle_apy: float = 0.10) -> dict:
    if replay.empty:
        return {}

    x = replay.sort_values("date").copy()
    total_return = float(x["nav_end"].iloc[-1] / x["nav_start"].iloc[0] - 1.0)
    n = len(x)
    annualized = (1.0 + total_return) ** (365.0 / max(n, 1)) - 1.0
    running_peak = x["nav_end"].cummax()
    drawdown = (x["nav_end"] / running_peak) - 1.0
    rolling_90d = x["nav_end"].pct_change(90)
    hurdle_90d = (1.0 + hurdle_apy) ** (90.0 / 365.0) - 1.0

    return {
        "days": int(n),
        "total_return": total_return,
        "annualized_return": annualized,
        "max_drawdown": float(drawdown.min()),
        "rolling_90d_median": float(rolling_90d.median(skipna=True)),
        "rolling_90d_worst": float(rolling_90d.min(skipna=True)),
        "rolling_90d_hit_rate": float(rolling_90d.ge(hurdle_90d).mean(skipna=True)),
        "alpha_gate_pass_rate": float(x["alpha_gate_pass"].mean()),
        "base_attribution": float(x["base_attr"].sum()),
        "alpha_attribution": float(x["alpha_attr"].sum()),
        "reserve_attribution": float(x["reserve_attr"].sum()),
        "avg_turnover_proxy": float(x["turnover_proxy"].mean()),
    }
