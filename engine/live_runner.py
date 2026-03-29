import pandas as pd
from collectors.drift import fetch_market_stats
from collectors.jupiter_lend import fetch_earn_tokens
from collectors.kamino import fetch_borrow_and_staking_history
from pipelines.normalize import canonicalize
from pipelines.features import build_daily_feature_table
from engine.strategy import target_weights, should_rebalance

def build_live_signal() -> tuple[dict, pd.DataFrame]:
    frames = []
    drift_df = fetch_market_stats()
    if not drift_df.empty:
        frames.append(canonicalize(drift_df, venue="drift"))
    jup_df = fetch_earn_tokens()
    if not jup_df.empty:
        frames.append(canonicalize(jup_df, venue="jupiter_lend"))
    kamino_df = fetch_borrow_and_staking_history()
    if not kamino_df.empty:
        frames.append(canonicalize(kamino_df, venue="kamino"))

    all_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    feat = build_daily_feature_table(all_df)
    if feat.empty:
        return {}, feat
    latest = feat.sort_values("ts").iloc[-1].to_dict()
    signal = {
        "carry_quality_score": float(latest["carry_quality_score"]),
        "liquidity_score": float(latest["liquidity_score"]),
        "volatility_score": float(latest["volatility_score"]),
    }
    return signal, feat

def run_live_recommendation(current_weights: dict | None = None) -> dict:
    current_weights = current_weights or {"base_yield": 0.70, "carry": 0.20, "reserve": 0.10}
    signal, feat = build_live_signal()
    if not signal:
        return {"error": "No live signal available. Check API configuration."}
    target = target_weights(signal)

    return {
        "signal": signal,
        "current_weights": current_weights,
        "target_weights": target,
        "rebalance": should_rebalance(current_weights, target),
        "feature_table_tail": feat.tail(5).to_dict(orient="records"),
    }
