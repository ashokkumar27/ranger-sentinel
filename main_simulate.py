from storage.db import init_db, engine
from pipelines.features import build_daily_feature_table
from engine.simulate import run_daily_replay, run_static_baseline
from engine.score import summarize_run
import pandas as pd
import os

def load_snapshots() -> pd.DataFrame:
    return pd.read_sql("select * from protocol_snapshots order by ts", engine)

def main():
    init_db()
    snapshots = load_snapshots()
    feat = build_daily_feature_table(snapshots)
    print("\n=== Feature sample ===")
    print(feat[["ts", "carry_quality_score", "carry_return_daily", "base_deposit_apy", "liquidity_score", "volatility_score"]].tail(20))
    
    print("\n=== carry_quality_score describe ===")
    print(feat["carry_quality_score"].describe())
    
    print("\n=== regime counts ===")
    print("risk_on   :", (feat["carry_quality_score"] >= 0.52).sum())
    print("defensive :", (feat["carry_quality_score"] <= 0.48).sum())
    print("neutral   :", ((feat["carry_quality_score"] > 0.48) & (feat["carry_quality_score"] < 0.52)).sum())
    adaptive = run_daily_replay(feat, policy_version="adaptive_v1")
    static = run_static_baseline(feat, policy_version="static_70_20_10")

    print("Adaptive summary:")
    print(summarize_run(adaptive))
    print("Static summary:")
    print(summarize_run(static))

    os.makedirs("data/parquet", exist_ok=True)
    adaptive.to_parquet("data/parquet/adaptive_replay.parquet", index=False)
    static.to_parquet("data/parquet/static_replay.parquet", index=False)
    feat.to_parquet("data/parquet/features.parquet", index=False)

if __name__ == "__main__":
    main()
