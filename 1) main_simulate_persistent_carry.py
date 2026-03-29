from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

from engine.opportunity_builder import build_opportunity_table
from pipelines.feature_model_v2 import build_feature_model_v2
from engine.allocator_persistent_carry_v1 import (
    AllocationConfig,
    allocate_daily_decisions,
)
from engine.metrics import replay_from_decisions, summarize_performance


ARTIFACT_DIR = Path("artifacts/persistent_carry")
DEFAULT_PARQUET = Path("data/protocol_snapshots.parquet")
DEFAULT_CSV = Path("data/protocol_snapshots.csv")


def load_snapshots() -> pd.DataFrame:
    if DEFAULT_PARQUET.exists():
        return pd.read_parquet(DEFAULT_PARQUET)
    if DEFAULT_CSV.exists():
        return pd.read_csv(DEFAULT_CSV)
    raise FileNotFoundError(
        "No snapshot file found. Expected one of:\n"
        f" - {DEFAULT_PARQUET}\n"
        f" - {DEFAULT_CSV}"
    )


def ensure_dirs() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ensure_dirs()

    print("[1/6] Loading canonical snapshots...")
    snapshots = load_snapshots()
    if snapshots.empty:
        raise ValueError("Snapshots dataset is empty.")

    print(f"Loaded snapshots: {snapshots.shape}")

    print("[2/6] Building opportunity table...")
    opps = build_opportunity_table(snapshots)
    if opps.empty:
        raise ValueError("Opportunity table is empty after build_opportunity_table().")
    print(f"Opportunities: {opps.shape}")

    print("[3/6] Building feature model v2...")
    feats = build_feature_model_v2(opps)
    if feats.empty:
        raise ValueError("Feature table is empty after build_feature_model_v2().")
    print(f"Features: {feats.shape}")

    print("[4/6] Running allocator...")
    cfg = AllocationConfig()
    decisions, allocations = allocate_daily_decisions(feats, cfg)
    if decisions.empty:
        raise ValueError("Allocator produced no decisions.")
    if allocations.empty:
        raise ValueError("Allocator produced no allocations.")
    print(f"Decisions: {decisions.shape}")
    print(f"Allocations: {allocations.shape}")

    print("[5/6] Replaying performance...")
    replay = replay_from_decisions(decisions, allocations)
    summary = summarize_performance(replay, hurdle_apy=0.10)

    print("[6/6] Saving artifacts...")
    opps.to_csv(ARTIFACT_DIR / "opportunities.csv", index=False)
    feats.to_csv(ARTIFACT_DIR / "features_v2.csv", index=False)
    decisions.to_csv(ARTIFACT_DIR / "daily_decisions.csv", index=False)
    allocations.to_csv(ARTIFACT_DIR / "daily_allocations.csv", index=False)
    replay.to_csv(ARTIFACT_DIR / "replay.csv", index=False)

    with open(ARTIFACT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== Persistent Carry Simulation Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nArtifacts saved to: {ARTIFACT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
