from __future__ import annotations

from pathlib import Path
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from engine.opportunity_builder import build_opportunity_table
from pipelines.feature_model_v2 import build_feature_model_v2
from engine.allocator_persistent_carry_v1 import (
    AllocationConfig,
    allocate_daily_decisions,
)


ARTIFACT_DIR = Path("artifacts/persistent_carry_live")
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


def pick_latest_decision_block(
    decisions: pd.DataFrame,
    allocations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date" not in decisions.columns:
        raise ValueError("Decisions must contain 'date' column.")
    latest_date = decisions["date"].max()

    latest_decisions = decisions.loc[decisions["date"] == latest_date].copy()
    latest_allocations = allocations.loc[allocations["date"] == latest_date].copy()

    if latest_decisions.empty:
        raise ValueError("No latest decision block found.")
    if latest_allocations.empty:
        raise ValueError("No latest allocation block found.")

    return latest_decisions, latest_allocations


def main() -> int:
    ensure_dirs()

    print("[1/5] Loading snapshots...")
    snapshots = load_snapshots()
    if snapshots.empty:
        raise ValueError("Snapshots dataset is empty.")

    print("[2/5] Building opportunities...")
    opps = build_opportunity_table(snapshots)
    if opps.empty:
        raise ValueError("Opportunity table is empty.")

    print("[3/5] Building features...")
    feats = build_feature_model_v2(opps)
    if feats.empty:
        raise ValueError("Feature table is empty.")

    print("[4/5] Running allocator...")
    cfg = AllocationConfig()
    decisions, allocations = allocate_daily_decisions(feats, cfg)

    latest_decisions, latest_allocations = pick_latest_decision_block(
        decisions, allocations
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    live_payload = {
        "generated_at_utc": generated_at,
        "latest_date": str(latest_decisions["date"].iloc[0]),
        "decision_summary": latest_decisions.to_dict(orient="records"),
        "allocations": latest_allocations.to_dict(orient="records"),
    }

    print("[5/5] Saving live recommendation artifacts...")
    latest_decisions.to_csv(ARTIFACT_DIR / "latest_decisions.csv", index=False)
    latest_allocations.to_csv(ARTIFACT_DIR / "latest_allocations.csv", index=False)

    with open(ARTIFACT_DIR / "latest_payload.json", "w", encoding="utf-8") as f:
        json.dump(live_payload, f, indent=2, default=str)

    print("\n=== Persistent Carry Live Output ===")
    print(json.dumps(live_payload, indent=2, default=str))
    print(f"\nArtifacts saved to: {ARTIFACT_DIR.resolve()}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise
