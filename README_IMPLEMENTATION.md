# Ranger Sentinel Persistent Carry Vault v1 Implementation Pack

This pack is designed to be dropped into the existing `ranger-sentinel` repo with **no new runtime dependencies** beyond the repo's current requirements:
- pandas
- numpy
- sqlalchemy
- pytest
- standard library

## Files

- `engine/policy_firewall.py`
  - Hard deny layer for banned structures, low-liquidity candidates, stressed venues, stale/invalid data.
- `engine/opportunity_builder.py`
  - Converts canonical snapshots into daily opportunity rows.
- `pipelines/feature_model_v2.py`
  - Builds persistence, spread, richness, exit-quality, funding-quality, and conviction features.
- `engine/allocator_persistent_carry_v1.py`
  - Implements Stage A eligibility gate and Stage B conviction sizing.
- `storage/decision_schemas.py`
  - SQLAlchemy schemas for daily decisions and allocations.
- `engine/metrics.py`
  - Replay from daily decisions plus performance summaries.
- `tests/test_persistent_carry_allocator.py`
  - Basic test coverage.

## Integration order

1. Keep the current repo's `pipelines/features.py` and `engine/strategy.py` as legacy v1.
2. Use `engine/opportunity_builder.py` after snapshot normalization.
3. Use `pipelines/feature_model_v2.py` on the opportunity table.
4. Use `engine/allocator_persistent_carry_v1.py` to emit:
   - daily decisions
   - daily allocations
5. Use `engine/metrics.py` to replay and summarize results.
6. Add the new SQLAlchemy models to your metadata if you want to persist decisions.

## Example flow

```python
from engine.opportunity_builder import build_opportunity_table
from pipelines.feature_model_v2 import build_feature_model_v2
from engine.allocator_persistent_carry_v1 import allocate_daily_decisions
from engine.metrics import replay_from_decisions, summarize_performance

opps = build_opportunity_table(snapshots_df)
features = build_feature_model_v2(opps)
decisions, allocations = allocate_daily_decisions(features)
replay = replay_from_decisions(decisions, allocations)
summary = summarize_performance(replay, hurdle_apy=0.10)
```

## Notes

- This pack intentionally does **not** add a migration framework.
- It also does **not** rewrite the existing collectors because the current repo appears lightweight and source schemas can drift.
- The right next step is to wire the new pipeline into `main_simulate.py` and `main_live.py` inside the repo.
