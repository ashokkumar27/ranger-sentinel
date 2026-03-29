from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_z(series: pd.Series, window: int = 30, min_periods: int = 5) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)



def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)



def build_feature_model_v2(opportunities: pd.DataFrame) -> pd.DataFrame:
    x = opportunities.copy()
    if x.empty:
        return x

    x["date"] = pd.to_datetime(x["date"], utc=True, errors="coerce")
    x = x.dropna(subset=["date"]).sort_values(["asset", "protocol", "market", "date"])

    group_cols = ["asset", "protocol", "market"]
    g = x.groupby(group_cols, dropna=False)

    # Yield persistence
    x["apy_avg_7d"] = g["effective_earn_apy"].transform(lambda s: s.rolling(7, min_periods=2).mean())
    x["apy_avg_30d"] = g["effective_earn_apy"].transform(lambda s: s.rolling(30, min_periods=5).mean())
    x["apy_std_7d"] = g["effective_earn_apy"].transform(lambda s: s.rolling(7, min_periods=2).std()).fillna(0.0)
    x["apy_std_30d"] = g["effective_earn_apy"].transform(lambda s: s.rolling(30, min_periods=5).std()).fillna(0.0)
    x["apy_momentum_7d"] = (x["effective_earn_apy"] - x["apy_avg_7d"]).fillna(0.0)
    x["apy_momentum_30d"] = (x["apy_avg_7d"] - x["apy_avg_30d"]).fillna(0.0)
    x["positive_days_14"] = g["effective_earn_apy"].transform(lambda s: s.gt(0).rolling(14, min_periods=3).mean()).fillna(0.0)
    x["downside_days_14"] = g["effective_earn_apy"].transform(lambda s: s.lt(s.rolling(14, min_periods=3).median()).rolling(14, min_periods=3).mean()).fillna(0.0)
    x["apy_stability"] = (1.0 / (1.0 + x["apy_std_30d"].abs() * 10.0)).clip(0.0, 1.0)

    # Spread quality
    x["spread_apy"] = (x["supply_apy"] - x["borrow_apy"]).fillna(0.0)
    x["spread_ratio"] = (x["supply_apy"] / x["borrow_apy"].replace({0: np.nan})).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x["spread_avg_7d"] = g["spread_apy"].transform(lambda s: s.rolling(7, min_periods=2).mean()).fillna(0.0)
    x["spread_positive_days_14"] = g["spread_apy"].transform(lambda s: s.gt(0).rolling(14, min_periods=3).mean()).fillna(0.0)
    x["spread_change_7d"] = g["spread_apy"].transform(lambda s: s.diff(7)).fillna(0.0)

    # Cross-venue richness per asset/day
    asset_day_median = x.groupby(["date", "asset"], dropna=False)["effective_earn_apy"].transform("median")
    x["richness_apy"] = (x["effective_earn_apy"] - asset_day_median).fillna(0.0)
    x["richness_z_30d"] = x.groupby(["asset", "protocol"], dropna=False)["richness_apy"].transform(_rolling_z)
    x["richness_positive_days_14"] = x.groupby(["asset", "protocol"], dropna=False)["richness_apy"].transform(lambda s: s.gt(0).rolling(14, min_periods=3).mean()).fillna(0.0)
    x["dispersion_stability"] = (
        1.0 / (1.0 + x.groupby(["asset", "protocol"], dropna=False)["richness_apy"].transform(lambda s: s.rolling(14, min_periods=3).std()).fillna(0.0).abs() * 20.0)
    ).clip(0.0, 1.0)

    # Exit quality
    liquidity_score = (np.log1p(x["available_liquidity_usd"].fillna(0.0)) / np.log1p(max(float(x["available_liquidity_usd"].fillna(0).max()), 1.0))).clip(0.0, 1.0)
    utilization_penalty = x["utilization"].fillna(0.0).clip(0.0, 1.0)
    x["exit_quality_score"] = (
        0.45 * liquidity_score
        + 0.25 * (1.0 - utilization_penalty)
        + 0.20 * (1.0 - x["withdraw_friction_score"].fillna(0.0).clip(0.0, 1.0))
        + 0.10 * x["data_quality_pass"].astype(float)
    ).clip(0.0, 1.0)

    # Funding quality
    x["funding_avg_3d"] = g["funding_rate_daily"].transform(lambda s: s.rolling(3, min_periods=2).mean()).fillna(0.0)
    x["funding_avg_7d"] = g["funding_rate_daily"].transform(lambda s: s.rolling(7, min_periods=2).mean()).fillna(0.0)
    x["funding_vol_7d"] = g["funding_rate_daily"].transform(lambda s: s.rolling(7, min_periods=2).std()).fillna(0.0)
    funding_sign = np.sign(x["funding_rate_daily"].fillna(0.0))
    x["funding_persistence_7d"] = x.groupby(group_cols, dropna=False)["funding_rate_daily"].transform(
        lambda s: np.sign(s.fillna(0.0)).eq(np.sign(s.fillna(0.0).rolling(7, min_periods=2).mean())).rolling(7, min_periods=2).mean()
    ).fillna(0.0)
    x["funding_level_norm"] = _clip01((x["funding_avg_7d"].fillna(0.0) + 0.10) / 0.20)
    x["funding_vol_norm"] = _clip01(x["funding_vol_7d"].fillna(0.0) / max(float(x["funding_vol_7d"].fillna(0).max() or 1.0), 1e-9))
    x["funding_quality_score"] = (
        0.35 * x["funding_persistence_7d"]
        + 0.25 * x["funding_level_norm"]
        + 0.20 * (1.0 - x["funding_vol_norm"])
        + 0.20 * (1.0 - x["oracle_stress_score"].fillna(0.0).clip(0.0, 1.0))
    ).clip(0.0, 1.0)

    # Composite scores
    x["persistence_score"] = (
        0.35 * x["apy_stability"]
        + 0.25 * x["positive_days_14"]
        + 0.20 * _clip01((x["apy_momentum_7d"].fillna(0.0) + 0.20) / 0.40)
        + 0.20 * (1.0 - x["downside_days_14"])
    ).clip(0.0, 1.0)

    x["spread_quality_score"] = (
        0.40 * _clip01((x["spread_avg_7d"].fillna(0.0) + 0.10) / 0.30)
        + 0.35 * x["spread_positive_days_14"]
        + 0.25 * _clip01((x["spread_change_7d"].fillna(0.0) + 0.10) / 0.30)
    ).clip(0.0, 1.0)

    x["richness_quality_score"] = (
        0.45 * _clip01((x["richness_z_30d"].fillna(0.0) + 2.0) / 4.0)
        + 0.35 * x["richness_positive_days_14"]
        + 0.20 * x["dispersion_stability"]
    ).clip(0.0, 1.0)

    x["conviction_score"] = (
        0.35 * x["persistence_score"]
        + 0.25 * x["richness_quality_score"]
        + 0.20 * x["spread_quality_score"]
        + 0.10 * x["exit_quality_score"]
        + 0.10 * x["funding_quality_score"]
    ).clip(0.0, 1.0)

    x["stress_flag"] = (
        (x["market_status"].astype(str).str.lower().isin(["halted", "paused", "settlement", "closed"]))
        | (x["oracle_stress_score"].fillna(0.0) > 0.60)
        | (x["utilization"].fillna(0.0) > 0.92)
        | (~x["data_quality_pass"].fillna(False))
    )

    return x.sort_values(["date", "asset", "protocol", "market"]).reset_index(drop=True)
