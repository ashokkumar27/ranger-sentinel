import pandas as pd


def apy_to_daily(apy: float) -> float:
    if apy is None or pd.isna(apy):
        return 0.0
    return (1 + max(float(apy), -0.9999)) ** (1 / 365.0) - 1


def build_daily_feature_table(snapshots: pd.DataFrame) -> pd.DataFrame:
    x = snapshots.copy()
    if x.empty:
        return x

    x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
    x = x.dropna(subset=["ts"])

    x["deposit_apy"] = pd.to_numeric(x["deposit_apy"], errors="coerce")
    x["borrow_apy"] = pd.to_numeric(x["borrow_apy"], errors="coerce")
    x["funding_rate_daily"] = pd.to_numeric(x["funding_rate_daily"], errors="coerce")
    x["available_liquidity_usd"] = pd.to_numeric(x["available_liquidity_usd"], errors="coerce")
    x["utilization"] = pd.to_numeric(x["utilization"], errors="coerce")
    x["price_usd"] = pd.to_numeric(x["price_usd"], errors="coerce")

    x["date"] = x["ts"].dt.floor("D")

    daily = x.groupby(["date", "protocol"], as_index=False).agg(
        deposit_apy=("deposit_apy", "mean"),
        borrow_apy=("borrow_apy", "mean"),
        funding_rate_daily=("funding_rate_daily", "mean"),
        available_liquidity_usd=("available_liquidity_usd", "mean"),
        utilization=("utilization", "mean"),
        price_usd=("price_usd", "mean"),
    )

    pivot = daily.pivot(index="date", columns="protocol")
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index().sort_values("date")

    for c in pivot.columns:
        if c != "date":
            pivot[c] = pivot[c].ffill()

    dep_cols = [c for c in pivot.columns if c.startswith("deposit_apy_")]
    if dep_cols:
        pivot["base_deposit_apy"] = pivot[dep_cols].mean(axis=1)
    else:
        pivot["base_deposit_apy"] = 0.0

    pivot["base_return_daily"] = pivot["base_deposit_apy"].apply(apy_to_daily)

    funding_proxy = pivot.get(
        "funding_rate_daily_drift",
        pd.Series(0.0, index=pivot.index),
    ).fillna(0.0)

    yield_spread_proxy = pivot["base_return_daily"].fillna(0.0)

    pivot["carry_return_daily"] = funding_proxy + 0.3 * yield_spread_proxy - 0.00005

    liq_cols = [c for c in pivot.columns if c.startswith("available_liquidity_usd_")]
    if liq_cols:
        pivot["liquidity_score"] = pivot[liq_cols].mean(axis=1).rank(pct=True).fillna(0.5)
    else:
        pivot["liquidity_score"] = 0.5

    px_col = "price_usd_drift" if "price_usd_drift" in pivot.columns else None
    if px_col:
        pivot["realized_vol_7d"] = (
            pivot[px_col]
            .pct_change(fill_method=None)
            .rolling(7)
            .std()
            .fillna(0.0)
        )
    else:
        pivot["realized_vol_7d"] = 0.0

    pivot["volatility_score"] = pivot["realized_vol_7d"]

    raw = (
        0.70 * pivot["carry_return_daily"].fillna(0.0)
        + 0.20 * pivot["liquidity_score"].fillna(0.0)
        - 0.10 * pivot["volatility_score"].fillna(0.0)
    )

    mn, mx = raw.min(), raw.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        pivot["carry_quality_score"] = 0.5
    else:
        pivot["carry_quality_score"] = (raw - mn) / (mx - mn)

    pivot = pivot.rename(columns={"date": "ts"})
    return pivot
