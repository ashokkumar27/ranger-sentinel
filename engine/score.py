import pandas as pd

def annualized_return(nav_series: pd.Series, periods_per_year: int = 365) -> float:
    total = nav_series.iloc[-1] / nav_series.iloc[0]
    n = len(nav_series)
    return total ** (periods_per_year / max(n, 1)) - 1

def max_drawdown(nav_series: pd.Series) -> float:
    peak = nav_series.cummax()
    dd = nav_series / peak - 1
    return abs(dd.min())

def rolling_return(nav_series: pd.Series, window: int = 90) -> pd.Series:
    return nav_series / nav_series.shift(window) - 1

def summarize_run(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    nav = df["nav_end"].reset_index(drop=True)
    ann = annualized_return(nav)
    mdd = max_drawdown(nav)
    turnover_proxy = df["rebalance_cost_usd"].sum() / max(df["nav_start"].iloc[0], 1)
    rr90 = rolling_return(nav, 90)

    summary = {
        "annualized_return": ann,
        "max_drawdown": mdd,
        "turnover_proxy": turnover_proxy,
        "rolling_90d_median": rr90.median(skipna=True),
        "rolling_90d_worst": rr90.min(skipna=True),
        "rolling_90d_positive_share": (rr90.dropna() > 0).mean() if not rr90.dropna().empty else None,
    }
    summary["score"] = summary["annualized_return"] - 0.5 * summary["max_drawdown"] - 0.2 * summary["turnover_proxy"]
    return summary
