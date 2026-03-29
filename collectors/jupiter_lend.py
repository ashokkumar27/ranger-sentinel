from __future__ import annotations

import requests
import pandas as pd

from configs.settings import settings

URL = "https://api.jup.ag/lend/v1/earn/tokens"


class JupiterLendAPIError(RuntimeError):
    pass


def _safe_num(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick_first(row: dict, *keys):
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def fetch_earn_tokens() -> pd.DataFrame:
    if not settings.jupiter_api_key:
        return pd.DataFrame()

    headers = {"x-api-key": settings.jupiter_api_key}
    r = requests.get(URL, headers=headers, timeout=30)

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        snippet = r.text[:500] if r.text else ""
        raise JupiterLendAPIError(
            f"Jupiter Lend API request failed: status={r.status_code}, url={r.url}, body={snippet}"
        ) from e

    data = r.json()
    if not isinstance(data, list):
        raise JupiterLendAPIError(f"Unexpected Jupiter response type: {type(data)}")

    if not data:
        return pd.DataFrame()

    rows = []
    ts_now = pd.Timestamp.utcnow()

    for item in data:
        asset_obj = item.get("asset") or {}
        liquidity_supply_data = item.get("liquiditySupplyData") or {}

        # Per current docs:
        # - supplyRate = base supply yield
        # - rewardsRate = extra incentive yield
        # - totalRate = total yield
        # We store totalRate as deposit_apy because that is the economically relevant earn yield.
    def _safe_rate_1e4(value):
        num = _safe_num(value)
        if num is None:
            return None
        return num / 1e4
    
    
    total_rate = _safe_rate_1e4(_pick_first(item, "totalRate"))
    supply_rate = _safe_rate_1e4(_pick_first(item, "supplyRate"))
    rewards_rate = _safe_rate_1e4(_pick_first(item, "rewardsRate"))
    
    fallback_rate = _safe_num(
        _pick_first(item, "apy", "supplyApy")
    )
    
    # Priority: totalRate → fallback → derived
    deposit_apy = total_rate
    if deposit_apy is None:
        deposit_apy = fallback_rate
    if deposit_apy is None and supply_rate is not None and rewards_rate is not None:
        deposit_apy = supply_rate + rewards_rate
    if deposit_apy is None:
        deposit_apy = supply_rate
    
    base_supply_apy = supply_rate

        # Try several liquidity candidates. Current docs mention totalAssets and liquiditySupplyData.
        available_liquidity_usd = None

        total_assets = _safe_num(item.get("totalAssets"))
        asset_price = _safe_num(asset_obj.get("price"))
        if total_assets is not None and asset_price is not None:
            # Note: totalAssets is raw token quantity in token base units in docs examples,
            # but decimals are also provided, so normalize when possible.
            decimals = item.get("decimals")
            try:
                decimals = int(decimals) if decimals is not None else None
            except (TypeError, ValueError):
                decimals = None

            if decimals is not None:
                available_liquidity_usd = (total_assets / (10 ** decimals)) * asset_price
            else:
                available_liquidity_usd = total_assets * asset_price

        # Fallbacks if totalAssets path is not enough or changes upstream
        if available_liquidity_usd is None:
            available_liquidity_usd = _safe_num(
                _pick_first(
                    item,
                    "liquidityUsd",
                    "availableLiquidityUsd",
                    "tvlUsd",
                )
            )

        if available_liquidity_usd is None and liquidity_supply_data:
            available_liquidity_usd = _safe_num(
                _pick_first(
                    liquidity_supply_data,
                    "liquidityUsd",
                    "availableLiquidityUsd",
                    "tvlUsd",
                    "usdValue",
                )
            )

        asset_symbol = _pick_first(
            asset_obj,
            "symbol",
        ) or _pick_first(
            item,
            "assetSymbol",
            "symbol",
        )

        asset_address = _pick_first(
            item,
            "assetAddress",
            "address",
        ) or asset_obj.get("address")

        row = {
            "ts": ts_now,
            "protocol": "jupiter_lend",
            "venue": "jupiter_lend",
            "market": "jupiter_earn",
            "asset": asset_symbol or asset_address or item.get("id"),
            "deposit_apy": deposit_apy,
            "borrow_apy": None,
            "funding_rate_daily": None,
            "utilization": None,
            "available_liquidity_usd": available_liquidity_usd,
            "price_usd": _safe_num(asset_obj.get("price")),
            "jupiter_supply_rate": base_supply_apy,
            "jupiter_rewards_rate": rewards_rate,
            "asset_address": asset_address,
            "raw_json": item,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
