import pandas as pd

from collectors.jupiter_lend import _safe_num


def test_safe_num_handles_strings():
    assert _safe_num("0.06") == 0.06
    assert _safe_num("") is None
    assert _safe_num(None) is None


def test_jupiter_mapping_shape_example():
    # This mirrors the current Jupiter docs shape closely enough
    payload = [
        {
            "id": "So11111111111111111111111111111111111111112",
            "address": "jlSOLAddress",
            "name": "Jupiter Lend SOL",
            "symbol": "jlSOL",
            "decimals": 9,
            "assetAddress": "So11111111111111111111111111111111111111112",
            "asset": {
                "address": "So11111111111111111111111111111111111111112",
                "name": "Wrapped SOL",
                "symbol": "SOL",
                "decimals": 9,
                "price": 171.34,
            },
            "totalAssets": "1500000000000",
            "totalSupply": "1450000000000",
            "rewardsRate": "0.035",
            "supplyRate": "0.025",
            "totalRate": "0.06",
        }
    ]

    rows = []
    ts_now = pd.Timestamp.utcnow()

    for item in payload:
        asset_obj = item.get("asset") or {}
        decimals = int(item["decimals"])
        total_assets = float(item["totalAssets"])
        asset_price = float(asset_obj["price"])
        available_liquidity_usd = (total_assets / (10 ** decimals)) * asset_price

        rows.append(
            {
                "ts": ts_now,
                "protocol": "jupiter_lend",
                "venue": "jupiter_lend",
                "market": "jupiter_earn",
                "asset": asset_obj["symbol"],
                "deposit_apy": float(item["totalRate"]),
                "available_liquidity_usd": available_liquidity_usd,
                "price_usd": float(asset_obj["price"]),
            }
        )

    df = pd.DataFrame(rows)

    assert not df.empty
    assert float(df.iloc[0]["deposit_apy"]) == 0.06
    assert df.iloc[0]["asset"] == "SOL"
    assert float(df.iloc[0]["available_liquidity_usd"]) > 0
