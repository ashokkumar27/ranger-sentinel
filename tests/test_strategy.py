from engine.strategy import target_weights, should_rebalance

def test_target_weights_sum_to_one():
    w = target_weights({"carry_quality_score": 0.8, "liquidity_score": 0.5, "volatility_score": 0.2})
    assert abs(sum(w.values()) - 1.0) < 1e-9

def test_reserve_floor():
    w = target_weights({"carry_quality_score": 0.9, "liquidity_score": 0.5, "volatility_score": 0.2})
    assert w["reserve"] >= 0.10

def test_rebalance_threshold():
    cur = {"base_yield": 0.70, "carry": 0.20, "reserve": 0.10}
    tar = {"base_yield": 0.62, "carry": 0.28, "reserve": 0.10}
    assert should_rebalance(cur, tar)
