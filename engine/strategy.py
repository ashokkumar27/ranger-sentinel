from dataclasses import dataclass


@dataclass
class PolicyConfig:
    base_weight_neutral: float = 0.70
    carry_weight_neutral: float = 0.20
    reserve_weight_neutral: float = 0.10

    max_carry_weight: float = 0.35
    min_reserve_weight: float = 0.10
    rebalance_threshold_bps: float = 50.0

    risk_on_score: float = 0.58
    defensive_score: float = 0.42


def normalize(base_yield: float, carry: float, reserve: float):
    total = base_yield + carry + reserve
    return {
        "base_yield": base_yield / total,
        "carry": carry / total,
        "reserve": reserve / total,
    }


def target_weights(signal: dict, cfg: PolicyConfig | None = None):
    cfg = cfg or PolicyConfig()
    score = float(signal["carry_quality_score"])

    if score >= cfg.risk_on_score:
        weights = normalize(0.55, min(cfg.max_carry_weight, 0.35), 0.10)
    elif score <= cfg.defensive_score:
        weights = normalize(0.85, 0.05, 0.10)
    else:
        weights = normalize(
            cfg.base_weight_neutral,
            cfg.carry_weight_neutral,
            cfg.reserve_weight_neutral,
        )

    if weights["reserve"] < cfg.min_reserve_weight:
        diff = cfg.min_reserve_weight - weights["reserve"]
        weights["reserve"] += diff
        weights["carry"] -= diff

    return weights


def should_rebalance(current: dict, target: dict, cfg: PolicyConfig | None = None) -> bool:
    cfg = cfg or PolicyConfig()
    shift_bps = sum(abs(current[k] - target[k]) for k in current) * 10000
    return shift_bps >= cfg.rebalance_threshold_bps
