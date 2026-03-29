from dataclasses import dataclass


@dataclass
class PolicyConfig:
    base_weight_neutral: float = 0.66
    carry_weight_neutral: float = 0.24
    reserve_weight_neutral: float = 0.10

    max_carry_weight: float = 0.30
    min_reserve_weight: float = 0.10
    rebalance_threshold_bps: float = 60.0

    risk_on_enter: float = 0.72
    risk_on_exit: float = 0.62

    defensive_enter: float = 0.16
    defensive_exit: float = 0.28


def normalize(base_yield: float, carry: float, reserve: float):
    total = base_yield + carry + reserve
    return {
        "base_yield": base_yield / total,
        "carry": carry / total,
        "reserve": reserve / total,
    }


def infer_regime(signal: dict, current: dict, cfg: PolicyConfig) -> str:
    score = float(signal["carry_quality_score"])

    # infer current regime from weights
    if abs(current["carry"] - 0.30) < 1e-9:
        current_regime = "risk_on"
    elif abs(current["carry"] - 0.16) < 1e-9:
        current_regime = "defensive"
    else:
        current_regime = "neutral"

    if current_regime == "risk_on":
        return "risk_on" if score >= cfg.risk_on_exit else "neutral"

    if current_regime == "defensive":
        return "defensive" if score <= cfg.defensive_exit else "neutral"

    if score >= cfg.risk_on_enter:
        return "risk_on"
    if score <= cfg.defensive_enter:
        return "defensive"
    return "neutral"


def target_weights(signal: dict, current: dict | None = None, cfg: PolicyConfig | None = None):
    cfg = cfg or PolicyConfig()
    current = current or {
        "base_yield": cfg.base_weight_neutral,
        "carry": cfg.carry_weight_neutral,
        "reserve": cfg.reserve_weight_neutral,
    }

    regime = infer_regime(signal, current, cfg)

    if regime == "risk_on":
        weights = normalize(0.60, 0.30, 0.10)
    elif regime == "defensive":
        weights = normalize(0.74, 0.16, 0.10)
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
