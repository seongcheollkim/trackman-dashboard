from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def numeric(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def grade_from_score(score: float) -> str:
    score = float(score)
    if score >= 95:
        return "A+"
    if score >= 90:
        return "A"
    if score >= 85:
        return "B+"
    if score >= 80:
        return "B"
    if score >= 75:
        return "C+"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def relative_score(
    today: float | None,
    baseline: float | None,
    *,
    higher_is_better: bool,
    neutral: float = 72.0,
    sensitivity: float = 0.10,
) -> float:
    """
    사용자 본인의 과거 평균 대비 오늘 값을 0~100 점으로 변환합니다.

    sensitivity=0.10이면 과거 평균 대비 약 ±10% 변화가 점수에
    크게 반영됩니다. 절대 기준이 아닌 '나의 최근 수준 대비 변화'가 핵심입니다.
    """
    t = numeric(today)
    b = numeric(baseline)
    if t is None:
        return 50.0
    if b is None or abs(b) < 1e-9:
        return neutral

    change = (t - b) / abs(b)
    signed = change if higher_is_better else -change
    return clamp(neutral + 25.0 * (signed / max(sensitivity, 1e-6)))


def closeness_score(
    today: float | None,
    baseline: float | None,
    *,
    neutral: float = 76.0,
    tolerance: float = 0.15,
) -> float:
    """
    높고 낮음의 방향성이 확실하지 않은 지표(런치, 스핀 등)는
    과거 자기 기준에서 얼마나 안정적으로 유지됐는지를 평가합니다.
    """
    t = numeric(today)
    b = numeric(baseline)
    if t is None:
        return 50.0
    if b is None or abs(b) < 1e-9:
        return neutral

    deviation = abs(t - b) / abs(b)
    return clamp(100.0 - 40.0 * (deviation / max(tolerance, 1e-6)))


def consistency_score(
    today_std: float | None,
    baseline_std: float | None,
    *,
    neutral: float = 72.0,
) -> float:
    """표준편차가 작아질수록 높은 점수."""
    t = numeric(today_std)
    b = numeric(baseline_std)
    if t is None:
        return 50.0
    if b is None or b <= 1e-9:
        return neutral
    return relative_score(t, b, higher_is_better=False, neutral=neutral, sensitivity=0.20)


def weighted_score(parts: list[tuple[float | None, float]]) -> float:
    usable = [(numeric(score), float(weight)) for score, weight in parts]
    usable = [(score, weight) for score, weight in usable if score is not None and weight > 0]
    if not usable:
        return 0.0
    numerator = sum(score * weight for score, weight in usable)
    denominator = sum(weight for _, weight in usable)
    return round(clamp(numerator / denominator), 1)


def confidence_score(shots: int, baseline_sessions: int, baseline_shots: int) -> int:
    """
    당일 샷 수 + 비교 가능한 과거 세션 수를 기반으로 진단 신뢰도를 산정합니다.
    1~2구만 친 클럽은 의도적으로 신뢰도를 낮춥니다.
    """
    shot_component = min(max(int(shots), 0), 20) / 20 * 52
    session_component = min(max(int(baseline_sessions), 0), 10) / 10 * 33
    history_component = min(max(int(baseline_shots), 0), 100) / 100 * 15
    return int(round(clamp(shot_component + session_component + history_component)))


def safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else None


def safe_std(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 2:
        return None
    return float(values.std(ddof=1))



def min_target_score(value: float | None, target_min: float, *, soft_margin: float) -> float:
    value = numeric(value)
    if value is None:
        return 50.0
    if value >= target_min:
        return 100.0
    margin = max(float(soft_margin), 1e-6)
    return clamp(100.0 - ((target_min - value) / margin) * 45.0)


def max_target_score(value: float | None, target_max: float, *, soft_margin: float) -> float:
    value = numeric(value)
    if value is None:
        return 50.0
    if value <= target_max:
        return 100.0
    margin = max(float(soft_margin), 1e-6)
    return clamp(100.0 - ((value - target_max) / margin) * 45.0)


def range_target_score(
    value: float | None,
    target_low: float,
    target_high: float,
    *,
    soft_margin: float,
) -> float:
    value = numeric(value)
    if value is None:
        return 50.0
    if target_low <= value <= target_high:
        return 100.0
    distance = target_low - value if value < target_low else value - target_high
    return clamp(100.0 - (distance / max(float(soft_margin), 1e-6)) * 45.0)
