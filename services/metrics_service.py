from __future__ import annotations

from typing import Any

import pandas as pd


METRIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "carry": ("Carry_m", "Avg_Carry_m"),
    "total": ("Total_m", "Avg_Total_m"),
    "ball_speed": ("BallSpeed_mps", "Avg_BallSpeed_mps"),
    "club_speed": ("ClubSpeed_mps", "Avg_ClubSpeed_mps"),
    "smash": ("SmashFactor", "Avg_Smash"),
    "spin": ("SpinRate_rpm", "Avg_Spin_rpm"),
    "launch": ("LaunchAngle_deg", "Avg_Launch_deg"),
    "side": ("TotalSide_m", "Avg_TotalSide_m"),
    "apex": ("Apex_m", "Avg_Apex_m"),
}


def _mean(df: pd.DataFrame, candidates: tuple[str, ...]) -> float:
    for column in candidates:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return float("nan")


def calculate_dashboard_metrics(df: pd.DataFrame) -> dict[str, float]:
    """현재 필터 결과에서 Dashboard KPI를 계산합니다."""
    return {
        name: _mean(df, candidates)
        for name, candidates in METRIC_COLUMNS.items()
    }


def calculate_metric_summary(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """평균/표본수/표준편차를 포함한 재사용 가능한 KPI 요약."""
    metrics: dict[str, dict[str, float]] = {}
    for name, candidates in METRIC_COLUMNS.items():
        values = pd.Series(dtype="float64")
        for column in candidates:
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="coerce").dropna()
                if not values.empty:
                    break
        metrics[name] = {
            "mean": float(values.mean()) if not values.empty else float("nan"),
            "std": float(values.std()) if len(values) > 1 else float("nan"),
            "count": float(len(values)),
            "best": float(values.max()) if not values.empty else float("nan"),
        }
    return metrics


def calculate_change(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float]:
    """현재 KPI와 비교 기준 KPI의 절대 변화량을 계산합니다."""
    result: dict[str, float] = {}
    for key in current.keys() & baseline.keys():
        try:
            current_value = float(current[key])
            baseline_value = float(baseline[key])
            if pd.isna(current_value) or pd.isna(baseline_value):
                result[key] = float("nan")
            else:
                result[key] = current_value - baseline_value
        except (TypeError, ValueError):
            result[key] = float("nan")
    return result


def calculate_kpi_items(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Dashboard UI가 바로 사용할 수 있는 KPI tuple을 생성합니다."""
    metrics = calculate_dashboard_metrics(df)

    def fmt(value: float, decimals: int = 1) -> str:
        return "-" if pd.isna(value) else f"{value:.{decimals}f}"

    return [
        ("Carry", fmt(metrics["carry"]), "m"),
        ("Total", fmt(metrics["total"]), "m"),
        ("Ball Speed", fmt(metrics["ball_speed"]), "m/s"),
        ("Club Speed", fmt(metrics["club_speed"]), "m/s"),
        ("Smash", fmt(metrics["smash"], 2), ""),
        ("Spin", fmt(metrics["spin"], 0), "rpm"),
        ("Launch", fmt(metrics["launch"]), "°"),
        ("Side", fmt(metrics["side"]), "m"),
        ("Apex", fmt(metrics["apex"]), "m"),
    ]
