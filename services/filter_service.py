from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


DEFAULT_RANGE_SPECS: tuple[tuple[str, str, float], ...] = (
    ("Carry_m", "캐리", 1.0),
    ("Total_m", "토탈", 1.0),
    ("BallSpeed_mps", "볼 스피드", 0.5),
    ("ClubSpeed_mps", "클럽 스피드", 0.5),
    ("SpinRate_rpm", "스핀량", 50.0),
    ("LaunchAngle_deg", "발사각", 0.5),
    ("TotalSide_m", "사이드", 1.0),
)


def numeric_range(series: pd.Series) -> tuple[float, float] | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min()), float(values.max())


def apply_numeric_range(
    df: pd.DataFrame,
    column: str,
    selected_range: tuple[float, float] | list[float],
) -> pd.DataFrame:
    """Inclusive numeric range filter."""
    if df.empty or column not in df.columns:
        return df.copy()
    low, high = float(selected_range[0]), float(selected_range[1])
    values = pd.to_numeric(df[column], errors="coerce")
    mask = values.between(low, high, inclusive="both")
    return df.loc[mask].copy()


def apply_filters(
    df: pd.DataFrame,
    *,
    club: str | None = None,
    selected_date: Any | None = None,
    selected_clubs: Iterable[str] | None = None,
    selected_dates: Iterable[Any] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """대시보드에서 공통으로 사용하는 기본 필터를 적용합니다.

    필터 계산은 UI와 분리되어 있어 향후 Sidebar, 분석 탭, 성장 분석에서
    동일한 규칙을 재사용할 수 있습니다.
    """
    result = df.copy()

    clubs = list(selected_clubs) if selected_clubs is not None else None
    if club is not None:
        clubs = [club]
    if clubs is not None and "Club" in result.columns:
        result = result[result["Club"].isin(clubs)].copy()

    dates = list(selected_dates) if selected_dates is not None else None
    if selected_date is not None:
        dates = [selected_date]
    if dates is not None and "Date" in result.columns:
        result = result[result["Date"].isin(dates)].copy()

    for column, selected_range in (ranges or {}).items():
        result = apply_numeric_range(result, column, selected_range)

    return result.reset_index(drop=True)


def build_range_bounds(
    df: pd.DataFrame,
    *,
    club: str | None = None,
    specs: tuple[tuple[str, str, float], ...] = DEFAULT_RANGE_SPECS,
) -> dict[str, tuple[float, float]]:
    """슬라이더 기본 범위를 계산합니다. 날짜 필터는 적용하지 않습니다."""
    base = df
    if club is not None and "Club" in base.columns:
        base = base[base["Club"] == club]

    bounds: dict[str, tuple[float, float]] = {}
    for column, _label, step in specs:
        if column not in base.columns:
            continue
        current = numeric_range(base[column])
        if current is None:
            continue
        low, high = current
        low = float(int(low / step) * step)
        high = float(-(-high // step) * step)
        if high <= low:
            high = low + step
        bounds[column] = (low, high)
    return bounds
