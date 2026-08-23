from __future__ import annotations

from typing import Any

import pandas as pd


COMPARE_METRICS: tuple[tuple[str, str, str, int, bool], ...] = (
    ("캐리", "Avg_Carry_m", "m", 0, False),
    ("토탈", "Avg_Total_m", "m", 0, False),
    ("볼 스피드", "Avg_BallSpeed_mps", "m/s", 1, False),
    ("클럽 스피드", "Avg_ClubSpeed_mps", "m/s", 1, False),
    ("스매시 팩터", "Avg_Smash", "", 2, False),
    ("스핀량", "Avg_Spin_rpm", "rpm", 0, False),
    ("발사각", "Avg_Launch_deg", "°", 1, False),
    ("좌우 편차", "Avg_AbsSide_m", "m", 1, True),
)


def _as_number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def period_delta(reference: Any, value: Any) -> float | None:
    reference_value = _as_number(reference)
    value_value = _as_number(value)
    if reference_value is None or value_value is None:
        return None
    return value_value - reference_value


def compare_metric(
    day: Any,
    month: Any,
    year: Any,
    *,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    """선택일을 기준으로 월간/연간 평균과 차이를 계산합니다."""
    day_value = _as_number(day)
    month_value = _as_number(month)
    year_value = _as_number(year)

    def result(reference: float | None) -> dict[str, Any]:
        delta = None if day_value is None or reference is None else day_value - reference
        if delta is None:
            status = "neutral"
        elif abs(delta) < 1e-12:
            status = "neutral"
        else:
            better = delta < 0 if lower_is_better else delta > 0
            status = "good" if better else "bad"
        return {"value": reference, "delta": delta, "status": status}

    return {
        "day": day_value,
        "month": result(month_value),
        "year": result(year_value),
    }


def build_compare_payload(
    day: pd.DataFrame,
    month: pd.DataFrame,
    year: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Dashboard 비교 카드에 필요한 순수 데이터만 생성합니다."""
    payload: list[dict[str, Any]] = []

    for title, column, unit, decimals, lower_is_better in COMPARE_METRICS:
        day_value = day[column].iloc[0] if column in day.columns and not day.empty else None
        month_value = month[column].iloc[0] if column in month.columns and not month.empty else None
        year_value = year[column].iloc[0] if column in year.columns and not year.empty else None

        payload.append(
            {
                "title": title,
                "column": column,
                "unit": unit,
                "decimals": decimals,
                "lower_is_better": lower_is_better,
                "comparison": compare_metric(
                    day_value,
                    month_value,
                    year_value,
                    lower_is_better=lower_is_better,
                ),
            }
        )

    return payload


def build_club_periods(
    df: pd.DataFrame,
    selected_date: Any,
    *,
    exclude_selected_day: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """선택일/월간/연간 비교용 기간 데이터를 생성합니다."""
    if df is None or df.empty or "Date" not in df.columns:
        empty = pd.DataFrame()
        return empty, empty, empty, "", ""

    selected = pd.to_datetime(selected_date)
    dates = pd.to_datetime(df["Date"], errors="coerce")

    day = df[dates.dt.date == selected.date()].copy()
    month = df[(dates.dt.year == selected.year) & (dates.dt.month == selected.month)].copy()
    year = df[dates.dt.year == selected.year].copy()

    if exclude_selected_day:
        month = month[month["Date"] != selected_date].copy()
        year = year[year["Date"] != selected_date].copy()

    return day, month, year, f"{selected.year}-{selected.month:02d}", str(selected.year)
