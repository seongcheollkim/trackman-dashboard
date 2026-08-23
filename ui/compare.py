from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


def _format_value(value: Any, decimals: int, unit: str) -> str:
    if value is None:
        return "-"
    try:
        if decimals == 0:
            text = f"{int(round(float(value))):,}"
        else:
            text = f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "-"
    return f"{text}{unit}"


def _format_delta(value: Any, decimals: int, unit: str) -> str:
    if value is None:
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    sign = "+" if numeric > 0 else ""
    if decimals == 0:
        text = f"{sign}{int(round(numeric)):,}"
    else:
        text = f"{sign}{numeric:.{decimals}f}"
    return f"{text}{unit}"


def _status_class(status: str) -> str:
    return {
        "good": "tm-good",
        "bad": "tm-bad",
    }.get(status, "tm-neutral")


def render_compare_cards(payload: list[dict[str, Any]]) -> None:
    """compare_service가 만든 순수 데이터를 화면 카드로 렌더링합니다."""
    cards: list[str] = []

    for item in payload:
        comparison = item.get("comparison", {})
        month = comparison.get("month", {})
        year = comparison.get("year", {})
        decimals = int(item.get("decimals", 1))
        unit = str(item.get("unit", ""))
        title = escape(str(item.get("title", "")))

        cards.append(
            "<div class='tm-compare-card'>"
            f"<div class='tm-compare-title'>{title}"
            f"<span style='float:right;color:#9fb0c2;font-weight:500'>{escape(unit)}</span></div>"
            "<div class='tm-compare-values'>"
            f"<div><div class='tm-compare-value tm-day'>{_format_value(comparison.get('day'), decimals, unit)}</div><div class='tm-compare-label'>선택일</div></div>"
            f"<div><div class='tm-compare-value tm-month'>{_format_value(month.get('value'), decimals, unit)}</div><div class='tm-compare-label'>월간 평균</div></div>"
            f"<div><div class='tm-compare-value tm-year'>{_format_value(year.get('value'), decimals, unit)}</div><div class='tm-compare-label'>연간 평균</div></div>"
            "</div>"
            "<div class='tm-deltas'>"
            f"<span>vs 월간 <b class='{_status_class(month.get('status'))}'>{_format_delta(month.get('delta'), decimals, unit)}</b></span>"
            f"<span>vs 연간 <b class='{_status_class(year.get('status'))}'>{_format_delta(year.get('delta'), decimals, unit)}</b></span>"
            "</div></div>"
        )

    if cards:
        st.markdown(
            "<div class='tm-compare-grid'>" + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )
