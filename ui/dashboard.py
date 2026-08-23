from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def _fmt(value: Any, decimals: int = 1, suffix: str = "") -> str:
    """Dashboard 표시용 안전한 숫자 포맷터."""
    try:
        if pd.isna(value):
            return "-"
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "-"


def render_kpi_cards(items: list[tuple[str, str, str]]) -> None:
    """Dashboard 상단 KPI 카드만 렌더링합니다.

    기존 ui.common.render_top_metrics의 화면 동작을 그대로 유지하면서
    Dashboard 전용 UI 책임을 별도 모듈로 분리합니다.
    """
    cards: list[str] = []

    for label, value, unit in items:
        unit_html = f"<span class='tm-kpi-unit'>{unit}</span>" if unit else ""
        cards.append(
            "<div class='tm-kpi-card'>"
            f"<div class='tm-kpi-label'>{label}</div>"
            f"<div class='tm-kpi-value'>{value}{unit_html}</div>"
            "</div>"
        )

    st.markdown(
        "<div class='tm-kpi-grid'>"
        + "".join(cards)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_dashboard_kpis(
    *,
    carry: Any = None,
    total: Any = None,
    ball_speed: Any = None,
    club_speed: Any = None,
    smash: Any = None,
    spin: Any = None,
    launch: Any = None,
    side: Any = None,
    apex: Any = None,
) -> None:
    """표준 Dashboard KPI 9개를 렌더링합니다.

    계산 로직은 포함하지 않습니다. KPI 계산은 이후 Step 7-4
    metrics_service.py로 분리하고, 이 모듈은 화면 표시만 담당합니다.
    """
    items = [
        ("Carry", _fmt(carry), "m"),
        ("Total", _fmt(total), "m"),
        ("Ball Speed", _fmt(ball_speed), "m/s"),
        ("Club Speed", _fmt(club_speed), "m/s"),
        ("Smash", _fmt(smash, 2), ""),
        ("Spin", _fmt(spin, 0), "rpm"),
        ("Launch", _fmt(launch), "°"),
        ("Side", _fmt(side), "m"),
        ("Apex", _fmt(apex), "m"),
    ]
    render_kpi_cards(items)


def render_dashboard_section_title(title: str, subtitle: str | None = None) -> None:
    """Dashboard 섹션 제목을 일관된 스타일로 렌더링합니다."""
    st.markdown(f"<div class='tm-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)
