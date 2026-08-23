from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


def mean_existing_column(df: pd.DataFrame, candidates: list[str]) -> float:
    """후보 컬럼 중 존재하면서 유효한 첫 번째 컬럼의 평균을 반환."""
    for column in candidates:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return float("nan")


def fmt(v: Any, nd: int = 1, suffix: str = "") -> str:
    try:
        if pd.isna(v):
            return "-"
        return f"{float(v):.{nd}f}{suffix}"
    except Exception:
        return "-"


def fmt_int(v: Any, comma: bool = False) -> str:
    """화면 표시용 정수 반올림."""
    try:
        if pd.isna(v):
            return "-"
        value = int(round(float(v)))
        return f"{value:,}" if comma else str(value)
    except Exception:
        return "-"


def side_text(v: Any) -> str:
    try:
        x = float(v)
        if abs(x) < 0.5:
            return "0"
        return f"{abs(int(round(x)))}{'R' if x > 0 else 'L'}"
    except Exception:
        return "-"


def render_top_metrics(items: list[tuple[str, str, str]]) -> None:
    """상단 KPI 카드를 Markdown 코드 블록으로 오인하지 않도록 한 줄 HTML로 렌더링."""
    cards: list[str] = []

    for label, value, unit in items:
        unit_html = f"<span class='tm-kpi-unit'>{unit}</span>" if unit else ""
        cards.append(
            "<div class='tm-kpi-card'>"
            f"<div class='tm-kpi-label'>{label}</div>"
            f"<div class='tm-kpi-value'>{value}{unit_html}</div>"
            "</div>"
        )

    cards_html = "".join(cards)
    st.markdown(
        f"<div class='tm-kpi-grid'>{cards_html}</div>",
        unsafe_allow_html=True,
    )


def classify_face_to_path(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "-"
    if v > 0.8:
        return "오픈"
    if v < -0.8:
        return "클로즈"
    return "중립"


def classify_path(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "-"
    if v > 1.0:
        return "인-아웃"
    if v < -1.0:
        return "아웃-인"
    return "중립"
