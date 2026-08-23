from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from ui.common import render_top_metrics


def render_single_shot_analysis(
    day_df: pd.DataFrame,
    month_df: pd.DataFrame,
    year_df: pd.DataFrame,
    club: str,
    selected_date: str,
    day_summary: pd.Series,
    month_summary: pd.Series,
    year_summary: pd.Series,
    *,
    shot_sort_columns: Callable[[pd.DataFrame], list[str]],
    shot_display_number: Callable[[pd.Series, int], Any],
    club_korean_name: Callable[[str], str],
    render_clickable_shot_distribution: Callable[..., int],
    impact_face_fig: Callable[..., Any],
    club_path_fig: Callable[..., Any],
    loft_spin_fig: Callable[..., Any],
    shot_metric_items: Callable[[pd.Series], list[tuple[str, str, str]]],
    render_shot_detail_panel: Callable[..., None],
    shot_table: Callable[..., int],
    render_shot_compare_cards: Callable[..., None],
) -> None:
    """
    샷별 상세 분석 UI coordinator.

    Step 4에서는 기능 변경 없이 기존 streamlit_app.py의 화면 조립 로직만
    ui/shot_analysis.py로 분리합니다. 그래프/테이블 helper는 아직 기존 모듈에
    유지하고 명시적인 dependency injection으로 전달합니다.
    """
    shots = day_df[day_df["Club"] == club].copy()
    sort_columns = shot_sort_columns(shots)
    if sort_columns:
        shots = shots.sort_values(sort_columns, kind="stable")
    shots = shots.reset_index(drop=True)

    if shots.empty:
        st.info("선택한 날짜에 해당 클럽의 샷 데이터가 없습니다.")
        return

    state_key = f"shot_index::{club}::{selected_date}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    st.session_state[state_key] = min(
        max(int(st.session_state[state_key]), 0),
        len(shots) - 1,
    )

    selected_index = st.session_state[state_key]
    row = shots.iloc[selected_index]
    shot_no = shot_display_number(row, selected_index)
    shot_time = str(row.get("ShotTimeLocal", "") or "")
    club_title = club_korean_name(club)

    st.markdown(
        f"<div class='tm-shot-heading'>"
        f"<div>"
        f"<div class='tm-shot-heading-title'>{club_title} 샷별 상세 분석</div>"
        f"<div class='tm-shot-heading-sub'>"
        f"선택 Shot {shot_no} · {shot_time} · {selected_index + 1}/{len(shots)}"
        f"</div>"
        f"</div>"
        f"<div class='tm-shot-heading-sub'>{selected_date} · 총 {len(shots)}샷</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    visual_cols = st.columns([1.18, 0.92, 0.92, 0.92])

    with visual_cols[0]:
        st.markdown(
            f"<div class='tm-panel-title'>{club_title} 탄착군</div>",
            unsafe_allow_html=True,
        )
        clicked_scatter_index = render_clickable_shot_distribution(
            shots,
            selected_index,
            key=f"shot_scatter::{club}::{selected_date}",
            distance_metric="Carry_m",
        )
        if clicked_scatter_index != selected_index:
            st.session_state[state_key] = clicked_scatter_index
            st.rerun()

    with visual_cols[1]:
        st.markdown(
            "<div class='tm-panel-title'>선택 샷 임팩트 위치</div>",
            unsafe_allow_html=True,
        )
        st.pyplot(
            impact_face_fig(
                row.get("ImpactOffset_mm"),
                row.get("ImpactHeight_mm"),
                club,
                points=None,
                figsize=(5.0, 3.45),
            ),
            clear_figure=True,
        )

    with visual_cols[2]:
        st.markdown(
            "<div class='tm-panel-title'>선택 샷 클럽 패스</div>",
            unsafe_allow_html=True,
        )
        st.pyplot(
            club_path_fig(row, figsize=(4.8, 3.45)),
            clear_figure=True,
        )

    with visual_cols[3]:
        st.markdown(
            "<div class='tm-panel-title'>선택 샷 로프트 / 스핀 로프트</div>",
            unsafe_allow_html=True,
        )
        st.pyplot(
            loft_spin_fig(row, figsize=(4.8, 3.45)),
            clear_figure=True,
        )

    st.markdown(
        "<div class='tm-shot-section-title'>선택 샷 상세 데이터</div>",
        unsafe_allow_html=True,
    )
    render_top_metrics(shot_metric_items(row))
    render_shot_detail_panel(
        row,
        state_suffix=f"{club}::{selected_date}",
    )

    st.markdown("### 샷 목록")
    control_left, control_right = st.columns([1.2, 3.8])

    with control_left:
        sort_by_ai_low = st.checkbox(
            "AI 낮은 순 보기",
            value=False,
            key=f"ai_sort::{club}::{selected_date}",
            help="개선이 필요한 샷부터 테이블 위에 표시합니다.",
        )

    with control_right:
        st.caption(
            "표의 행 또는 위 탄착군의 점을 클릭하면 같은 샷이 선택됩니다. "
            "Excellent 85~100 · Good 70~84 · Poor 50~69 · Miss 0~49"
        )

    clicked_index = shot_table(
        shots,
        selected_index,
        key=f"shot_table::{club}::{selected_date}",
        sort_by_ai_low=sort_by_ai_low,
    )
    if clicked_index != selected_index:
        st.session_state[state_key] = clicked_index
        st.rerun()

    st.markdown("### 선택 샷 vs 당일·월간·연간 평균")
    render_shot_compare_cards(
        row,
        day_summary,
        month_summary,
        year_summary,
    )
