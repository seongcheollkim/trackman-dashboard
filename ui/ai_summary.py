from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ai import diagnose_practice, diagnosis_html, goal_options


def render_ai_summary(
    df: pd.DataFrame,
    selected_date: str,
) -> Any:
    """AI 스윙 진단 탭의 상단 종합 리포트를 렌더링하고 ai_report를 반환합니다."""
    st.markdown(
        "<div class='tm-title'>🤖 AI 스윙 진단</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "오늘 연습한 전체 클럽을 종합해 클럽군별 품질, 베스트 클럽, 우선 개선 클럽과 "
        "다음 연습 방향을 요약합니다."
    )

    goal_col, guide_col = st.columns(
        [1.15, 2.85],
        vertical_alignment="bottom",
    )

    with goal_col:
        ai_goal_labels = goal_options()
        ai_goal_keys = list(ai_goal_labels.keys())
        ai_goal = st.selectbox(
            "🎯 목표 수준",
            ai_goal_keys,
            index=(
                ai_goal_keys.index("single")
                if "single" in ai_goal_keys
                else 0
            ),
            format_func=lambda key: ai_goal_labels[key],
            key="dodos_ai_goal",
        )

    with guide_col:
        st.caption(
            "AI 종합 점수는 Performance 50% · Consistency 30% · Trend 20%로 계산하며, "
            "상단 리포트에서는 이 계산식보다 오늘 무엇을 유지하고 무엇을 보완할지를 우선 보여줍니다."
        )

    ai_report = diagnose_practice(
        df,
        selected_date,
        goal=ai_goal,
        recent_sessions=10,
        min_shots_per_club=2,
    )

    st.markdown(
        diagnosis_html(ai_report),
        unsafe_allow_html=True,
    )

    return ai_report
