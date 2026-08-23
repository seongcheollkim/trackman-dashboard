from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st


def render_average_analysis(
    *,
    club: str,
    selected_date: str,
    month_label: str,
    year_label: str,
    mode: str,
    day_df: pd.DataFrame,
    month_df: pd.DataFrame,
    year_df: pd.DataFrame,
    day_summary: pd.Series,
    month_summary: pd.Series,
    year_summary: pd.Series,
    filtered_df: pd.DataFrame,
    clubs: list[str],
    dates: list[str],
    render_compare_cards: Callable[..., None],
    club_korean_name: Callable[[str], str],
    distance_chart: Callable[..., None],
    side_chart: Callable[..., None],
    trend_chart: Callable[..., None],
    impact_face_fig: Callable[..., Any],
    club_path_fig: Callable[..., Any],
    loft_spin_fig: Callable[..., Any],
    period_row: Callable[..., pd.Series],
    auto_text: Callable[..., str],
    make_summary: Callable[..., list[dict]],
    summary_columns: list[str],
    club_sort_key: Callable[[str], Any],
    render_club_cards: Callable[..., None],
    safe_dataframe_for_streamlit: Callable[[pd.DataFrame], pd.DataFrame],
    render_dark_dataframe: Callable[..., None],
) -> None:
    """
    평균 분석 탭 UI coordinator.

    Step 6에서는 기능/화면을 변경하지 않고, 기존 streamlit_app.py의
    평균 분석 탭 조립 코드만 ui/average_analysis.py로 분리합니다.
    평균 분석 전용 helper는 아직 기존 모듈에 두고 명시적으로 전달합니다.
    """
    st.markdown(
        f"<div class='tm-title'>기간 비교 ({club})</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='tm-legend'>"
        f"<span><i class='tm-dot tm-dot-day'></i>선택일 ({selected_date})</span>"
        f"<span><i class='tm-dot tm-dot-month'></i>월간 평균 ({month_label})</span>"
        f"<span><i class='tm-dot tm-dot-year'></i>연간 평균 ({year_label})</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    render_compare_cards(day_summary, month_summary, year_summary)

    # 세 패널의 제목/컨트롤 행을 먼저 만들고 그래프 행을 분리해 상단 정렬을 맞춥니다.
    compare_headers = st.columns(3)

    with compare_headers[0]:
        title_col, control_col = st.columns(
            [1.0, 0.72],
            vertical_alignment="center",
        )
        with title_col:
            st.markdown(
                "<div class='tm-panel-title'>거리 분포 비교</div>",
                unsafe_allow_html=True,
            )
        with control_col:
            avg_distance_choice = st.radio(
                "거리 기준",
                ["캐리", "토탈"],
                horizontal=True,
                key=f"avg_distance_metric::{club}::{selected_date}",
                label_visibility="collapsed",
            )

    with compare_headers[1]:
        st.markdown(
            f"<div class='tm-panel-title'>{club_korean_name(club)} 탄착군 비교</div>",
            unsafe_allow_html=True,
        )

    with compare_headers[2]:
        distance_label = "캐리" if avg_distance_choice == "캐리" else "토탈"
        st.markdown(
            f"<div class='tm-panel-title'>월별 {distance_label} 추세</div>",
            unsafe_allow_html=True,
        )

    avg_distance_metric = (
        "Carry_m" if avg_distance_choice == "캐리" else "Total_m"
    )

    compare_cols = st.columns(3)
    with compare_cols[0]:
        distance_chart(
            day_df,
            month_df,
            year_df,
            club,
            avg_distance_metric,
        )
    with compare_cols[1]:
        side_chart(
            day_df,
            month_df,
            year_df,
            club,
            avg_distance_metric,
        )
    with compare_cols[2]:
        trend_chart(
            filtered_df,
            club,
            pd.to_datetime(selected_date).year,
            mode,
            avg_distance_metric,
        )

    periods = [
        ("선택일", selected_date, day_df, day_summary, "전체 샷 평균"),
        ("월간 평균", month_label, month_df, month_summary, mode),
        ("연간 평균", year_label, year_df, year_summary, mode),
    ]

    st.markdown("### 임팩트 위치 비교")
    cols = st.columns(3)
    for column, (title, label, raw_df, summary, period_mode) in zip(cols, periods):
        with column:
            st.markdown(f"**{title} ({label})**")
            if summary.empty:
                st.info("비교 데이터 없음")
            else:
                st.pyplot(
                    impact_face_fig(
                        summary.get("Avg_ImpactOffset_mm"),
                        summary.get("Avg_ImpactHeight_mm"),
                        club,
                        raw_df[raw_df["Club"] == club],
                        figsize=(5, 3.2),
                    ),
                    clear_figure=True,
                )

    st.markdown("### 클럽 패스 비교")
    cols = st.columns(3)
    for column, (title, label, raw_df, summary, period_mode) in zip(cols, periods):
        with column:
            st.markdown(f"**{title} ({label})**")
            if summary.empty:
                st.info("비교 데이터 없음")
            else:
                st.pyplot(
                    club_path_fig(
                        period_row(raw_df, summary, club, period_mode),
                        figsize=(4.8, 3.4),
                    ),
                    clear_figure=True,
                )

    st.markdown("### 런치 앵글 / 스핀 로프트 비교")
    cols = st.columns(3)
    for column, (title, label, raw_df, summary, period_mode) in zip(cols, periods):
        with column:
            st.markdown(f"**{title} ({label})**")
            if summary.empty:
                st.info("비교 데이터 없음")
            else:
                st.pyplot(
                    loft_spin_fig(
                        period_row(raw_df, summary, club, period_mode),
                        figsize=(4.8, 3.4),
                    ),
                    clear_figure=True,
                )

    st.markdown("### 자동 분석 요약")
    st.markdown(
        f"<div class='tm-auto-summary'>"
        f"{auto_text(day_summary, month_summary, year_summary, club)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # 상세 분석 화면에서 유용했던 기능은 평균 분석 하단의 접이식 메뉴로 통합합니다.
    with st.expander("전체 클럽 요약", expanded=False):
        selected_day_all_clubs = filtered_df[
            filtered_df["Date"] == selected_date
        ].copy()

        all_club_summary = pd.DataFrame(
            make_summary(selected_day_all_clubs.to_dict("records"))
        )

        if all_club_summary.empty:
            st.info("선택한 날짜의 클럽 요약 데이터가 없습니다.")
        else:
            all_club_summary = all_club_summary[
                [
                    column
                    for column in summary_columns
                    if column in all_club_summary.columns
                ]
            ]

            if "Club" in all_club_summary.columns:
                all_club_summary = (
                    all_club_summary.assign(
                        _club_order=all_club_summary["Club"].map(club_sort_key)
                    )
                    .sort_values("_club_order")
                    .drop(columns="_club_order")
                )

            render_club_cards(all_club_summary, max_cards=6)

            st.dataframe(
                safe_dataframe_for_streamlit(all_club_summary),
                width="stretch",
                hide_index=True,
            )

    with st.expander("원본 샷 데이터", expanded=False):
        raw_filter_col1, raw_filter_col2 = st.columns(2)

        with raw_filter_col1:
            raw_clubs = st.multiselect(
                "클럽",
                clubs,
                default=[club],
                key=f"raw_shot_clubs::{club}::{selected_date}",
            )

        with raw_filter_col2:
            raw_dates = st.multiselect(
                "날짜",
                sorted(dates, reverse=True),
                default=[selected_date],
                key=f"raw_shot_dates::{club}::{selected_date}",
            )

        raw_df = filtered_df.copy()

        if raw_clubs:
            raw_df = raw_df[raw_df["Club"].isin(raw_clubs)]

        if raw_dates:
            raw_df = raw_df[raw_df["Date"].isin(raw_dates)]

        raw_columns = [
            column
            for column in [
                "Club",
                "StrokeNo",
                "Date",
                "ShotTimeLocal",
                "Carry_m",
                "Total_m",
                "Run_m",
                "BallSpeed_mps",
                "ClubSpeed_mps",
                "SmashFactor",
                "SpinRate_rpm",
                "LaunchAngle_deg",
                "AttackAngle_deg",
                "ClubPath_deg",
                "FaceAngle_deg",
                "FaceToPath_deg",
                "TotalSide_m",
                "ImpactOffset_mm",
                "ImpactHeight_mm",
            ]
            if column in raw_df.columns
        ]

        sort_columns = [
            column
            for column in ["Date", "Club", "StrokeNo"]
            if column in raw_df.columns
        ]

        if not raw_df.empty and sort_columns:
            ascending = [False, True, True][: len(sort_columns)]
            raw_df = raw_df.sort_values(
                sort_columns,
                ascending=ascending,
            )

        st.caption(f"표시 샷 수: {len(raw_df):,}개")

        raw_display_df = raw_df.loc[:, raw_columns].reset_index(drop=True)
        csv_bytes = raw_display_df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            "CSV 다운로드",
            data=csv_bytes,
            file_name=f"trackman_raw_{selected_date}.csv",
            mime="text/csv",
            key=f"raw_csv::{club}::{selected_date}",
        )

        render_dark_dataframe(raw_display_df)
