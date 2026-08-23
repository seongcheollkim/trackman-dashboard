from __future__ import annotations

import streamlit as st


def _club_korean_name(name: str) -> str:
    labels = {
        "Driver": "드라이버",
        "3Wood": "3번 우드",
        "5Wood": "5번 우드",
        "7Wood": "7번 우드",
        "4Hybrid": "4번 유틸리티",
        "5Hybrid": "5번 유틸리티",
        "PitchingWedge": "피칭 웨지",
        "GapWedge": "갭 웨지",
        "SandWedge": "샌드 웨지",
        "50Wedge": "50도 웨지",
        "52Wedge": "52도 웨지",
        "54Wedge": "54도 웨지",
        "56Wedge": "56도 웨지",
        "58Wedge": "58도 웨지",
    }

    if name in labels:
        return labels[name]

    if str(name).endswith("Iron"):
        return f"{str(name)[:-4]}번 아이언"

    return str(name)


def render_ai_club_detail(
    *,
    ai_report,
    selected_club: str,
    selected_date: str,
) -> None:
    if not ai_report.clubs:
        return

    with st.container(key="ai_club_detail"):
        st.markdown(
            "<div class='ai-club-detail-title'>클럽별 상세 진단</div>",
            unsafe_allow_html=True,
        )

        club_names = [item.club for item in ai_report.clubs]
        default_index = (
            club_names.index(selected_club)
            if selected_club in club_names
            else 0
        )

        ai_selected_club = st.selectbox(
            "클럽별 AI 평가",
            club_names,
            index=default_index,
            format_func=_club_korean_name,
            key=f"ai_club::{selected_date}",
        )

        club_report = next(
            item for item in ai_report.clubs
            if item.club == ai_selected_club
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI 점수", f"{club_report.score:.0f}/100")
        c2.metric("등급", club_report.grade)
        c3.metric("신뢰도", f"{club_report.confidence}%")
        c4.metric("분석 샷", f"{club_report.shots}개")

        b1, b2, b3 = st.columns(3)
        b1.metric("Performance", f"{club_report.performance_score:.0f}/100")
        b2.metric("Consistency", f"{club_report.consistency_score:.0f}/100")
        b3.metric("Trend", f"{club_report.trend_score:.0f}/100")

        if (
            "Wedge" in ai_selected_club
            or ai_selected_club in {
                "PitchingWedge",
                "GapWedge",
                "SandWedge",
            }
        ):
            wedge_bucket_count = club_report.metrics.get("wedge_valid_buckets")
            wedge_radius = club_report.metrics.get("wedge_dispersion_radius")
            wedge_lateral = club_report.metrics.get("wedge_lateral_abs_mean")
            wedge_miss = club_report.metrics.get("wedge_big_miss_rate_pct")

            if wedge_bucket_count and wedge_bucket_count > 0:
                st.caption("웨지 v5 · 10m 거리대별 탄착군 평가")

                w1, w2, w3, w4 = st.columns(4)
                w1.metric("평가 거리대", f"{int(wedge_bucket_count)}개")
                w2.metric(
                    "68% 탄착군 반경",
                    "-" if wedge_radius is None else f"{float(wedge_radius):.1f}m",
                )
                w3.metric(
                    "평균 좌우 편차",
                    "-" if wedge_lateral is None else f"{float(wedge_lateral):.1f}m",
                )
                w4.metric(
                    "큰 미스 비율",
                    "-" if wedge_miss is None else f"{float(wedge_miss):.0f}%",
                )
            else:
                st.info(
                    "웨지 전용 탄착군 평가를 위해 같은 10m 거리대에 최소 3샷이 필요합니다. "
                    "현재는 일반 평가를 참고용으로 표시합니다."
                )

        detail_left, detail_right = st.columns(2)

        with detail_left:
            st.markdown("#### 장점")
            if club_report.strengths:
                for message in club_report.strengths:
                    st.success(message)
            else:
                st.info("뚜렷한 우위 지표가 없습니다.")

        with detail_right:
            st.markdown("#### 개선 필요")
            if club_report.improvements:
                for message in club_report.improvements:
                    st.warning(message)
            else:
                st.success("최근 기준 대비 뚜렷한 악화 지표가 없습니다.")

        st.markdown("#### 추천 연습")
        for task in club_report.tasks:
            st.markdown(f"- {task}")

        if club_report.baseline_sessions < 3:
            st.caption(
                "※ 해당 클럽의 과거 비교 세션이 3회 미만이므로 "
                "현재 점수는 절대 평가보다 참고용 성격이 강합니다."
            )
