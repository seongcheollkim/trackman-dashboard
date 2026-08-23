from __future__ import annotations

import math

import streamlit as st

from services.practice_journal_service import PracticeJournalService


@st.cache_resource
def _practice_journal_service(user_email: str) -> PracticeJournalService:
    return PracticeJournalService(user_email=user_email)


def _journal_club_name(name: str | None) -> str:
    if not name:
        return "-"

    labels = {
        "Driver": "드라이버",
        "3Wood": "3번 우드",
        "5Wood": "5번 우드",
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


def _journal_stars(value: int | None) -> str:
    score = max(0, min(int(value or 0), 5))
    return "★" * score + "☆" * (5 - score)


def _journal_summary_card(detail) -> str:
    score = "-" if detail.ai_score is None else f"{detail.ai_score:.1f}"
    grade = f" ({detail.ai_grade})" if detail.ai_grade else ""

    return (
        "<div class='journal-card'><div class='journal-summary-grid'>"
        "<div class='journal-kpi'><div class='journal-kpi-label'>Practice</div>"
        f"<div class='journal-kpi-value'>{detail.shot_count:,} Shots · "
        f"{detail.club_count} Clubs</div></div>"
        "<div class='journal-kpi'><div class='journal-kpi-label'>AI Score</div>"
        f"<div class='journal-kpi-value accent'>{score}{grade}</div></div>"
        "<div class='journal-kpi'><div class='journal-kpi-label'>Best Club</div>"
        f"<div class='journal-kpi-value'>{_journal_club_name(detail.best_club)}</div></div>"
        "<div class='journal-kpi'><div class='journal-kpi-label'>Focus Club</div>"
        f"<div class='journal-kpi-value'>{_journal_club_name(detail.focus_club)}</div></div>"
        "</div></div>"
    )


def _journal_ai_card(detail) -> str:
    lines = [
        f"<div class='journal-ai-line'>• {message}</div>"
        for message in detail.ai_strengths[:2]
    ]

    if detail.focus_club:
        lines.append(
            "<div class='journal-ai-line'>"
            f"• 다음 우선순위: <b>{_journal_club_name(detail.focus_club)}</b>"
            "</div>"
        )

    coach = (
        f"<div class='journal-ai-coach'>{detail.coaching_summary}</div>"
        if detail.coaching_summary
        else ""
    )

    if not lines and not coach:
        lines.append(
            "<div class='journal-ai-line'>저장된 AI 분석이 없습니다.</div>"
        )

    return (
        "<div class='journal-card'>"
        "<div class='journal-ai-title'>🤖 AI Coach Summary</div>"
        + "".join(lines)
        + coach
        + "</div>"
    )


def auto_textarea_height(
    text: str | None,
    min_lines: int = 4,
    max_lines: int = 40,
    chars_per_line: int = 55,
) -> int:
    """저장된 멀티라인 텍스트가 다시 열릴 때 내용 길이에 맞춰 초기 높이를 계산."""
    value = str(text or "")
    visual_lines = 0

    for line in value.split("\n"):
        # 실제 줄바꿈뿐 아니라 긴 한 줄이 화면 폭에서 자동 줄바꿈되는 경우도 반영
        wrapped_lines = max(
            1,
            math.ceil(max(1, len(line)) / chars_per_line),
        )
        visual_lines += wrapped_lines

    visual_lines = max(min_lines, min(visual_lines, max_lines))
    return 42 + (visual_lines * 24)


def render_practice_journal_tab(
    *,
    user_email: str,
    initial_date: str | None = None,
) -> None:
    """
    DODOS Golf Solution - Phase 3 Practice Journal UI.

    기존 streamlit_app.py의 연습일지 UI를 그대로 모듈화한 함수입니다.
    """
    with st.container(key="journal_scope"):
        st.markdown(
            "<div class='journal-title'>📓 연습 일지</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='journal-sub'>"
            "TrackMan 데이터와 AI 분석은 자동 연결되고, "
            "오늘의 체감과 메모만 기록합니다."
            "</div>",
            unsafe_allow_html=True,
        )

        try:
            service = _practice_journal_service(user_email)
            sessions = service.list_sessions(limit=72)
        except Exception as exc:
            st.error(f"연습 일지 DB 연결 실패: {exc}")
            return

        if not sessions:
            st.info("저장된 연습 세션이 없습니다.")
            return

        default_index = 0
        if initial_date:
            for idx, item in enumerate(sessions):
                if item.practice_date == str(initial_date):
                    default_index = idx
                    break

        def session_label(item) -> str:
            score = "-" if item.ai_score is None else f"{item.ai_score:.1f}"
            note = " · 📝" if item.has_note else ""
            return (
                f"{item.practice_date} · {item.shot_count}샷 · "
                f"{item.club_count} Clubs · AI {score}{note}"
            )

        nav_col, body_col = st.columns([1.0, 3.15], gap="large")

        with nav_col:
            selected_idx = st.selectbox(
                "최근 연습",
                range(len(sessions)),
                index=default_index,
                format_func=lambda idx: session_label(sessions[idx]),
                key="journal_session_select",
            )
            selected_session = sessions[int(selected_idx)]

            st.markdown(
                "<div class='journal-section-title'>최근 기록</div>",
                unsafe_allow_html=True,
            )

            for item in sessions[:7]:
                score = "-" if item.ai_score is None else f"{item.ai_score:.1f}"
                note = " · 일지 있음" if item.has_note else ""

                st.markdown(
                    "<div class='journal-history-card'>"
                    f"<div class='journal-history-date'>{item.practice_date}</div>"
                    f"<div class='journal-history-sub'>{item.shot_count}샷 · "
                    f"{item.club_count} Clubs · AI {score}{note}</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

        try:
            detail = service.get_journal_detail_by_session_id(
                selected_session.session_id
            )
        except Exception as exc:
            st.error(f"연습 일지 상세 조회 실패: {exc}")
            return

        if detail is None:
            st.warning("선택한 연습의 상세 정보를 찾지 못했습니다.")
            return

        with body_col:
            summary_col, ai_col = st.columns(
                [1.15, 1.85],
                gap="medium",
            )

            with summary_col:
                st.markdown(
                    _journal_summary_card(detail),
                    unsafe_allow_html=True,
                )

            with ai_col:
                st.markdown(
                    _journal_ai_card(detail),
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='journal-section-title'>오늘의 체감</div>",
                unsafe_allow_html=True,
            )

            condition_col, satisfaction_col = st.columns(2)

            with condition_col:
                condition = st.radio(
                    "컨디션",
                    [1, 2, 3, 4, 5],
                    index=max(0, (detail.condition_score or 3) - 1),
                    format_func=_journal_stars,
                    horizontal=True,
                    key=f"journal_condition::{detail.session_id}",
                )

            with satisfaction_col:
                satisfaction = st.radio(
                    "만족도",
                    [1, 2, 3, 4, 5],
                    index=max(0, (detail.satisfaction_score or 3) - 1),
                    format_func=_journal_stars,
                    horizontal=True,
                    key=f"journal_satisfaction::{detail.session_id}",
                )

            practice_goal = st.text_area(
                "오늘의 연습 목표",
                value=detail.practice_goal,
                height=auto_textarea_height(
                    detail.practice_goal,
                    4,
                ),
            )

            memo_col, lesson_col = st.columns(2)

            with memo_col:
                memo = st.text_area(
                    "오늘 메모",
                    value=detail.memo,
                    height=auto_textarea_height(
                        detail.memo,
                        6,
                    ),
                )

            with lesson_col:
                lesson_note = st.text_area(
                    "레슨 / 스윙 노트",
                    value=detail.lesson_note,
                    height=auto_textarea_height(
                        detail.lesson_note,
                        6,
                    ),
                )

            save_col, state_col = st.columns(
                [1.0, 2.6],
                vertical_alignment="bottom",
            )

            with save_col:
                save_clicked = st.button(
                    "💾 연습 일지 저장",
                    type="primary",
                    width="stretch",
                    key=f"journal_save::{detail.session_id}",
                )

            with state_col:
                if detail.condition_score or detail.satisfaction_score:
                    st.caption(
                        "저장된 기록 · "
                        f"컨디션 {_journal_stars(detail.condition_score)} · "
                        f"만족도 {_journal_stars(detail.satisfaction_score)}"
                    )
                else:
                    st.caption("아직 저장된 개인 기록이 없습니다.")

            if save_clicked:
                try:
                    service.save_journal(
                        session_id=detail.session_id,
                        practice_date=detail.practice_date,
                        condition_score=int(condition),
                        satisfaction_score=int(satisfaction),
                        practice_goal=practice_goal,
                        memo=memo,
                        lesson_note=lesson_note,
                    )
                    st.toast(
                        "연습 일지를 저장했습니다.",
                        icon="✅",
                    )
                    st.success(
                        f"{detail.practice_date} 연습 일지가 저장되었습니다."
                    )
                except Exception as exc:
                    st.error(f"연습 일지 저장 실패: {exc}")
