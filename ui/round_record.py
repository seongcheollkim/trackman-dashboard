from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from services.hole19_service import Hole19Round, fetch_hole19_round
from services.round_service import RoundService


def _round_service(user_email: str) -> RoundService:
    return RoundService(user_email=user_email)


def _empty_holes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "홀": i,
                "Par": 4,
                "Score": None,
                "Putts": None,
                "FIR": None,
                "GIR": None,
                "Sand": 0,
                "Penalty": 0,
            }
            for i in range(1, 19)
        ]
    )


def _holes_from_imported(round_data: Hole19Round) -> pd.DataFrame:
    rows = []
    for hole in round_data.holes:
        fir = hole.get("fir")
        if isinstance(fir, str):
            fir_label = fir
        elif fir is None:
            fir_label = None
        else:
            fir_label = "hit" if fir else "miss"
        rows.append(
            {
                "홀": hole.get("hole_number"),
                "Par": hole.get("par"),
                "Score": hole.get("score"),
                "Putts": hole.get("putts"),
                "FIR": fir_label,
                "GIR": hole.get("gir"),
                "Sand": hole.get("sand_shots") or 0,
                "Penalty": hole.get("penalties") or 0,
            }
        )
    return pd.DataFrame(rows)


def _normalize_holes_editor(df: pd.DataFrame) -> list[dict[str, Any]]:
    holes: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        hole_number = row.get("홀")
        if pd.isna(hole_number):
            continue
        def clean(value):
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
            return value

        holes.append(
            {
                "hole_number": int(hole_number),
                "par": int(clean(row.get("Par")) or 0),
                "score": int(clean(row.get("Score"))) if clean(row.get("Score")) is not None else None,
                "putts": int(clean(row.get("Putts"))) if clean(row.get("Putts")) is not None else None,
                "fir": clean(row.get("FIR")),
                "gir": clean(row.get("GIR")),
                "sand_shots": int(clean(row.get("Sand")) or 0),
                "penalties": int(clean(row.get("Penalty")) or 0),
            }
        )
    return holes


def _round_metrics(holes: list[dict[str, Any]]) -> tuple[int | None, int | None, float | None, float | None, int]:
    scores = [h["score"] for h in holes if h.get("score") is not None]
    putts = [h["putts"] for h in holes if h.get("putts") is not None]
    penalty = sum(int(h.get("penalties") or 0) for h in holes)
    par3 = [h for h in holes if int(h.get("par") or 0) > 3 and h.get("fir") not in (None, "")]
    fir_hits = sum(1 for h in par3 if str(h.get("fir")).lower() not in {"miss", "false", "0", "none"})
    gir_rows = [h for h in holes if h.get("gir") is not None]
    gir_hits = sum(1 for h in gir_rows if bool(h.get("gir")))
    fir_pct = round(fir_hits / len(par3) * 100, 1) if par3 else None
    gir_pct = round(gir_hits / len(gir_rows) * 100, 1) if gir_rows else None
    return (sum(scores) if scores else None, sum(putts) if putts else None, fir_pct, gir_pct, penalty)


def _save_round(
    *,
    user_email: str,
    round_date: str,
    course_name: str,
    tee_name: str,
    course_par: int | None,
    scoring_mode: str | None,
    playing_hcp: int | None,
    holes: list[dict[str, Any]],
    source: str,
    source_round_id: str | None,
    source_url: str | None,
    best_shot: str,
    weakness: str,
    next_goal: str,
    notes: str,
) -> None:
    service = _round_service(user_email)
    service.save_round(
        round_date=round_date,
        course_name=course_name,
        tee_name=tee_name,
        course_par=course_par,
        scoring_mode=scoring_mode,
        playing_hcp=playing_hcp,
        holes=holes,
        source=source,
        source_round_id=source_round_id,
        source_url=source_url,
        best_shot=best_shot,
        weakness=weakness,
        next_goal=next_goal,
        notes=notes,
    )


def _render_summary(holes: list[dict[str, Any]]) -> None:
    score, putts, fir, gir, penalties = _round_metrics(holes)
    cols = st.columns(5)
    values = [
        ("Score", "-" if score is None else str(score)),
        ("Putts", "-" if putts is None else str(putts)),
        ("FIR", "-" if fir is None else f"{fir:.1f}%"),
        ("GIR", "-" if gir is None else f"{gir:.1f}%"),
        ("Penalty", str(penalties)),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            st.metric(label, value)


def _render_manual(user_email: str) -> None:
    st.markdown("#### 직접 입력")
    st.caption("필드에서 기록한 내용을 18홀 단위로 직접 입력합니다.")

    with st.form("round_manual_form"):
        top = st.columns([1.0, 1.8, 1.0, 1.0])
        with top[0]:
            round_date = st.date_input("라운드 날짜", value=date.today())
        with top[1]:
            course_name = st.text_input("골프장", placeholder="예: 솔라고 CC")
        with top[2]:
            tee_name = st.text_input("티박스", placeholder="White")
        with top[3]:
            playing_hcp = st.number_input("핸디캡", min_value=0, max_value=54, value=0, step=1)

        st.markdown("##### 홀별 기록")
        edited = st.data_editor(
            _empty_holes(),
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            column_config={
                "홀": st.column_config.NumberColumn("홀", disabled=True, width="small"),
                "Par": st.column_config.NumberColumn("Par", min_value=1, max_value=8, step=1),
                "Score": st.column_config.NumberColumn("Score", min_value=1, max_value=20, step=1),
                "Putts": st.column_config.NumberColumn("Putts", min_value=0, max_value=10, step=1),
                "FIR": st.column_config.SelectboxColumn("FIR", options=["hit", "left", "right", "miss"], required=False),
                "GIR": st.column_config.CheckboxColumn("GIR"),
                "Sand": st.column_config.NumberColumn("Sand", min_value=0, max_value=10, step=1),
                "Penalty": st.column_config.NumberColumn("Penalty", min_value=0, max_value=10, step=1),
            },
            key="round_manual_holes",
        )

        holes = _normalize_holes_editor(edited)
        _render_summary(holes)

        note_col, goal_col = st.columns(2)
        with note_col:
            notes = st.text_area("라운드 메모", height=100)
            best_shot = st.text_input("베스트 샷")
            weakness = st.text_input("아쉬운 점")
        with goal_col:
            next_goal = st.text_area("다음 라운드 목표", height=100)

        submitted = st.form_submit_button("💾 라운드 저장", type="primary", width="stretch")

    if submitted:
        if not course_name.strip():
            st.error("골프장명을 입력해 주세요.")
            return
        if not any(h.get("score") is not None for h in holes):
            st.error("최소 한 홀의 Score를 입력해 주세요.")
            return
        try:
            _save_round(
                user_email=user_email,
                round_date=round_date.isoformat(),
                course_name=course_name,
                tee_name=tee_name,
                course_par=sum(int(h.get("par") or 0) for h in holes) or None,
                scoring_mode="manual",
                playing_hcp=int(playing_hcp) if playing_hcp else None,
                holes=holes,
                source="manual",
                source_round_id=None,
                source_url=None,
                best_shot=best_shot,
                weakness=weakness,
                next_goal=next_goal,
                notes=notes,
            )
            st.success("라운드를 저장했습니다.")
        except Exception as exc:
            st.error(f"라운드 저장 실패: {exc}")
            st.caption("Supabase에 Phase 4 라운드 테이블을 먼저 생성했는지 확인해 주세요.")


def _render_import(user_email: str) -> None:
    st.markdown("#### Hole19에서 가져오기")
    st.caption("Hole19 라운드 URL을 붙여 넣으면 페이지에 포함된 라운드 데이터를 읽어옵니다.")

    url = st.text_input(
        "Hole19 라운드 URL",
        placeholder="https://www.hole19golf.com/performance/rounds/dvvRBQ",
        key="hole19_round_url",
    )

    if st.button("🔎 라운드 불러오기", type="primary", width="stretch"):
        if not url.strip():
            st.error("Hole19 라운드 URL을 입력해 주세요.")
            return
        try:
            imported = fetch_hole19_round(url)
            st.session_state["hole19_imported_round"] = imported
            st.success(
                f"가져오기 성공 · {imported.course_name} · {len(imported.holes)}홀"
            )
        except Exception as exc:
            st.error(f"Hole19 라운드 가져오기 실패: {exc}")

    imported = st.session_state.get("hole19_imported_round")
    if not imported:
        return

    st.markdown("##### 가져온 라운드 확인")
    info = st.columns(4)
    info[0].metric("골프장", imported.course_name)
    info[1].metric("Par", imported.course_par if imported.course_par is not None else "-")
    info[2].metric("핸디캡", imported.playing_hcp if imported.playing_hcp is not None else "-")
    info[3].metric("홀", imported.holes_number)

    holes_df = _holes_from_imported(imported)
    edited = st.data_editor(
        holes_df,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        column_config={
            "홀": st.column_config.NumberColumn("홀", disabled=True),
            "Par": st.column_config.NumberColumn("Par", disabled=True),
            "Score": st.column_config.NumberColumn("Score", min_value=1, max_value=20, step=1),
            "Putts": st.column_config.NumberColumn("Putts", min_value=0, max_value=10, step=1),
            "FIR": st.column_config.TextColumn("FIR"),
            "GIR": st.column_config.CheckboxColumn("GIR"),
            "Sand": st.column_config.NumberColumn("Sand", min_value=0, max_value=10, step=1),
            "Penalty": st.column_config.NumberColumn("Penalty", min_value=0, max_value=10, step=1),
        },
        key="hole19_import_holes",
    )
    holes = _normalize_holes_editor(edited)
    _render_summary(holes)

    note_col, goal_col = st.columns(2)
    with note_col:
        notes = st.text_area("라운드 메모", key="hole19_notes")
        best_shot = st.text_input("베스트 샷", key="hole19_best_shot")
        weakness = st.text_input("아쉬운 점", key="hole19_weakness")
    with goal_col:
        next_goal = st.text_area("다음 라운드 목표", key="hole19_next_goal")

    if st.button("💾 DODOS에 저장", type="primary", width="stretch", key="hole19_save"):
        try:
            round_date = (imported.played_at or "")[:10] or date.today().isoformat()
            _save_round(
                user_email=user_email,
                round_date=round_date,
                course_name=imported.course_name,
                tee_name="",
                course_par=imported.course_par,
                scoring_mode=imported.scoring_mode,
                playing_hcp=imported.playing_hcp,
                holes=holes,
                source="hole19",
                source_round_id=imported.source_round_id,
                source_url=imported.source_url,
                best_shot=best_shot,
                weakness=weakness,
                next_goal=next_goal,
                notes=notes,
            )
            st.success("Hole19 라운드를 DODOS에 저장했습니다.")
        except Exception as exc:
            st.error(f"Hole19 라운드 저장 실패: {exc}")
            st.caption("Supabase에 Phase 4 라운드 테이블을 먼저 생성했는지 확인해 주세요.")


def render_round_record(*, user_email: str) -> None:
    """Phase 4 Round Record: manual entry + Hole19 import."""
    st.markdown("## ⛳ 라운드 기록")
    st.caption("직접 입력하거나 Hole19 라운드 URL에서 데이터를 가져올 수 있습니다.")

    mode = st.radio(
        "기록 방식",
        ["✍️ 직접 입력", "📥 Hole19 가져오기"],
        horizontal=True,
        label_visibility="collapsed",
        key="round_record_mode",
    )

    if mode == "✍️ 직접 입력":
        _render_manual(user_email)
    else:
        _render_import(user_email)
