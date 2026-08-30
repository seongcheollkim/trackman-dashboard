from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from services.hole19_service import Hole19Round, fetch_hole19_round
from services.round_service import RoundService, RoundRecord


ROUND_CSS = """
<style>
:root {
  --rr-bg: #07101a;
  --rr-panel: #101b27;
  --rr-panel2: #0d1722;
  --rr-line: #263548;
  --rr-text: #eef4fb;
  --rr-muted: #9aaabd;
  --rr-accent: #ff6b1a;
  --rr-blue: #3d94ff;
}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at 40% 0%, #102033 0%, #07101a 42%, #050a10 100%);
}
[data-testid="stSidebar"] { background:#0b1520; border-right:1px solid #1e2c3c; }
.block-container {
  padding-top:1.5rem !important;
  padding-bottom:1.5rem !important;
  padding-left:3rem !important;
  padding-right:3rem !important;
  max-width:1800px !important;
}
header[data-testid="stHeader"] { background:transparent !important; }
[data-testid="stDecoration"] { display:none !important; }
#MainMenu, footer { visibility:hidden; }
hr { border-color:#223044; }

.rr-title { font-size:1.55rem; font-weight:800; color:#f7fbff; margin:6px 0 4px; }
.rr-subtitle { color:#9aaabd; font-size:.9rem; margin-bottom:18px; }
.rr-section {
  border:1px solid #263548;
  border-radius:12px;
  padding:16px 18px;
  background:linear-gradient(180deg,#111e2b,#0d1721);
  margin-bottom:14px;
}
.rr-kpi {
  border:1px solid #263548;
  border-radius:10px;
  padding:13px 15px;
  background:linear-gradient(180deg,#121f2d,#0d1721);
  min-height:84px;
}
.rr-kpi-label { color:#9aaabd; font-size:.82rem; margin-bottom:7px; }
.rr-kpi-value { color:#f3f8ff; font-size:1.45rem; font-weight:800; }
.rr-muted { color:#9aaabd; font-size:.86rem; }
.rr-source {
  display:inline-block; padding:4px 9px; border-radius:999px;
  border:1px solid #314256; background:#111c29; color:#dce8f5; font-size:.78rem;
}
</style>
"""


def _round_service(user_email: str) -> RoundService:
    return RoundService(user_email=user_email)


def _is_fir_hit(value: Any) -> bool:
    return str(value or "").strip().lower() in {"target", "hit", "yes", "true", "1"}


def _is_gir_hit(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"hit", "yes", "true", "1"}


def _empty_holes() -> pd.DataFrame:
    return pd.DataFrame([
        {"홀": i, "Par": 4, "Score": None, "Putts": None, "FIR": None, "GIR": None, "Sand": 0, "Penalty": 0}
        for i in range(1, 19)
    ])


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
        rows.append({
            "홀": hole.get("hole_number"),
            "Par": hole.get("par"),
            "Score": hole.get("score"),
            "Putts": hole.get("putts"),
            "FIR": fir_label,
            "GIR": hole.get("gir"),
            "Sand": hole.get("sand_shots") or 0,
            "Penalty": hole.get("penalties") or 0,
        })
    return pd.DataFrame(rows)


def _normalize_holes_editor(df: pd.DataFrame) -> list[dict[str, Any]]:
    holes: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        hole_number = row.get("홀")
        if pd.isna(hole_number):
            continue

        def clean(value: Any) -> Any:
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
            return value

        score = clean(row.get("Score"))
        putts = clean(row.get("Putts"))
        holes.append({
            "hole_number": int(hole_number),
            "par": int(clean(row.get("Par")) or 0),
            "score": int(score) if score is not None else None,
            "putts": int(putts) if putts is not None else None,
            "fir": clean(row.get("FIR")),
            "gir": clean(row.get("GIR")),
            "sand_shots": int(clean(row.get("Sand")) or 0),
            "penalties": int(clean(row.get("Penalty")) or 0),
        })
    return holes


def _round_metrics(holes: list[dict[str, Any]]) -> tuple[int | None, int | None, float | None, float | None, int]:
    scores = [h["score"] for h in holes if h.get("score") is not None]
    putts = [h["putts"] for h in holes if h.get("putts") is not None]
    penalty = sum(int(h.get("penalties") or 0) for h in holes)
    fir_rows = [h for h in holes if int(h.get("par") or 0) > 3 and h.get("fir") not in (None, "")]
    fir_hits = sum(1 for h in fir_rows if _is_fir_hit(h.get("fir")))
    gir_rows = [h for h in holes if h.get("gir") is not None]
    gir_hits = sum(1 for h in gir_rows if _is_gir_hit(h.get("gir")))
    fir_pct = round(fir_hits / len(fir_rows) * 100, 1) if fir_rows else None
    gir_pct = round(gir_hits / len(gir_rows) * 100, 1) if gir_rows else None
    return sum(scores) if scores else None, sum(putts) if putts else None, fir_pct, gir_pct, penalty


def _save_round(*, user_email: str, round_date: str, course_name: str, tee_name: str, course_par: int | None, scoring_mode: str | None, playing_hcp: int | None, holes: list[dict[str, Any]], source: str, source_round_id: str | None, source_url: str | None, best_shot: str, weakness: str, next_goal: str, notes: str) -> None:
    _round_service(user_email).save_round(
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
            st.markdown(f'<div class="rr-kpi"><div class="rr-kpi-label">{label}</div><div class="rr-kpi-value">{value}</div></div>', unsafe_allow_html=True)


def _source_label(source: str) -> str:
    return "Hole19" if source.lower() == "hole19" else "직접 입력"


def _format_date(value: str) -> str:
    return value[:10] if value else "-"


def _render_round_list(user_email: str) -> None:
    service = _round_service(user_email)
    st.markdown('<div class="rr-title">📚 저장된 라운드</div>', unsafe_allow_html=True)
    st.markdown('<div class="rr-subtitle">DODOS에 저장된 라운드 기록을 날짜순으로 확인합니다.</div>', unsafe_allow_html=True)

    top = st.columns([1, 5])
    with top[0]:
        refresh = st.button("↻ 새로고침", use_container_width=True)
    if refresh:
        st.rerun()

    try:
        rounds = service.list_rounds(limit=100)
    except Exception as exc:
        st.error(f"라운드 기록을 불러오지 못했습니다: {exc}")
        return

    if not rounds:
        st.info("저장된 라운드가 없습니다. ‘새 라운드 기록’ 탭에서 라운드를 추가해 주세요.")
        return

    rows = []
    for r in rounds:
        rows.append({
            "날짜": _format_date(r.round_date),
            "골프장": r.course_name,
            "티": r.tee_name or "-",
            "Score": "-" if r.total_score is None else r.total_score,
            "Putts": "-" if r.total_putts is None else r.total_putts,
            "FIR": "-" if r.fir_pct is None else f"{r.fir_pct:.1f}%",
            "GIR": "-" if r.gir_pct is None else f"{r.gir_pct:.1f}%",
            "Penalty": "-" if r.penalties is None else r.penalties,
            "기록 방식": _source_label(r.source),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    labels = []
    by_label: dict[str, RoundRecord] = {}
    for r in rounds:
        label = f"{_format_date(r.round_date)} · {r.course_name} · {r.total_score if r.total_score is not None else '-'}타 · {_source_label(r.source)}"
        labels.append(label)
        by_label[label] = r

    selected_label = st.selectbox("상세히 볼 라운드", labels, key="round_history_selected")
    selected = by_label[selected_label]

    try:
        detail = service.get_round(selected.id)
    except Exception as exc:
        st.error(f"라운드 상세 정보를 불러오지 못했습니다: {exc}")
        return
    if not detail:
        st.warning("선택한 라운드의 상세 정보가 없습니다.")
        return

    st.markdown("### 라운드 상세")
    info = st.columns([2.4, 1, 1, 1, 1])
    with info[0]:
        st.markdown(f"**{detail.get('course_name') or '-'}**")
        st.caption(f"{_format_date(str(detail.get('round_date') or ''))} · {_source_label(str(detail.get('source') or 'manual'))}")
    info[1].metric("Score", detail.get("total_score") if detail.get("total_score") is not None else "-")
    info[2].metric("Putts", detail.get("total_putts") if detail.get("total_putts") is not None else "-")
    info[3].metric("FIR", f"{float(detail['fir_pct']):.1f}%" if detail.get("fir_pct") is not None else "-")
    info[4].metric("GIR", f"{float(detail['gir_pct']):.1f}%" if detail.get("gir_pct") is not None else "-")

    holes = detail.get("holes") or []
    if holes:
        hole_rows = []
        for h in holes:
            hole_rows.append({
                "홀": h.get("hole_no", h.get("hole_number")),
                "Par": h.get("par"),
                "Score": h.get("score"),
                "Putts": h.get("putts"),
                "FIR": h.get("fir") if h.get("fir") not in (None, "") else "-",
                "GIR": "○" if _is_gir_hit(h.get("gir")) else "-",
                "Sand": h.get("sand_shots") or 0,
                "Penalty": h.get("penalties") or 0,
            })
        st.dataframe(pd.DataFrame(hole_rows), hide_index=True, width="stretch")

    note_cols = st.columns(3)
    for col, title, key in [
        (note_cols[0], "베스트 샷", "best_shot"),
        (note_cols[1], "아쉬운 점", "weakness"),
        (note_cols[2], "다음 라운드 목표", "next_goal"),
    ]:
        with col:
            st.markdown(f"**{title}**")
            st.markdown(detail.get(key) or "-" )
    st.markdown("**라운드 메모**")
    st.markdown(detail.get("notes") or "-")


def _render_manual(user_email: str) -> None:
    st.markdown("#### 직접 입력")
    st.caption("필드에서 기록한 내용을 18홀 단위로 직접 입력합니다.")
    with st.form("round_manual_form"):
        top = st.columns([1.0, 1.8, 1.0, 1.0])
        with top[0]: round_date = st.date_input("라운드 날짜", value=date.today())
        with top[1]: course_name = st.text_input("골프장", placeholder="예: 솔라고 CC")
        with top[2]: tee_name = st.text_input("티박스", placeholder="White")
        with top[3]: playing_hcp = st.number_input("핸디캡", min_value=0, max_value=54, value=0, step=1)
        st.markdown("##### 홀별 기록")
        edited = st.data_editor(
            _empty_holes(), hide_index=True, width="stretch", num_rows="fixed",
            column_config={
                "홀": st.column_config.NumberColumn("홀", disabled=True, width="small"),
                "Par": st.column_config.NumberColumn("Par", min_value=1, max_value=8, step=1),
                "Score": st.column_config.NumberColumn("Score", min_value=1, max_value=20, step=1),
                "Putts": st.column_config.NumberColumn("Putts", min_value=0, max_value=10, step=1),
                "FIR": st.column_config.SelectboxColumn("FIR", options=["target", "left", "right", "other", "miss"], required=False),
                "GIR": st.column_config.CheckboxColumn("GIR"),
                "Sand": st.column_config.NumberColumn("Sand", min_value=0, max_value=10, step=1),
                "Penalty": st.column_config.NumberColumn("Penalty", min_value=0, max_value=10, step=1),
            }, key="round_manual_holes",
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
                user_email=user_email, round_date=round_date.isoformat(), course_name=course_name,
                tee_name=tee_name, course_par=sum(int(h.get("par") or 0) for h in holes) or None,
                scoring_mode="manual", playing_hcp=int(playing_hcp) if playing_hcp else None,
                holes=holes, source="manual", source_round_id=None, source_url=None,
                best_shot=best_shot, weakness=weakness, next_goal=next_goal, notes=notes,
            )
            st.success("라운드를 저장했습니다. ‘저장된 라운드’ 탭에서 확인할 수 있습니다.")
        except Exception as exc:
            st.error(f"라운드 저장 실패: {exc}")
            st.caption("Supabase에 Phase 4 라운드 테이블이 생성되어 있는지 확인해 주세요.")


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
            st.success(f"가져오기 성공 · {imported.course_name} · {len(imported.holes)}홀")
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
    edited = st.data_editor(
        _holes_from_imported(imported), hide_index=True, width="stretch", num_rows="fixed",
        column_config={
            "홀": st.column_config.NumberColumn("홀", disabled=True),
            "Par": st.column_config.NumberColumn("Par", disabled=True),
            "Score": st.column_config.NumberColumn("Score", min_value=1, max_value=20, step=1),
            "Putts": st.column_config.NumberColumn("Putts", min_value=0, max_value=10, step=1),
            "FIR": st.column_config.TextColumn("FIR"),
            "GIR": st.column_config.CheckboxColumn("GIR"),
            "Sand": st.column_config.NumberColumn("Sand", min_value=0, max_value=10, step=1),
            "Penalty": st.column_config.NumberColumn("Penalty", min_value=0, max_value=10, step=1),
        }, key="hole19_import_holes",
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
                user_email=user_email, round_date=round_date, course_name=imported.course_name,
                tee_name="", course_par=imported.course_par, scoring_mode=imported.scoring_mode,
                playing_hcp=imported.playing_hcp, holes=holes, source="hole19",
                source_round_id=imported.source_round_id, source_url=imported.source_url,
                best_shot=best_shot, weakness=weakness, next_goal=next_goal, notes=notes,
            )
            st.success("Hole19 라운드를 DODOS에 저장했습니다. ‘저장된 라운드’ 탭에서 확인할 수 있습니다.")
        except Exception as exc:
            st.error(f"Hole19 라운드 저장 실패: {exc}")
            st.caption("Supabase에 Phase 4 라운드 테이블이 생성되어 있는지 확인해 주세요.")


def _render_new_round(user_email: str) -> None:
    st.markdown('<div class="rr-title">➕ 새 라운드 기록</div>', unsafe_allow_html=True)
    st.markdown('<div class="rr-subtitle">직접 입력하거나 Hole19에서 라운드 데이터를 가져옵니다.</div>', unsafe_allow_html=True)
    manual_tab, import_tab = st.tabs(["✍️ 직접 입력", "📥 Hole19 가져오기"])
    with manual_tab:
        _render_manual(user_email)
    with import_tab:
        _render_import(user_email)


def render_round_record(*, user_email: str) -> None:
    st.markdown(ROUND_CSS, unsafe_allow_html=True)
    st.markdown('<div class="rr-title">⛳ 라운드 기록</div>', unsafe_allow_html=True)
    st.markdown('<div class="rr-subtitle">라운드 기록을 저장하고, 과거 라운드를 다시 조회할 수 있습니다.</div>', unsafe_allow_html=True)
    history_tab, new_round_tab = st.tabs(["📚 저장된 라운드", "➕ 새 라운드 기록"])
    with history_tab:
        _render_round_list(user_email)
    with new_round_tab:
        _render_new_round(user_email)
