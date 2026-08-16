from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from supabase import Client

from dodos_supabase import DodosSupabaseConfig, create_dodos_client, get_dodos_user


@dataclass(frozen=True)
class JournalListItem:
    session_id: str
    practice_date: str
    shot_count: int
    club_count: int
    ai_score: float | None
    ai_grade: str | None
    best_club: str | None
    focus_club: str | None
    has_note: bool


@dataclass(frozen=True)
class JournalDetail:
    session_id: str
    practice_date: str
    shot_count: int
    club_count: int
    ai_score: float | None
    ai_grade: str | None
    best_club: str | None
    best_club_score: float | None
    focus_club: str | None
    focus_club_score: float | None
    category_scores: dict[str, float]
    ai_strengths: list[str]
    ai_improvements: list[str]
    ai_tasks: list[str]
    coaching_summary: str
    condition_score: int | None
    satisfaction_score: int | None
    practice_goal: str
    memo: str
    lesson_note: str


def _first(rows):
    return rows[0] if rows else None


def _f(v):
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def _i(v):
    try:
        return None if v is None else int(v)
    except Exception:
        return None


class PracticeJournalService:
    def __init__(self, *, client: Client | None = None,
                 config: DodosSupabaseConfig | None = None,
                 user_email: str | None = None) -> None:
        self.config = config or DodosSupabaseConfig.load()
        self.client = client or create_dodos_client(self.config)
        email = (user_email or self.config.user_email).lower().strip()
        if not email:
            raise RuntimeError("DODOS_USER_EMAIL 또는 user_email이 필요합니다.")
        self.user = get_dodos_user(self.client, email=email)
        self.user_id = str(self.user["id"])

    def list_sessions(self, *, limit: int = 60) -> list[JournalListItem]:
        sessions = (
            self.client.table("dodos_practice_sessions")
            .select("id,practice_date,shot_count,club_count,current_ai_score,current_ai_grade")
            .eq("user_id", self.user_id)
            .eq("status", "complete")
            .order("practice_date", desc=True)
            .limit(limit)
            .execute()
        )
        rows = list(sessions.data or [])
        if not rows:
            return []

        ids = [str(r["id"]) for r in rows]

        snapshots = (
            self.client.table("dodos_daily_snapshots")
            .select("session_id,best_club,focus_club,best_club_score,focus_club_score")
            .in_("session_id", ids)
            .execute()
        )
        smap = {str(r["session_id"]): r for r in (snapshots.data or [])}

        notes = (
            self.client.table("dodos_practice_notes")
            .select("session_id")
            .in_("session_id", ids)
            .execute()
        )
        note_ids = {str(r["session_id"]) for r in (notes.data or [])}

        out = []
        for r in rows:
            sid = str(r["id"])
            s = smap.get(sid, {})
            out.append(JournalListItem(
                session_id=sid,
                practice_date=str(r["practice_date"]),
                shot_count=int(r.get("shot_count") or 0),
                club_count=int(r.get("club_count") or 0),
                ai_score=_f(r.get("current_ai_score")),
                ai_grade=r.get("current_ai_grade"),
                best_club=s.get("best_club"),
                focus_club=s.get("focus_club"),
                has_note=sid in note_ids,
            ))
        return out

    def get_session_by_date(self, practice_date: str):
        r = (
            self.client.table("dodos_practice_sessions")
            .select("*")
            .eq("user_id", self.user_id)
            .eq("practice_date", practice_date)
            .eq("status", "complete")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return _first(r.data)

    def get_journal_detail(self, practice_date: str) -> JournalDetail | None:
        session = self.get_session_by_date(practice_date)
        if not session:
            return None
        sid = str(session["id"])

        snap = _first((
            self.client.table("dodos_daily_snapshots")
            .select("*").eq("session_id", sid).limit(1).execute()
        ).data) or {}

        ai = _first((
            self.client.table("dodos_ai_session_reports")
            .select("overall_score,grade,category_scores,strengths,improvements,tasks,coaching_summary,best_club,best_club_score,focus_club,focus_club_score")
            .eq("session_id", sid).order("created_at", desc=True).limit(1).execute()
        ).data) or {}

        note = _first((
            self.client.table("dodos_practice_notes")
            .select("*").eq("session_id", sid).limit(1).execute()
        ).data) or {}

        lesson = _first((
            self.client.table("dodos_lesson_notes")
            .select("*").eq("session_id", sid).order("created_at", desc=True).limit(1).execute()
        ).data) or {}

        return JournalDetail(
            session_id=sid,
            practice_date=str(session["practice_date"]),
            shot_count=int(session.get("shot_count") or 0),
            club_count=int(session.get("club_count") or 0),
            ai_score=_f(ai.get("overall_score") if ai else session.get("current_ai_score")),
            ai_grade=ai.get("grade") if ai else session.get("current_ai_grade"),
            best_club=ai.get("best_club") or snap.get("best_club"),
            best_club_score=_f(ai.get("best_club_score") or snap.get("best_club_score")),
            focus_club=ai.get("focus_club") or snap.get("focus_club"),
            focus_club_score=_f(ai.get("focus_club_score") or snap.get("focus_club_score")),
            category_scores=dict(ai.get("category_scores") or {}),
            ai_strengths=list(ai.get("strengths") or []),
            ai_improvements=list(ai.get("improvements") or []),
            ai_tasks=list(ai.get("tasks") or []),
            coaching_summary=str(ai.get("coaching_summary") or ""),
            condition_score=_i(note.get("condition_score")),
            satisfaction_score=_i(note.get("satisfaction_score")),
            practice_goal=str(note.get("practice_goal") or ""),
            memo=str(note.get("memo") or ""),
            lesson_note=str(lesson.get("note") or ""),
        )


    def get_journal_detail_by_session_id(
        self,
        session_id: str,
    ) -> JournalDetail | None:
        """같은 날짜의 여러 연습도 session_id로 정확히 조회합니다."""
        session = _first(
            (
                self.client.table("dodos_practice_sessions")
                .select("*")
                .eq("id", session_id)
                .eq("user_id", self.user_id)
                .limit(1)
                .execute()
            ).data
        )
        if not session:
            return None

        sid = str(session["id"])

        snapshot = _first(
            (
                self.client.table("dodos_daily_snapshots")
                .select("*")
                .eq("session_id", sid)
                .limit(1)
                .execute()
            ).data
        ) or {}

        ai_report = _first(
            (
                self.client.table("dodos_ai_session_reports")
                .select(
                    "overall_score,grade,category_scores,strengths,"
                    "improvements,tasks,coaching_summary,"
                    "best_club,best_club_score,focus_club,focus_club_score"
                )
                .eq("session_id", sid)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            ).data
        ) or {}

        note = _first(
            (
                self.client.table("dodos_practice_notes")
                .select("*")
                .eq("session_id", sid)
                .limit(1)
                .execute()
            ).data
        ) or {}

        lesson = _first(
            (
                self.client.table("dodos_lesson_notes")
                .select("*")
                .eq("session_id", sid)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            ).data
        ) or {}

        return JournalDetail(
            session_id=sid,
            practice_date=str(session["practice_date"]),
            shot_count=int(session.get("shot_count") or 0),
            club_count=int(session.get("club_count") or 0),
            ai_score=_f(
                ai_report.get("overall_score")
                if ai_report
                else session.get("current_ai_score")
            ),
            ai_grade=(
                ai_report.get("grade")
                if ai_report
                else session.get("current_ai_grade")
            ),
            best_club=ai_report.get("best_club") or snapshot.get("best_club"),
            best_club_score=_f(
                ai_report.get("best_club_score")
                or snapshot.get("best_club_score")
            ),
            focus_club=ai_report.get("focus_club") or snapshot.get("focus_club"),
            focus_club_score=_f(
                ai_report.get("focus_club_score")
                or snapshot.get("focus_club_score")
            ),
            category_scores=dict(ai_report.get("category_scores") or {}),
            ai_strengths=list(ai_report.get("strengths") or []),
            ai_improvements=list(ai_report.get("improvements") or []),
            ai_tasks=list(ai_report.get("tasks") or []),
            coaching_summary=str(ai_report.get("coaching_summary") or ""),
            condition_score=_i(note.get("condition_score")),
            satisfaction_score=_i(note.get("satisfaction_score")),
            practice_goal=str(note.get("practice_goal") or ""),
            memo=str(note.get("memo") or ""),
            lesson_note=str(lesson.get("note") or ""),
        )

    def save_practice_note(self, *, session_id: str,
                           condition_score: int | None,
                           satisfaction_score: int | None,
                           practice_goal: str,
                           memo: str,
                           tags: list[str] | None = None):
        payload = {
            "session_id": session_id,
            "user_id": self.user_id,
            "condition_score": condition_score,
            "satisfaction_score": satisfaction_score,
            "practice_goal": practice_goal.strip(),
            "memo": memo.strip(),
            "tags": tags or [],
        }
        r = (
            self.client.table("dodos_practice_notes")
            .upsert(payload, on_conflict="session_id")
            .execute()
        )
        if not r.data:
            raise RuntimeError("연습 일지 저장 결과가 없습니다.")
        return r.data[0]

    def save_lesson_note(self, *, session_id: str, practice_date: str,
                         note: str, title: str = "연습 스윙 노트"):
        existing = _first((
            self.client.table("dodos_lesson_notes")
            .select("id")
            .eq("session_id", session_id)
            .eq("user_id", self.user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        ).data)

        if not note.strip():
            if existing:
                (
                    self.client.table("dodos_lesson_notes")
                    .delete()
                    .eq("id", existing["id"])
                    .eq("user_id", self.user_id)
                    .execute()
                )
            return None

        payload = {
            "user_id": self.user_id,
            "session_id": session_id,
            "lesson_date": practice_date,
            "title": title,
            "note": note.strip(),
            "focus_clubs": [],
            "focus_points": [],
            "next_tasks": [],
        }

        if existing:
            r = (
                self.client.table("dodos_lesson_notes")
                .update(payload)
                .eq("id", existing["id"])
                .execute()
            )
        else:
            r = self.client.table("dodos_lesson_notes").insert(payload).execute()

        if not r.data:
            raise RuntimeError("레슨/스윙 노트 저장 결과가 없습니다.")
        return r.data[0]

    def save_journal(self, *, session_id: str, practice_date: str,
                     condition_score: int | None,
                     satisfaction_score: int | None,
                     practice_goal: str, memo: str,
                     lesson_note: str):
        self.save_practice_note(
            session_id=session_id,
            condition_score=condition_score,
            satisfaction_score=satisfaction_score,
            practice_goal=practice_goal,
            memo=memo,
        )
        self.save_lesson_note(
            session_id=session_id,
            practice_date=practice_date,
            note=lesson_note,
        )


if __name__ == "__main__":
    svc = PracticeJournalService()
    for item in svc.list_sessions(limit=5):
        print(item)
