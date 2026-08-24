from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from supabase import Client

from dodos_supabase import DodosSupabaseConfig, create_dodos_client, get_dodos_user


@dataclass(frozen=True)
class RoundRecord:
    id: str
    round_date: str
    course_name: str
    tee_name: str | None
    total_score: int | None
    total_putts: int | None
    fir_pct: float | None
    gir_pct: float | None
    penalties: int | None
    source: str
    source_round_id: str | None


def _first(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _i(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _f(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: Any) -> float | None:
    value = _f(value)
    return None if value is None else round(value, 1)


def _is_fir_hit(value: Any) -> bool:
    return str(value or "").strip().lower() in {"hit", "yes", "true", "1"}


def _is_gir_hit(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"hit", "yes", "true", "1"}


class RoundService:
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

    def list_rounds(self, *, limit: int = 50) -> list[RoundRecord]:
        result = (
            self.client.table("dodos_rounds")
            .select("id,round_date,course_name,tee_name,total_score,total_putts,fir_pct,gir_pct,penalties,source,source_round_id")
            .eq("user_id", self.user_id)
            .order("round_date", desc=True)
            .limit(limit)
            .execute()
        )
        return [
            RoundRecord(
                id=str(row["id"]),
                round_date=str(row.get("round_date") or ""),
                course_name=str(row.get("course_name") or ""),
                tee_name=row.get("tee_name"),
                total_score=_i(row.get("total_score")),
                total_putts=_i(row.get("total_putts")),
                fir_pct=_pct(row.get("fir_pct")),
                gir_pct=_pct(row.get("gir_pct")),
                penalties=_i(row.get("penalties")),
                source=str(row.get("source") or "manual"),
                source_round_id=row.get("source_round_id"),
            )
            for row in (result.data or [])
        ]

    def get_round(self, round_id: str) -> dict[str, Any] | None:
        result = (
            self.client.table("dodos_rounds")
            .select("*")
            .eq("id", round_id)
            .eq("user_id", self.user_id)
            .limit(1)
            .execute()
        )
        row = _first(result.data or [])
        if not row:
            return None
        holes = (
            self.client.table("dodos_round_holes")
            .select("*")
            .eq("round_id", round_id)
            .eq("user_id", self.user_id)
            .order("hole_number")
            .execute()
        )
        row["holes"] = list(holes.data or [])
        return row

    def save_round(
        self,
        *,
        round_date: str,
        course_name: str,
        tee_name: str | None,
        course_par: int | None,
        scoring_mode: str | None,
        playing_hcp: int | None,
        holes: list[dict[str, Any]],
        source: str = "manual",
        source_round_id: str | None = None,
        source_url: str | None = None,
        best_shot: str = "",
        weakness: str = "",
        next_goal: str = "",
        notes: str = "",
        duration_minutes: int | None = None,
        distance_km: float | None = None,
    ) -> dict[str, Any]:
        clean_holes = [h for h in holes if h.get("hole_number")]
        scores = [h.get("score") for h in clean_holes if h.get("score") is not None]
        putts = [h.get("putts") for h in clean_holes if h.get("putts") is not None]
        penalties = [h.get("penalties") for h in clean_holes if h.get("penalties") is not None]
        fir_values = [h.get("fir") for h in clean_holes if h.get("par", 0) > 3 and h.get("fir") not in (None, "")]
        gir_values = [h.get("gir") for h in clean_holes if h.get("par") and h.get("gir") is not None]

        fir_hits = sum(1 for value in fir_values if _is_fir_hit(value))
        gir_hits = sum(1 for value in gir_values if _is_gir_hit(value))
        total_fir = len(fir_values)
        total_gir = len(gir_values)

        payload = {
            "user_id": self.user_id,
            "round_date": round_date,
            "course_name": course_name.strip(),
            "tee_name": (tee_name or "").strip() or None,
            "course_par": course_par,
            "scoring_mode": scoring_mode,
            "playing_hcp": playing_hcp,
            "total_score": sum(scores) if scores else None,
            "total_putts": sum(putts) if putts else None,
            "fir_pct": round(fir_hits / total_fir * 100, 1) if total_fir else None,
            "gir_pct": round(gir_hits / total_gir * 100, 1) if total_gir else None,
            "penalties": sum(penalties) if penalties else 0,
            "source": source,
            "source_round_id": source_round_id,
            "source_url": source_url,
            "best_shot": best_shot.strip(),
            "weakness": weakness.strip(),
            "next_goal": next_goal.strip(),
            "notes": notes.strip(),
            "duration_minutes": duration_minutes,
            "distance_km": distance_km,
        }

        existing = None
        if source_round_id:
            existing = _first(
                self.client.table("dodos_rounds")
                .select("id")
                .eq("user_id", self.user_id)
                .eq("source", source)
                .eq("source_round_id", source_round_id)
                .limit(1)
                .execute().data or []
            )

        if existing:
            round_id = str(existing["id"])
            result = (
                self.client.table("dodos_rounds")
                .update(payload)
                .eq("id", round_id)
                .eq("user_id", self.user_id)
                .execute()
            )
        else:
            result = self.client.table("dodos_rounds").insert(payload).execute()

        row = _first(result.data or [])
        if not row:
            raise RuntimeError("라운드 기본 정보 저장 결과가 없습니다.")
        round_id = str(row["id"])

        self.client.table("dodos_round_holes").delete().eq("round_id", round_id).eq("user_id", self.user_id).execute()

        hole_rows = []
        for hole in clean_holes:
            hole_number = hole.get("hole_number")
            hole_rows.append({
                "round_id": round_id,
                "user_id": self.user_id,
                # Keep both names during migration because an older DB schema
                # may still have NOT NULL `hole_no` while the new model uses
                # `hole_number`.
                "hole_no": hole_number,
                "hole_number": hole_number,
                "par": hole.get("par"),
                "stroke_index": hole.get("stroke_index"),
                "distance_m": hole.get("distance_m"),
                "score": hole.get("score"),
                "putts": hole.get("putts"),
                "fir": hole.get("fir"),
                "gir": hole.get("gir"),
                "sand_shots": hole.get("sand_shots"),
                "penalties": hole.get("penalties"),
                "extra_strokes": hole.get("extra_strokes"),
                "net_score": hole.get("net_score"),
                "stableford_points": hole.get("stableford_points"),
            })
        if hole_rows:
            self.client.table("dodos_round_holes").insert(hole_rows).execute()

        return self.get_round(round_id) or row
