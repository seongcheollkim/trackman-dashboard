from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, is_dataclass

import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from supabase import Client

from dodos_supabase import (
    DodosSupabaseConfig,
    create_dodos_client,
    get_dodos_user,
)
from trackman_core import make_summary, parse_trackman_report


@dataclass(frozen=True)
class ArchiveResult:
    file_name: str
    practice_date: str
    expected_shots: int
    archived_shots: int
    verified: bool
    skipped: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.verified and not self.error


def _f(value: Any) -> float | None:
    try:
        if value is None:
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _i(value: Any) -> int | None:
    try:
        if value is None:
            return None
        x = float(value)
        return int(round(x)) if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _practice_date(payload: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    for row in rows:
        value = row.get("Date")
        if value:
            return str(value)[:10]

    for group in payload.get("StrokeGroups") or []:
        value = group.get("Date")
        if value:
            return str(value)[:10]

    raise ValueError("TrackMan JSON에서 연습 날짜를 찾지 못했습니다.")


def _shot_key(row: dict[str, Any], index: int) -> str:
    stroke_id = row.get("StrokeId")
    if stroke_id:
        return f"id:{stroke_id}"

    return "fallback:" + "|".join([
        str(row.get("GroupId") or ""),
        str(row.get("Club") or ""),
        str(row.get("StrokeNo") or index + 1),
        str(row.get("StrokeTime") or ""),
    ])


def _shot_db_row(
    row: dict[str, Any],
    *,
    session_id: str,
    user_id: str,
    practice_date: str,
    index: int,
) -> dict[str, Any]:
    """
    trackman_core.parse_trackman_report()의 출력 컬럼을
    dodos_shots 컬럼으로 1:1 매핑합니다.
    """
    return {
        "session_id": session_id,
        "user_id": user_id,
        "practice_date": practice_date,
        "shot_key": _shot_key(row, index),

        "group_id": row.get("GroupId"),
        "group_club": row.get("GroupClub"),
        "stroke_no": _i(row.get("StrokeNo")),
        "stroke_id": row.get("StrokeId"),
        "stroke_time": row.get("StrokeTime"),

        "club": str(row.get("Club") or "Unknown"),
        "ball": row.get("Ball"),
        "measurement_kind": row.get("MeasurementKind"),

        "club_speed_mps": _f(row.get("ClubSpeed_mps")),
        "ball_speed_mps": _f(row.get("BallSpeed_mps")),
        "smash_factor": _f(row.get("SmashFactor")),

        "carry_m": _f(row.get("Carry_m")),
        "total_m": _f(row.get("Total_m")),
        "run_m": _f(row.get("Run_m")),
        "carry_side_m": _f(row.get("CarrySide_m")),
        "total_side_m": _f(row.get("TotalSide_m")),
        "abs_total_side_m": _f(row.get("AbsTotalSide_m")),

        "attack_angle_deg": _f(row.get("AttackAngle_deg")),
        "club_path_deg": _f(row.get("ClubPath_deg")),
        "face_angle_deg": _f(row.get("FaceAngle_deg")),
        "face_to_path_deg": _f(row.get("FaceToPath_deg")),

        "launch_angle_deg": _f(row.get("LaunchAngle_deg")),
        "launch_direction_deg": _f(row.get("LaunchDirection_deg")),
        "spin_rate_rpm": _i(row.get("SpinRate_rpm")),
        "spin_axis_deg": _f(row.get("SpinAxis_deg")),

        "max_height_m": _f(row.get("MaxHeight_m")),
        "landing_angle_deg": _f(row.get("LandingAngle_deg")),
        "hang_time_s": _f(row.get("HangTime_s")),

        "dynamic_loft_deg": _f(row.get("DynamicLoft_deg")),
        "spin_loft_deg": _f(row.get("SpinLoft_deg")),
        "swing_plane_deg": _f(row.get("SwingPlane_deg")),
        "swing_direction_deg": _f(row.get("SwingDirection_deg")),

        "low_point_distance_m": _f(row.get("LowPointDistance_m")),
        "low_point_height_m": _f(row.get("LowPointHeight_m")),
        "low_point_side_m": _f(row.get("LowPointSide_m")),

        "impact_offset_mm": _f(row.get("ImpactOffset_mm")),
        "impact_height_mm": _f(row.get("ImpactHeight_mm")),
        "dynamic_lie_deg": _f(row.get("DynamicLie_deg")),

        "extra_metrics": {},
    }


def _std(values: Iterable[Any]) -> float | None:
    nums = [_f(v) for v in values]
    nums = [x for x in nums if x is not None]
    return statistics.stdev(nums) if len(nums) > 1 else None


def _summary_db_row(
    summary: dict[str, Any],
    club_rows: list[dict[str, Any]],
    *,
    session_id: str,
    user_id: str,
) -> dict[str, Any]:
    side_values = [
        _f(row.get("TotalSide_m"))
        for row in club_rows
        if _f(row.get("TotalSide_m")) is not None
    ]

    return {
        "session_id": session_id,
        "user_id": user_id,
        "club": summary["Club"],
        "shot_count": int(summary.get("Shots") or len(club_rows)),

        "avg_carry_m": _f(summary.get("Avg_Carry_m")),
        "avg_total_m": _f(summary.get("Avg_Total_m")),
        "avg_ball_speed_mps": _f(summary.get("Avg_BallSpeed_mps")),
        "avg_club_speed_mps": _f(summary.get("Avg_ClubSpeed_mps")),
        "avg_smash_factor": _f(summary.get("Avg_Smash")),
        "avg_spin_rate_rpm": _f(summary.get("Avg_Spin_rpm")),
        "avg_launch_angle_deg": _f(summary.get("Avg_Launch_deg")),
        "avg_attack_angle_deg": _f(summary.get("Avg_Attack_deg")),
        "avg_club_path_deg": _f(summary.get("Avg_Path_deg")),
        "avg_face_angle_deg": _f(summary.get("Avg_Face_deg")),
        "avg_face_to_path_deg": _f(summary.get("Avg_FaceToPath_deg")),
        "avg_total_side_m": _f(summary.get("Avg_TotalSide_m")),

        "carry_std_m": _f(summary.get("Std_Carry_m")),
        "total_side_std_m": _std(side_values),

        # AI 웨지 엔진의 10m bucket dispersion은 AI snapshot 단계에서 별도 저장 가능.
        "dispersion_radius_68_m": None,
        "lateral_abs_mean_m": _f(summary.get("Avg_AbsSide_m")),
        "big_miss_rate_pct": None,

        "summary_metrics": {
            k: v
            for k, v in summary.items()
            if k not in {"Club", "Shots"}
        },
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


DB_TO_AI_COLUMNS = {
    "practice_date": "Date",
    "group_id": "GroupId",
    "group_club": "GroupClub",
    "stroke_no": "StrokeNo",
    "stroke_id": "StrokeId",
    "stroke_time": "StrokeTime",
    "club": "Club",
    "ball": "Ball",
    "measurement_kind": "MeasurementKind",
    "club_speed_mps": "ClubSpeed_mps",
    "ball_speed_mps": "BallSpeed_mps",
    "smash_factor": "SmashFactor",
    "carry_m": "Carry_m",
    "total_m": "Total_m",
    "run_m": "Run_m",
    "carry_side_m": "CarrySide_m",
    "total_side_m": "TotalSide_m",
    "abs_total_side_m": "AbsTotalSide_m",
    "attack_angle_deg": "AttackAngle_deg",
    "club_path_deg": "ClubPath_deg",
    "face_angle_deg": "FaceAngle_deg",
    "face_to_path_deg": "FaceToPath_deg",
    "launch_angle_deg": "LaunchAngle_deg",
    "launch_direction_deg": "LaunchDirection_deg",
    "spin_rate_rpm": "SpinRate_rpm",
    "spin_axis_deg": "SpinAxis_deg",
    "max_height_m": "MaxHeight_m",
    "landing_angle_deg": "LandingAngle_deg",
    "hang_time_s": "HangTime_s",
    "dynamic_loft_deg": "DynamicLoft_deg",
    "spin_loft_deg": "SpinLoft_deg",
    "swing_plane_deg": "SwingPlane_deg",
    "swing_direction_deg": "SwingDirection_deg",
    "low_point_distance_m": "LowPointDistance_m",
    "low_point_height_m": "LowPointHeight_m",
    "low_point_side_m": "LowPointSide_m",
    "impact_offset_mm": "ImpactOffset_mm",
    "impact_height_mm": "ImpactHeight_mm",
    "dynamic_lie_deg": "DynamicLie_deg",
}


class DodosTrackmanArchive:
    """
    기존 TrackMan JSON을 DODOS DB v1.0에 저장하는 공통 엔진.

    사용처:
      - trackman_backfill.py : 과거 전체 JSON
      - nightly automation   : 앞으로 신규 JSON
    """

    def __init__(
        self,
        *,
        client: Client | None = None,
        config: DodosSupabaseConfig | None = None,
        user_email: str | None = None,
    ) -> None:
        self.config = config or DodosSupabaseConfig.load()
        self.client = client or create_dodos_client(self.config)

        email = (user_email or self.config.user_email).lower().strip()
        self.user = get_dodos_user(self.client, email=email)
        self.user_id = str(self.user["id"])
        self.user_email = str(self.user["email"])

    def _find_existing_session(
        self,
        *,
        practice_date: str,
        source_file_name: str,
    ) -> dict[str, Any] | None:
        result = (
            self.client.table("dodos_practice_sessions")
            .select("*")
            .eq("user_id", self.user_id)
            .eq("practice_date", practice_date)
            .eq("source_file_name", source_file_name)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    def _upsert_raw_file(
        self,
        *,
        path: Path,
        practice_date: str,
        expected_shots: int,
    ) -> dict[str, Any]:
        object_name = f"{self.config.cloud_prefix}/{path.name}"
        payload = {
            "user_id": self.user_id,
            "practice_date": practice_date,
            "source_file_name": path.name,
            "storage_bucket": self.config.bucket,
            "storage_object": object_name,
            "file_size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "expected_shot_count": expected_shots,
            "archive_status": "pending",
        }

        result = (
            self.client.table("dodos_raw_files")
            .upsert(payload, on_conflict="user_id,source_file_name")
            .execute()
        )
        if not result.data:
            raise RuntimeError("dodos_raw_files upsert 결과가 없습니다.")
        return result.data[0]

    def _upsert_session(
        self,
        *,
        raw_file_id: str,
        path: Path,
        practice_date: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        clubs = {
            str(row.get("Club"))
            for row in rows
            if row.get("Club")
        }
        payload = {
            "user_id": self.user_id,
            "practice_date": practice_date,
            "raw_file_id": raw_file_id,
            "source_file_name": path.name,
            "shot_count": len(rows),
            "club_count": len(clubs),
            "practice_type": "trackman",
            "status": "complete",
        }

        result = (
            self.client.table("dodos_practice_sessions")
            .upsert(
                payload,
                on_conflict="user_id,practice_date,source_file_name",
            )
            .execute()
        )
        if not result.data:
            raise RuntimeError("dodos_practice_sessions upsert 결과가 없습니다.")
        return result.data[0]

    def _upsert_shots(
        self,
        *,
        session_id: str,
        practice_date: str,
        rows: list[dict[str, Any]],
    ) -> None:
        db_rows = [
            _shot_db_row(
                row,
                session_id=session_id,
                user_id=self.user_id,
                practice_date=practice_date,
                index=index,
            )
            for index, row in enumerate(rows)
        ]

        # PostgREST payload를 작게 유지
        chunk_size = 200
        for start in range(0, len(db_rows), chunk_size):
            chunk = db_rows[start:start + chunk_size]
            (
                self.client.table("dodos_shots")
                .upsert(chunk, on_conflict="session_id,shot_key")
                .execute()
            )

    def _upsert_club_summary(
        self,
        *,
        session_id: str,
        rows: list[dict[str, Any]],
    ) -> None:
        summaries = make_summary(rows)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            club = str(row.get("Club") or "")
            if club:
                grouped.setdefault(club, []).append(row)

        payloads = []
        for summary in summaries:
            club = str(summary.get("Club") or "")
            if not club:
                continue
            payloads.append(
                _summary_db_row(
                    summary,
                    grouped.get(club, []),
                    session_id=session_id,
                    user_id=self.user_id,
                )
            )

        if payloads:
            (
                self.client.table("dodos_session_club_summary")
                .upsert(payloads, on_conflict="session_id,club")
                .execute()
            )

    def _verify_and_mark(
        self,
        *,
        raw_file_id: str,
        session_id: str,
        expected_shots: int,
    ) -> int:
        count_result = (
            self.client.table("dodos_shots")
            .select("id", count="exact")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        actual = int(count_result.count or 0)
        verified = actual == expected_shots
        now = datetime.now(timezone.utc).isoformat()

        (
            self.client.table("dodos_raw_files")
            .update({
                "archived_shot_count": actual,
                "shot_db_verified": verified,
                "archive_status": "verified" if verified else "failed",
                "archive_error": None if verified else (
                    f"shot count mismatch: JSON={expected_shots}, DB={actual}"
                ),
                "archived_at": now,
                "verified_at": now if verified else None,
            })
            .eq("id", raw_file_id)
            .execute()
        )

        return actual


    # ---------------------------------------------------------
    # AI snapshot
    # ---------------------------------------------------------
    def _target_level(self) -> str:
        result = (
            self.client.table("dodos_user_settings")
            .select("target_level")
            .eq("user_id", self.user_id)
            .limit(1)
            .execute()
        )
        if result.data:
            value = str(result.data[0].get("target_level") or "").strip()
            if value in {"bogey", "80s", "single"}:
                return value
        return "single"

    def _ai_run_exists(self, session_id: str) -> bool:
        result = (
            self.client.table("dodos_ai_runs")
            .select("id")
            .eq("session_id", session_id)
            .eq("is_current", True)
            .limit(1)
            .execute()
        )
        return bool(result.data)

    def _load_ai_dataframe(self) -> pd.DataFrame:
        page_size = 1000
        start = 0
        records: list[dict[str, Any]] = []

        while True:
            result = (
                self.client.table("dodos_shots")
                .select("*")
                .eq("user_id", self.user_id)
                .order("practice_date")
                .order("stroke_no")
                .range(start, start + page_size - 1)
                .execute()
            )
            batch = list(result.data or [])
            if not batch:
                break

            for item in batch:
                records.append({
                    ai_name: item.get(db_name)
                    for db_name, ai_name in DB_TO_AI_COLUMNS.items()
                })

            if len(batch) < page_size:
                break
            start += page_size

        return pd.DataFrame(records)

    def _save_ai_snapshot(
        self,
        *,
        session_id: str,
        practice_date: str,
        force: bool = False,
    ) -> None:
        if self._ai_run_exists(session_id) and not force:
            return

        try:
            from ai import diagnose_practice
        except Exception as exc:
            raise RuntimeError(
                f"현재 프로젝트의 ai 패키지를 불러오지 못했습니다: {exc}"
            ) from exc

        df = self._load_ai_dataframe()
        if df.empty:
            raise RuntimeError("AI 분석용 dodos_shots 데이터가 없습니다.")

        goal = self._target_level()
        report = diagnose_practice(
            df,
            practice_date,
            goal=goal,
            recent_sessions=10,
            min_shots_per_club=2,
        )

        # 기존 current는 history로 남김
        (
            self.client.table("dodos_ai_runs")
            .update({"is_current": False})
            .eq("session_id", session_id)
            .eq("is_current", True)
            .execute()
        )

        run_result = (
            self.client.table("dodos_ai_runs")
            .insert({
                "user_id": self.user_id,
                "session_id": session_id,
                "model_name": "DODOS AI Coach",
                "model_version": "v6",
                "scoring_version": "goal-aware+wedge-v5+coach-v6",
                "target_level": report.goal,
                "is_current": True,
                "parameters": {
                    "recent_sessions": 10,
                    "min_shots_per_club": 2,
                },
            })
            .execute()
        )
        if not run_result.data:
            raise RuntimeError("dodos_ai_runs insert 결과가 없습니다.")
        ai_run_id = str(run_result.data[0]["id"])

        full_report = _jsonable(report.to_dict())

        (
            self.client.table("dodos_ai_session_reports")
            .insert({
                "ai_run_id": ai_run_id,
                "session_id": session_id,
                "overall_score": _f(report.score),
                "grade": report.grade,
                "confidence": int(report.confidence),
                "performance": _f(report.performance_score),
                "consistency": _f(report.consistency_score),
                "trend": _f(report.trend_score),
                "best_club": report.best_club or None,
                "best_club_score": _f(report.best_club_score),
                "focus_club": report.focus_club or None,
                "focus_club_score": _f(report.focus_club_score),
                "category_scores": _jsonable(report.category_scores),
                "strengths": _jsonable(report.strengths),
                "improvements": _jsonable(report.improvements),
                "tasks": _jsonable(report.tasks),
                "coaching_summary": report.coaching_summary,
                "full_report": full_report,
            })
            .execute()
        )

        club_rows = []
        for item in report.clubs:
            club_rows.append({
                "ai_run_id": ai_run_id,
                "session_id": session_id,
                "club": item.club,
                "shot_count": int(item.shots),
                "score": _f(item.score),
                "grade": item.grade,
                "confidence": int(item.confidence),
                "performance": _f(item.performance_score),
                "consistency": _f(item.consistency_score),
                "trend": _f(item.trend_score),
                "metrics": _jsonable(item.metrics),
                "strengths": _jsonable(item.strengths),
                "improvements": _jsonable(item.improvements),
                "tasks": _jsonable(item.tasks),
            })

        if club_rows:
            (
                self.client.table("dodos_ai_club_reports")
                .insert(club_rows)
                .execute()
            )

        (
            self.client.table("dodos_practice_sessions")
            .update({
                "current_ai_score": _f(report.score),
                "current_ai_grade": report.grade,
                "current_target_level": report.goal,
            })
            .eq("id", session_id)
            .execute()
        )

        categories = report.category_scores or {}

        def cat(*names: str) -> float | None:
            for name in names:
                if name in categories:
                    return _f(categories[name])
            return None

        (
            self.client.table("dodos_daily_snapshots")
            .upsert({
                "user_id": self.user_id,
                "session_id": session_id,
                "practice_date": practice_date,
                "overall_score": _f(report.score),
                "driver_score": cat("드라이버", "Driver", "driver"),
                "wood_score": cat("우드", "Wood", "wood"),
                "hybrid_score": cat("유틸리티", "Hybrid", "Utility", "hybrid"),
                "iron_score": cat("아이언", "Iron", "iron"),
                "wedge_score": cat("웨지", "Wedge", "wedge"),
                "best_club": report.best_club or None,
                "best_club_score": _f(report.best_club_score),
                "focus_club": report.focus_club or None,
                "focus_club_score": _f(report.focus_club_score),
                "target_level": report.goal,
                "shot_count": int(report.total_shots),
                "snapshot_metrics": {
                    "ai_run_id": ai_run_id,
                    "category_stars": _jsonable(report.category_stars),
                    "headline": report.headline,
                    "model_version": "v6",
                    "scoring_version": "goal-aware+wedge-v5+coach-v6",
                },
            }, on_conflict="user_id,practice_date")
            .execute()
        )

    def archive_file(
        self,
        path: str | Path,
        *,
        force: bool = False,
    ) -> ArchiveResult:
        source = Path(path).expanduser().resolve()

        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or "StrokeGroups" not in payload:
                raise ValueError("StrokeGroups가 없는 TrackMan JSON입니다.")

            rows = parse_trackman_report(payload)
            practice_date = _practice_date(payload, rows)
            expected = len(rows)

            if expected == 0:
                raise ValueError("파싱된 샷이 0개입니다.")

            existing = self._find_existing_session(
                practice_date=practice_date,
                source_file_name=source.name,
            )

            if existing and not force:
                session_id = str(existing["id"])
                count_result = (
                    self.client.table("dodos_shots")
                    .select("id", count="exact")
                    .eq("session_id", session_id)
                    .limit(1)
                    .execute()
                )
                actual = int(count_result.count or 0)
                if actual == expected:
                    self._save_ai_snapshot(
                        session_id=session_id,
                        practice_date=practice_date,
                        force=False,
                    )
                    return ArchiveResult(
                        file_name=source.name,
                        practice_date=practice_date,
                        expected_shots=expected,
                        archived_shots=actual,
                        verified=True,
                        skipped=True,
                    )

            raw_file = self._upsert_raw_file(
                path=source,
                practice_date=practice_date,
                expected_shots=expected,
            )
            session = self._upsert_session(
                raw_file_id=str(raw_file["id"]),
                path=source,
                practice_date=practice_date,
                rows=rows,
            )
            session_id = str(session["id"])

            self._upsert_shots(
                session_id=session_id,
                practice_date=practice_date,
                rows=rows,
            )
            self._upsert_club_summary(
                session_id=session_id,
                rows=rows,
            )
            actual = self._verify_and_mark(
                raw_file_id=str(raw_file["id"]),
                session_id=session_id,
                expected_shots=expected,
            )

            if actual == expected:
                self._save_ai_snapshot(
                    session_id=session_id,
                    practice_date=practice_date,
                    force=force,
                )

            return ArchiveResult(
                file_name=source.name,
                practice_date=practice_date,
                expected_shots=expected,
                archived_shots=actual,
                verified=(actual == expected),
                skipped=False,
                error=None if actual == expected else (
                    f"shot count mismatch: JSON={expected}, DB={actual}"
                ),
            )

        except Exception as exc:
            return ArchiveResult(
                file_name=source.name,
                practice_date="",
                expected_shots=0,
                archived_shots=0,
                verified=False,
                skipped=False,
                error=str(exc),
            )

    def archive_directory(
        self,
        report_dir: str | Path,
        *,
        force: bool = False,
    ) -> list[ArchiveResult]:
        report_dir = Path(report_dir).expanduser().resolve()
        paths = sorted(
            p for p in report_dir.glob("*.json")
            if p.is_file() and not p.name.startswith("_")
        )
        return [
            self.archive_file(path, force=force)
            for path in paths
        ]
