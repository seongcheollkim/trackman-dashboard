from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .ai_reference import club_target, goal_description, wedge_benchmark
from .ai_rules import Insight, build_metric_insights, club_family, korean_club, practice_task_from_insight
from .ai_score import (
    confidence_score,
    grade_from_score,
    max_target_score,
    min_target_score,
    numeric,
    range_target_score,
    relative_score,
    safe_mean,
    safe_std,
    weighted_score,
)



FAMILY_LABELS = {
    "driver": "드라이버",
    "wood": "우드",
    "hybrid": "유틸리티",
    "iron": "아이언",
    "wedge": "웨지",
    "other": "기타",
}


def _family_label(club: str) -> str:
    return FAMILY_LABELS.get(club_family(club), club_family(club))


def _star_rating(score: float) -> float:
    """0.5 단위의 5점 별점."""
    stars = max(0.5, min(5.0, float(score) / 20.0))
    return round(stars * 2.0) / 2.0

@dataclass(frozen=True)
class ClubDiagnosis:
    club: str
    score: float
    grade: str
    confidence: int
    shots: int
    baseline_sessions: int
    goal: str
    goal_label: str
    performance_score: float
    consistency_score: float
    trend_score: float
    metrics: dict[str, float | None]
    baseline: dict[str, float | None]
    target: dict[str, Any]
    strengths: tuple[str, ...]
    improvements: tuple[str, ...]
    tasks: tuple[str, ...]


@dataclass(frozen=True)
class PracticeDiagnosis:
    date: str
    score: float
    grade: str
    confidence: int
    total_shots: int
    goal: str
    goal_label: str
    goal_description: str
    performance_score: float
    consistency_score: float
    trend_score: float
    category_scores: dict[str, float]
    category_stars: dict[str, float]
    best_club: str
    best_club_score: float
    focus_club: str
    focus_club_score: float
    coaching_summary: str
    clubs: tuple[ClubDiagnosis, ...]
    headline: str
    strengths: tuple[str, ...]
    improvements: tuple[str, ...]
    tasks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_dates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["_ai_date"] = pd.to_datetime(result["Date"], errors="coerce")
    return result[result["_ai_date"].notna()].copy()


def _metric_snapshot(part: pd.DataFrame) -> dict[str, float | None]:
    if part.empty:
        return {}

    if "AbsTotalSide_m" in part.columns:
        abs_side_series = pd.to_numeric(part["AbsTotalSide_m"], errors="coerce")
    elif "TotalSide_m" in part.columns:
        abs_side_series = pd.to_numeric(part["TotalSide_m"], errors="coerce").abs()
    else:
        abs_side_series = pd.Series(dtype=float)

    carry = safe_mean(part["Carry_m"]) if "Carry_m" in part else None
    carry_std = safe_std(part["Carry_m"]) if "Carry_m" in part else None
    carry_cv = None
    if carry is not None and carry_std is not None and abs(carry) > 1e-9:
        carry_cv = carry_std / abs(carry) * 100.0

    return {
        "carry": carry,
        "ball_speed": safe_mean(part["BallSpeed_mps"]) if "BallSpeed_mps" in part else None,
        "club_speed": safe_mean(part["ClubSpeed_mps"]) if "ClubSpeed_mps" in part else None,
        "smash": safe_mean(part["SmashFactor"]) if "SmashFactor" in part else None,
        "abs_side": safe_mean(abs_side_series),
        "launch": safe_mean(part["LaunchAngle_deg"]) if "LaunchAngle_deg" in part else None,
        "spin": safe_mean(part["SpinRate_rpm"]) if "SpinRate_rpm" in part else None,
        "attack": safe_mean(part["AttackAngle_deg"]) if "AttackAngle_deg" in part else None,
        "face_to_path": safe_mean(part["FaceToPath_deg"].abs()) if "FaceToPath_deg" in part else None,
        "carry_std": carry_std,
        "side_std": safe_std(abs_side_series),
        "launch_std": safe_std(part["LaunchAngle_deg"]) if "LaunchAngle_deg" in part else None,
        "carry_cv_pct": carry_cv,
    }


def _daily_baseline(part: pd.DataFrame) -> dict[str, float | None]:
    if part.empty:
        return {}
    snapshots = [
        _metric_snapshot(group)
        for _, group in part.groupby(part["_ai_date"].dt.date)
    ]
    keys = {key for snapshot in snapshots for key in snapshot.keys()}
    result: dict[str, float | None] = {}
    for key in keys:
        values = pd.to_numeric(
            pd.Series([snapshot.get(key) for snapshot in snapshots]),
            errors="coerce",
        ).dropna()
        result[key] = float(values.mean()) if not values.empty else None
    return result



def _wedge_bucket_metrics(
    part: pd.DataFrame,
    *,
    benchmark: dict[str, Any],
) -> dict[str, float | int | None]:
    """
    웨지 전용 거리 버킷 평가.

    - Carry를 10m 단위(기본값)로 자동 그룹화
    - 최소 3샷 이상 버킷만 평가
    - 각 버킷 중심점(Carry, Side) 기준 2D 거리의 68 percentile을 탄착군 반경으로 사용
    - 전체 Carry 표준편차는 사용하지 않음
    """
    required = {"Carry_m", "TotalSide_m"}
    if part.empty or not required.issubset(part.columns):
        return {
            "valid_buckets": 0,
            "evaluated_shots": 0,
            "dispersion_radius": None,
            "lateral_abs_mean": None,
            "lateral_std": None,
            "big_miss_rate_pct": None,
            "launch_std": None,
        }

    data = part.copy()
    data["_w_carry"] = pd.to_numeric(data["Carry_m"], errors="coerce")
    data["_w_side"] = pd.to_numeric(data["TotalSide_m"], errors="coerce")
    data = data[data["_w_carry"].notna() & data["_w_side"].notna()].copy()
    if data.empty:
        return {
            "valid_buckets": 0,
            "evaluated_shots": 0,
            "dispersion_radius": None,
            "lateral_abs_mean": None,
            "lateral_std": None,
            "big_miss_rate_pct": None,
            "launch_std": None,
        }

    bucket_size = max(1, int(benchmark.get("bucket_size_m", 10)))
    min_shots = max(2, int(benchmark.get("min_shots_per_bucket", 3)))

    # 47m -> 50m, 52m -> 50m. Python round의 bankers rounding을 피합니다.
    data["_w_bucket"] = (
        np.floor((data["_w_carry"] + bucket_size / 2.0) / bucket_size) * bucket_size
    ).astype(int)

    target_radius = float(benchmark.get("dispersion_radius_max_m", 7.0))
    target_lateral = float(benchmark.get("lateral_abs_mean_max_m", 5.5))
    big_radius = target_radius * 1.5
    big_lateral = target_lateral * 1.5

    bucket_rows: list[dict[str, float]] = []

    for _, group in data.groupby("_w_bucket"):
        if len(group) < min_shots:
            continue

        carry = group["_w_carry"].to_numpy(dtype=float)
        side = group["_w_side"].to_numpy(dtype=float)

        # 평균보다 이상치에 덜 흔들리는 중앙값을 탄착군 중심으로 사용합니다.
        center_carry = float(np.median(carry))
        center_side = float(np.median(side))
        radial = np.sqrt((carry - center_carry) ** 2 + (side - center_side) ** 2)

        dispersion_radius = float(np.percentile(radial, 68))
        lateral_abs_mean = float(np.mean(np.abs(side)))
        lateral_std = float(np.std(side, ddof=1)) if len(side) >= 2 else 0.0

        big_miss = (radial > big_radius) | (np.abs(side) > big_lateral)
        big_miss_rate_pct = float(np.mean(big_miss) * 100.0)

        launch_std = None
        if "LaunchAngle_deg" in group.columns:
            launch = pd.to_numeric(group["LaunchAngle_deg"], errors="coerce").dropna()
            if len(launch) >= 2:
                launch_std = float(launch.std(ddof=1))

        bucket_rows.append({
            "shots": float(len(group)),
            "dispersion_radius": dispersion_radius,
            "lateral_abs_mean": lateral_abs_mean,
            "lateral_std": lateral_std,
            "big_miss_rate_pct": big_miss_rate_pct,
            "launch_std": launch_std if launch_std is not None else np.nan,
        })

    if not bucket_rows:
        return {
            "valid_buckets": 0,
            "evaluated_shots": 0,
            "dispersion_radius": None,
            "lateral_abs_mean": None,
            "lateral_std": None,
            "big_miss_rate_pct": None,
            "launch_std": None,
        }

    frame = pd.DataFrame(bucket_rows)
    weights = frame["shots"].to_numpy(dtype=float)

    def weighted_metric(column: str) -> float | None:
        values = pd.to_numeric(frame[column], errors="coerce")
        mask = values.notna().to_numpy()
        if not mask.any():
            return None
        return float(np.average(values.to_numpy(dtype=float)[mask], weights=weights[mask]))

    return {
        "valid_buckets": int(len(frame)),
        "evaluated_shots": int(frame["shots"].sum()),
        "dispersion_radius": weighted_metric("dispersion_radius"),
        "lateral_abs_mean": weighted_metric("lateral_abs_mean"),
        "lateral_std": weighted_metric("lateral_std"),
        "big_miss_rate_pct": weighted_metric("big_miss_rate_pct"),
        "launch_std": weighted_metric("launch_std"),
    }


def _wedge_history_baseline(
    history: pd.DataFrame,
    *,
    benchmark: dict[str, Any],
) -> dict[str, float | None]:
    """최근 세션들의 웨지 버킷 지표를 연습일 동일 가중 평균으로 계산합니다."""
    if history.empty:
        return {}

    session_metrics: list[dict[str, float | int | None]] = []
    for _, group in history.groupby(history["_ai_date"].dt.date):
        metrics = _wedge_bucket_metrics(group, benchmark=benchmark)
        if int(metrics.get("valid_buckets") or 0) > 0:
            session_metrics.append(metrics)

    if not session_metrics:
        return {}

    keys = [
        "dispersion_radius",
        "lateral_abs_mean",
        "lateral_std",
        "big_miss_rate_pct",
        "launch_std",
    ]
    result: dict[str, float | None] = {}
    for key in keys:
        values = pd.to_numeric(
            pd.Series([item.get(key) for item in session_metrics]),
            errors="coerce",
        ).dropna()
        result[key] = float(values.mean()) if not values.empty else None
    return result


def _wedge_performance_score(
    metrics: dict[str, float | int | None],
    benchmark: dict[str, Any],
) -> float:
    if int(metrics.get("valid_buckets") or 0) <= 0:
        return 72.0

    weights = benchmark.get("performance_weights", {})
    return weighted_score([
        (
            max_target_score(
                metrics.get("dispersion_radius"),
                benchmark["dispersion_radius_max_m"],
                soft_margin=max(3.0, float(benchmark["dispersion_radius_max_m"]) * .8),
            ),
            weights.get("dispersion_radius", .45),
        ),
        (
            max_target_score(
                metrics.get("lateral_abs_mean"),
                benchmark["lateral_abs_mean_max_m"],
                soft_margin=max(3.0, float(benchmark["lateral_abs_mean_max_m"]) * .8),
            ),
            weights.get("lateral_accuracy", .30),
        ),
        (
            max_target_score(
                metrics.get("big_miss_rate_pct"),
                benchmark["big_miss_rate_max_pct"],
                soft_margin=max(10.0, float(benchmark["big_miss_rate_max_pct"])),
            ),
            weights.get("big_miss_rate", .15),
        ),
        (
            max_target_score(
                metrics.get("launch_std"),
                benchmark["launch_std_max_deg"],
                soft_margin=4.0,
            ),
            weights.get("launch_stability", .10),
        ),
    ])


def _wedge_consistency_score(
    metrics: dict[str, float | int | None],
    benchmark: dict[str, Any],
) -> float:
    if int(metrics.get("valid_buckets") or 0) <= 0:
        return 72.0

    weights = benchmark.get("consistency_weights", {})
    return weighted_score([
        (
            max_target_score(
                metrics.get("dispersion_radius"),
                benchmark["dispersion_radius_max_m"],
                soft_margin=max(3.0, float(benchmark["dispersion_radius_max_m"])),
            ),
            weights.get("dispersion_radius", .60),
        ),
        (
            max_target_score(
                metrics.get("lateral_std"),
                benchmark["lateral_std_max_m"],
                soft_margin=max(3.0, float(benchmark["lateral_std_max_m"])),
            ),
            weights.get("lateral_spread", .40),
        ),
    ])


def _wedge_trend_score(
    today: dict[str, float | int | None],
    baseline: dict[str, float | None],
    benchmark: dict[str, Any],
) -> float:
    if not baseline:
        return 82.0

    weights = benchmark.get("trend_weights", {})
    return weighted_score([
        (
            relative_score(
                today.get("dispersion_radius"),
                baseline.get("dispersion_radius"),
                higher_is_better=False,
                neutral=85,
                sensitivity=.30,
            ),
            weights.get("dispersion_radius", .50),
        ),
        (
            relative_score(
                today.get("lateral_abs_mean"),
                baseline.get("lateral_abs_mean"),
                higher_is_better=False,
                neutral=85,
                sensitivity=.35,
            ),
            weights.get("lateral_accuracy", .30),
        ),
        (
            relative_score(
                today.get("big_miss_rate_pct"),
                baseline.get("big_miss_rate_pct"),
                higher_is_better=False,
                neutral=85,
                sensitivity=.50,
            ),
            weights.get("big_miss_rate", .20),
        ),
    ])


def _wedge_insights(
    *,
    club: str,
    metrics: dict[str, float | int | None],
    baseline: dict[str, float | None],
    benchmark: dict[str, Any],
) -> list[Insight]:
    """웨지 전용 진단 문장. 전체 Carry 분산은 절대 사용하지 않습니다."""
    name = korean_club(club)
    insights: list[Insight] = []

    radius = numeric(metrics.get("dispersion_radius"))
    lateral = numeric(metrics.get("lateral_abs_mean"))
    miss = numeric(metrics.get("big_miss_rate_pct"))
    valid_buckets = int(metrics.get("valid_buckets") or 0)

    if valid_buckets <= 0:
        return insights

    if radius is not None:
        target = float(benchmark["dispersion_radius_max_m"])
        if radius <= target:
            insights.append(Insight(
                "strength",
                f"{name} 탄착군",
                f"거리대별 68% 탄착군 반경이 {radius:.1f}m로 목표 기준 {target:.1f}m 이내입니다.",
                78.0,
                "wedge_dispersion",
                club,
            ))
        elif radius > target * 1.25:
            insights.append(Insight(
                "improvement",
                f"{name} 탄착군",
                f"거리대별 68% 탄착군 반경이 {radius:.1f}m로 목표 기준 {target:.1f}m보다 큽니다.",
                82.0,
                "wedge_dispersion",
                club,
            ))

    if lateral is not None:
        target = float(benchmark["lateral_abs_mean_max_m"])
        if lateral <= target:
            insights.append(Insight(
                "strength",
                f"{name} 좌우 정확도",
                f"거리대별 평균 좌우 편차가 {lateral:.1f}m로 목표 기준 {target:.1f}m 이내입니다.",
                74.0,
                "wedge_lateral",
                club,
            ))
        elif lateral > target * 1.25:
            insights.append(Insight(
                "improvement",
                f"{name} 좌우 정확도",
                f"거리대별 평균 좌우 편차가 {lateral:.1f}m로 목표 기준 {target:.1f}m보다 큽니다.",
                80.0,
                "wedge_lateral",
                club,
            ))

    if miss is not None:
        target = float(benchmark["big_miss_rate_max_pct"])
        if miss <= target:
            insights.append(Insight(
                "strength",
                f"{name} 큰 미스 관리",
                f"큰 미스 비율이 {miss:.0f}%로 목표 허용치 {target:.0f}% 이내입니다.",
                70.0,
                "wedge_miss",
                club,
            ))
        elif miss > target * 1.3:
            insights.append(Insight(
                "improvement",
                f"{name} 큰 미스 관리",
                f"큰 미스 비율이 {miss:.0f}%로 목표 허용치 {target:.0f}%보다 높습니다.",
                76.0,
                "wedge_miss",
                club,
            ))

    # 최근 대비 개선 메시지
    for key, label in [
        ("dispersion_radius", "탄착군 반경"),
        ("lateral_abs_mean", "좌우 편차"),
    ]:
        t = numeric(metrics.get(key))
        b = numeric(baseline.get(key))
        if t is None or b is None or b <= 1e-9:
            continue
        change = (b - t) / abs(b) * 100.0
        if change >= 10.0:
            insights.append(Insight(
                "strength",
                f"{name} {label} 개선",
                f"{label}이 최근 기준보다 {change:.1f}% 줄었습니다.",
                min(88.0, 60.0 + change),
                f"wedge_trend_{key}",
                club,
            ))

    return insights


def _wedge_task(insight: Insight) -> str:
    name = korean_club(insight.club)
    if insight.metric in {"wedge_dispersion", "wedge_trend_dispersion_radius"}:
        return f"{name}: 같은 거리대 5구씩 목표선을 정해 탄착군 크기 확인"
    if insight.metric in {"wedge_lateral", "wedge_trend_lateral_abs_mean"}:
        return f"{name}: 거리보다 시작 방향과 페이스 정렬을 우선해 좌우 편차 축소"
    if insight.metric == "wedge_miss":
        return f"{name}: 성공률 우선 70~80% 스윙으로 큰 미스 없는 10구 연속 연습"
    return f"{name}: 현재 거리별 루틴을 유지하며 탄착군 품질 확인"

def _performance_score(today: dict[str, float | None], target: dict[str, Any] | None) -> float:
    if not target:
        return 72.0

    weights = target["weights"]
    parts: list[tuple[float, float]] = []

    parts.append((
        min_target_score(today.get("smash"), target["smash_min"], soft_margin=0.10),
        weights.get("smash", 0.0),
    ))
    parts.append((
        max_target_score(today.get("abs_side"), target["abs_side_max_m"], soft_margin=max(4.0, target["abs_side_max_m"] * .75)),
        weights.get("abs_side", 0.0),
    ))
    parts.append((
        max_target_score(today.get("face_to_path"), target["face_to_path_abs_max_deg"], soft_margin=4.0),
        weights.get("face_to_path", 0.0),
    ))
    parts.append((
        max_target_score(today.get("carry_cv_pct"), target["carry_cv_max_pct"], soft_margin=max(3.0, target["carry_cv_max_pct"] * .8)),
        weights.get("carry_cv", 0.0),
    ))
    parts.append((
        range_target_score(today.get("launch"), *target["launch_range_deg"], soft_margin=6.0),
        weights.get("launch", 0.0),
    ))
    parts.append((
        range_target_score(today.get("spin"), *target["spin_range_rpm"], soft_margin=1800.0),
        weights.get("spin", 0.0),
    ))
    parts.append((
        range_target_score(today.get("attack"), *target["attack_range_deg"], soft_margin=5.0),
        weights.get("attack", 0.0),
    ))
    return weighted_score(parts)


def _trend_score(today: dict[str, float | None], baseline: dict[str, float | None]) -> float:
    """
    최근 최대 10회 대비 추세 점수.

    v3 원칙:
    - 이미 좋은 수준을 유지하는 골퍼가 '성장하지 않았다'는 이유로 감점되지 않도록
      최근 평균과 동일하면 약 85점을 부여합니다.
    - 뚜렷한 향상은 90~100점, 뚜렷한 하락은 70점 이하로 반영합니다.
    - Ball Speed / Carry는 핸디캡 절대기준이 아니라 이 Trend 영역에서만 평가합니다.
    """
    if not baseline:
        return 82.0

    return weighted_score([
        (
            relative_score(
                today.get("smash"),
                baseline.get("smash"),
                higher_is_better=True,
                neutral=85,
                sensitivity=.06,
            ),
            .22,
        ),
        (
            relative_score(
                today.get("ball_speed"),
                baseline.get("ball_speed"),
                higher_is_better=True,
                neutral=85,
                sensitivity=.08,
            ),
            .20,
        ),
        (
            relative_score(
                today.get("carry"),
                baseline.get("carry"),
                higher_is_better=True,
                neutral=85,
                sensitivity=.10,
            ),
            .16,
        ),
        (
            relative_score(
                today.get("abs_side"),
                baseline.get("abs_side"),
                higher_is_better=False,
                neutral=85,
                sensitivity=.35,
            ),
            .24,
        ),
        (
            relative_score(
                today.get("face_to_path"),
                baseline.get("face_to_path"),
                higher_is_better=False,
                neutral=85,
                sensitivity=.35,
            ),
            .18,
        ),
    ])

def _consistency_score(today: dict[str, float | None], target: dict[str, Any] | None) -> float:
    """
    당일 재현성을 절대 목표 기준으로 평가.
    방향성의 평균값은 absolute에서, 샷 간 흔들림은 여기서 평가합니다.
    """
    if not target:
        return 72.0

    carry_cv = max_target_score(
        today.get("carry_cv_pct"),
        target["carry_cv_max_pct"],
        soft_margin=max(3.0, target["carry_cv_max_pct"]),
    )

    # Side/launch std는 샷 수와 클럽별 특성 영향을 크게 받으므로
    # 지나치게 빡빡한 절대 컷 대신 목표 방향성의 크기를 기준으로 유연하게 평가.
    side_std_target = target["abs_side_max_m"] * .85
    side_std = max_target_score(
        today.get("side_std"),
        side_std_target,
        soft_margin=max(4.0, side_std_target),
    )

    launch_std_target = max(2.2, (target["launch_range_deg"][1] - target["launch_range_deg"][0]) * .45)
    launch_std = max_target_score(
        today.get("launch_std"),
        launch_std_target,
        soft_margin=3.5,
    )

    return weighted_score([
        (carry_cv, .50),
        (side_std, .35),
        (launch_std, .15),
    ])


def _headline(
    score: float,
    confidence: int,
    goal_label: str,
    performance: float,
    consistency: float,
    trend: float,
) -> str:
    if confidence < 45:
        return f"{goal_label} 기준 진단이지만 샷 수가 적어 오늘 결과는 참고용입니다."

    weakest = min(
        [("Performance", performance), ("Consistency", consistency), ("Trend", trend)],
        key=lambda item: item[1],
    )[0]

    if score >= 90:
        return (
            f"{goal_label} 기준으로 매우 좋은 연습입니다. "
            "스윙을 크게 바꾸기보다 현재 리듬과 재현성을 유지하세요."
        )
    if score >= 84:
        return (
            f"{goal_label} 기준에 상당히 근접했습니다. "
            f"현재 가장 낮은 영역인 {weakest}만 가볍게 보완하면 됩니다."
        )
    if score >= 76:
        return (
            f"{goal_label} 기준에서 무난한 수준입니다. "
            f"{weakest}에서 가장 큰 손실이 있으니 한 가지에만 집중하세요."
        )
    return (
        f"{goal_label} 목표와 비교하면 {weakest} 영역의 손실이 큽니다. "
        "한 번에 여러 요소를 바꾸지 말고 우선순위를 좁히세요."
    )


def _category_summary(clubs: list[ClubDiagnosis]) -> tuple[dict[str, float], dict[str, float]]:
    """
    오늘 연습을 드라이버/우드/유틸리티/아이언/웨지 그룹으로 요약.
    한 클럽을 많이 친 날이 과도하게 그룹 점수를 좌우하지 않도록
    클럽별 점수를 동일 가중 평균합니다.
    """
    buckets: dict[str, list[float]] = {}
    for item in clubs:
        label = _family_label(item.club)
        buckets.setdefault(label, []).append(float(item.score))

    scores = {
        label: round(sum(values) / len(values), 1)
        for label, values in buckets.items()
        if values
    }
    stars = {label: _star_rating(score) for label, score in scores.items()}
    return scores, stars


def _coaching_summary(
    goal_label: str,
    overall_score: float,
    best: ClubDiagnosis,
    focus: ClubDiagnosis,
) -> str:
    best_name = korean_club(best.club)
    focus_name = korean_club(focus.club)

    if overall_score >= 88:
        if focus.score >= 82:
            return (
                f"{best_name}을 포함해 전체 흐름이 좋습니다. "
                "큰 스윙 수정은 필요하지 않고 현재 리듬과 재현성을 유지하세요."
            )
        return (
            f"{best_name}은 {goal_label} 목표에 잘 맞았습니다. "
            f"전체 스윙을 바꾸기보다 {focus_name}의 약점만 선택적으로 보완하세요."
        )

    if best.score >= 88 and focus.score <= 75:
        return (
            f"{best_name}은 충분히 좋은 상태입니다. "
            f"오늘 점수를 낮춘 핵심은 {focus_name}이므로 다음 연습은 이 클럽에 우선순위를 두세요."
        )

    if focus.consistency_score + 8 < focus.performance_score:
        return (
            f"{focus_name}은 기본 샷 품질보다 재현성 손실이 큽니다. "
            "스윙을 바꾸기보다 같은 리듬과 같은 피니시를 반복하는 연습이 우선입니다."
        )

    if focus.performance_score + 8 < focus.consistency_score:
        return (
            f"{focus_name}은 반복성은 나쁘지 않지만 목표 수준의 샷 품질이 부족합니다. "
            "타점·방향성·탄도 중 가장 낮은 항목 하나만 교정하세요."
        )

    return (
        f"{best_name}의 좋은 감각은 유지하고 {focus_name}에 집중하세요. "
        "한 번에 여러 요소를 바꾸기보다 가장 큰 손실 지표 하나만 다루는 것이 효율적입니다."
    )



def _club_coaching_strength(item: ClubDiagnosis) -> str:
    """
    상단 '오늘 잘된 점'용 클럽 중심 문장.
    metric 이름 나열보다 사용자가 바로 이해할 수 있는 메시지를 우선합니다.
    """
    name = korean_club(item.club)
    family = club_family(item.club)

    if family == "wedge":
        radius = numeric(item.metrics.get("wedge_dispersion_radius"))
        lateral = numeric(item.metrics.get("wedge_lateral_abs_mean"))
        if item.score >= 90:
            if radius is not None:
                return (
                    f"{name}: 거리대별 탄착군이 매우 안정적입니다 "
                    f"(68% 반경 {radius:.1f}m). 현재 거리별 루틴을 유지하세요."
                )
            return f"{name}: 오늘 웨지 샷 품질이 매우 좋았습니다. 현재 거리별 루틴을 유지하세요."
        if lateral is not None and item.performance_score >= 85:
            return (
                f"{name}: 좌우 정확도와 탄착군이 목표 수준에 가깝습니다. "
                "거리별 루틴을 그대로 유지하세요."
            )

    if family == "driver":
        if item.performance_score >= 90 and item.consistency_score >= 82:
            return (
                f"{name}: 샷 품질과 재현성이 모두 좋습니다. "
                "스윙을 수정하기보다 현재 리듬을 유지하세요."
            )
        if item.performance_score >= 88:
            return (
                f"{name}: 목표 수준의 샷 품질을 보이고 있습니다. "
                "추가적인 스윙 수정 없이 리듬 유지가 우선입니다."
            )

    if item.consistency_score >= 90:
        return (
            f"{name}: 오늘 샷 재현성이 매우 좋았습니다. "
            "같은 템포와 피니시를 계속 유지하세요."
        )

    if item.trend_score >= 90:
        return (
            f"{name}: 최근 연습 대비 흐름이 뚜렷하게 좋아졌습니다. "
            "현재 감각을 반복해서 정착시키세요."
        )

    if item.performance_score >= 88:
        return (
            f"{name}: 목표 수준에 가까운 샷 품질입니다. "
            "현재 스윙을 크게 바꾸지 말고 좋은 감각을 유지하세요."
        )

    return (
        f"{name}: 오늘 전체적으로 안정적인 결과였습니다. "
        "좋았던 리듬과 셋업을 다음 연습에서도 반복하세요."
    )


def _club_coaching_improvement(item: ClubDiagnosis) -> str:
    """상단 '오늘 아쉬웠던 점'용 클럽 중심 문장."""
    name = korean_club(item.club)
    family = club_family(item.club)

    # 가장 낮은 축을 먼저 판단
    axes = [
        ("Performance", item.performance_score),
        ("Consistency", item.consistency_score),
        ("Trend", item.trend_score),
    ]
    weakest_axis, weakest_value = min(axes, key=lambda x: x[1])

    if family == "wedge":
        radius = numeric(item.metrics.get("wedge_dispersion_radius"))
        lateral = numeric(item.metrics.get("wedge_lateral_abs_mean"))
        miss = numeric(item.metrics.get("wedge_big_miss_rate_pct"))

        if radius is not None and item.performance_score < 80:
            return (
                f"{name}: 거리대별 탄착군이 아직 넓습니다 "
                f"(68% 반경 {radius:.1f}m). 거리마다 목표선을 정해 같은 거리 5구씩 묶어 연습하세요."
            )
        if lateral is not None and item.consistency_score < 80:
            return (
                f"{name}: 좌우 분산이 오늘 점수를 낮췄습니다 "
                f"(평균 {lateral:.1f}m). 거리보다 시작 방향과 페이스 정렬에 우선순위를 두세요."
            )
        if miss is not None and miss > 15:
            return (
                f"{name}: 큰 미스 비율이 {miss:.0f}%로 높았습니다. "
                "스윙 크기를 줄이고 성공률 높은 샷을 먼저 반복하세요."
            )

    if weakest_axis == "Consistency":
        return (
            f"{name}: 기본 샷 품질보다 재현성 손실이 큽니다. "
            "같은 리듬·같은 피니시로 10구를 반복하며 편차를 줄이세요."
        )

    if weakest_axis == "Performance":
        return (
            f"{name}: 반복성보다 목표 수준의 샷 품질이 부족합니다. "
            "방향성·타점·탄도 중 가장 약한 한 가지에만 집중하세요."
        )

    return (
        f"{name}: 최근 기록 대비 흐름이 좋지 않습니다. "
        "힘을 더 쓰기보다 평소 리듬과 셋업을 먼저 회복하세요."
    )


def _club_priority_task(item: ClubDiagnosis) -> str:
    """다음 연습 우선순위 문장."""
    name = korean_club(item.club)
    family = club_family(item.club)

    if family == "wedge":
        radius = numeric(item.metrics.get("wedge_dispersion_radius"))
        lateral = numeric(item.metrics.get("wedge_lateral_abs_mean"))

        if radius is not None and item.performance_score < 85:
            return f"{name}: 같은 거리대 5구씩 묶어서 탄착군 반경을 줄이는 연습"
        if lateral is not None and item.consistency_score < 85:
            return f"{name}: 목표선 기준 좌우 편차를 줄이는 방향성 연습 10구"
        return f"{name}: 현재 거리별 루틴을 유지하며 각 거리대 탄착군 확인"

    if item.consistency_score <= item.performance_score and item.consistency_score <= item.trend_score:
        return f"{name}: 같은 리듬·같은 피니시로 재현성 확인 10구"

    if item.performance_score <= item.trend_score:
        return f"{name}: 방향성과 중심 타격을 우선한 품질 샷 10구"

    return f"{name}: 평소 리듬과 셋업을 유지하는 회복 샷 10구"


def _daily_coaching_summary_blocks(
    clubs: list[ClubDiagnosis],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """
    v6 상단 요약.
    - metric별 상위 3개가 아니라 '클럽'을 기준으로 3개까지 선별
    - 같은 클럽이 장점/개선점에 동시에 과도하게 노출되지 않도록 함
    """
    if not clubs:
        return (), (), ()

    ranked_best = sorted(
        clubs,
        key=lambda item: (
            item.score,
            item.performance_score,
            item.consistency_score,
        ),
        reverse=True,
    )
    ranked_focus = sorted(
        clubs,
        key=lambda item: (
            item.score,
            item.consistency_score,
            item.performance_score,
        ),
    )

    strengths: list[str] = []
    used_strength_clubs: set[str] = set()
    for item in ranked_best:
        if item.club in used_strength_clubs:
            continue
        # 80점 미만 클럽까지 억지로 '장점'으로 올리지 않음
        if item.score < 80 and strengths:
            continue
        strengths.append(_club_coaching_strength(item))
        used_strength_clubs.add(item.club)
        if len(strengths) >= 3:
            break

    improvements: list[str] = []
    used_improvement_clubs: set[str] = set()
    for item in ranked_focus:
        if item.club in used_improvement_clubs:
            continue
        # 상위권 클럽을 굳이 개선 영역에 넣지 않음.
        if item.score >= 85 and improvements:
            continue
        improvements.append(_club_coaching_improvement(item))
        used_improvement_clubs.add(item.club)
        if len(improvements) >= 3:
            break

    tasks: list[str] = []
    for item in ranked_focus:
        task = _club_priority_task(item)
        if task not in tasks:
            tasks.append(task)
        if len(tasks) >= 3:
            break

    return tuple(strengths), tuple(improvements), tuple(tasks)


def diagnose_practice(
    df: pd.DataFrame,
    selected_date: str,
    *,
    goal: str = "80s",
    recent_sessions: int = 10,
    min_shots_per_club: int = 2,
) -> PracticeDiagnosis:
    empty = lambda message: PracticeDiagnosis(
        date=str(selected_date), score=0.0, grade="-", confidence=0, total_shots=0,
        goal=goal, goal_label=goal, goal_description="", performance_score=0.0,
        consistency_score=0.0, trend_score=0.0, category_scores={}, category_stars={},
        best_club="", best_club_score=0.0, focus_club="", focus_club_score=0.0,
        coaching_summary=message, clubs=(), headline=message,
        strengths=(), improvements=(), tasks=(),
    )

    if df is None or df.empty or "Date" not in df.columns or "Club" not in df.columns:
        return empty("분석할 TrackMan 데이터가 없습니다.")

    data = _clean_dates(df)
    target_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(target_date):
        raise ValueError(f"올바르지 않은 분석 날짜입니다: {selected_date}")

    day = data[data["_ai_date"].dt.date == target_date.date()].copy()
    if day.empty:
        return empty("선택한 날짜의 샷 데이터가 없습니다.")

    club_results: list[ClubDiagnosis] = []
    all_strengths: list[Insight] = []
    all_improvements: list[Insight] = []

    for club, today_part in day.groupby("Club"):
        club = str(club)
        if len(today_part) < min_shots_per_club:
            continue

        history = data[
            (data["Club"].astype(str) == club)
            & (data["_ai_date"].dt.date < target_date.date())
        ].copy()
        historical_dates = sorted(
            history["_ai_date"].dt.date.dropna().unique().tolist(),
            reverse=True,
        )[:recent_sessions]
        if historical_dates:
            history = history[history["_ai_date"].dt.date.isin(historical_dates)]

        today_metrics = _metric_snapshot(today_part)
        baseline_metrics = _daily_baseline(history)
        target = club_target(club, goal) or {}

        is_wedge = club_family(club) == "wedge"
        wedge_metrics: dict[str, float | int | None] = {}
        wedge_baseline: dict[str, float | None] = {}
        wedge_target: dict[str, Any] = {}

        if is_wedge:
            wedge_target = wedge_benchmark(goal)
            wedge_metrics = _wedge_bucket_metrics(
                today_part,
                benchmark=wedge_target,
            )
            wedge_baseline = _wedge_history_baseline(
                history,
                benchmark=wedge_target,
            )

            # AI 상세 진단에서 확인할 수 있도록 웨지 지표를 metrics에 같이 저장합니다.
            today_metrics.update({
                "wedge_valid_buckets": float(wedge_metrics.get("valid_buckets") or 0),
                "wedge_evaluated_shots": float(wedge_metrics.get("evaluated_shots") or 0),
                "wedge_dispersion_radius": wedge_metrics.get("dispersion_radius"),
                "wedge_lateral_abs_mean": wedge_metrics.get("lateral_abs_mean"),
                "wedge_lateral_std": wedge_metrics.get("lateral_std"),
                "wedge_big_miss_rate_pct": wedge_metrics.get("big_miss_rate_pct"),
                "wedge_launch_std": wedge_metrics.get("launch_std"),
            })

            if int(wedge_metrics.get("valid_buckets") or 0) > 0:
                performance = _wedge_performance_score(wedge_metrics, wedge_target)
                consistency = _wedge_consistency_score(wedge_metrics, wedge_target)
                trend = _wedge_trend_score(wedge_metrics, wedge_baseline, wedge_target)
            else:
                # 3샷 이상 거리 버킷이 없으면 기존 일반 엔진으로 폴백하되,
                # Confidence를 아래에서 낮춰 해석의 확신을 제한합니다.
                performance = _performance_score(today_metrics, target)
                consistency = _consistency_score(today_metrics, target)
                trend = _trend_score(today_metrics, baseline_metrics)
        else:
            performance = _performance_score(today_metrics, target)
            consistency = _consistency_score(today_metrics, target)
            trend = _trend_score(today_metrics, baseline_metrics)

        score = weighted_score([
            (performance, .50),
            (consistency, .30),
            (trend, .20),
        ])

        baseline_sessions = int(history["_ai_date"].dt.date.nunique()) if not history.empty else 0
        confidence = confidence_score(
            shots=len(today_part),
            baseline_sessions=baseline_sessions,
            baseline_shots=len(history),
        )
        if is_wedge and int(wedge_metrics.get("valid_buckets") or 0) == 0:
            confidence = min(confidence, 45)

        if is_wedge and int(wedge_metrics.get("valid_buckets") or 0) > 0:
            insights = _wedge_insights(
                club=club,
                metrics=wedge_metrics,
                baseline=wedge_baseline,
                benchmark=wedge_target,
            )
        else:
            insights = build_metric_insights(
                club=club,
                today=today_metrics,
                baseline=baseline_metrics,
            )
        strengths = sorted([x for x in insights if x.kind == "strength"], key=lambda x: x.priority, reverse=True)
        improvements = sorted([x for x in insights if x.kind == "improvement"], key=lambda x: x.priority, reverse=True)

        # 목표 절대 기준에서 크게 벗어난 핵심 항목을 개선점에 보충.
        # 웨지는 별도 탄착군 엔진에서 평가하므로 일반 기준을 추가하지 않습니다.
        if target and not is_wedge:
            smash = numeric(today_metrics.get("smash"))
            if smash is not None and smash < target["smash_min"] - .04:
                improvements.append(Insight(
                    "improvement", f"{korean_club(club)} 목표 스매시",
                    f"목표({target['goal_label']}) 기준 스매시 {target['smash_min']:.2f}에 비해 오늘 {smash:.2f}입니다.",
                    70, "smash", club
                ))
            side = numeric(today_metrics.get("abs_side"))
            if side is not None and side > target["abs_side_max_m"] * 1.25:
                improvements.append(Insight(
                    "improvement", f"{korean_club(club)} 목표 방향성",
                    f"평균 좌우 편차 {side:.1f}m로 {target['goal_label']} 목표 {target['abs_side_max_m']:.1f}m보다 큽니다.",
                    72, "abs_side", club
                ))

        if is_wedge:
            tasks = [_wedge_task(x) for x in improvements[:2]]
        else:
            tasks = [practice_task_from_insight(x) for x in improvements[:2]]

        if not tasks:
            if is_wedge:
                tasks = [f"{korean_club(club)}: 현재 거리별 루틴을 유지하며 각 거리대 탄착군 확인"]
            else:
                tasks = [f"{korean_club(club)}: 현재 좋은 리듬을 유지하는 품질 샷 10구"]

        all_strengths.extend(strengths)
        all_improvements.extend(improvements)

        club_results.append(ClubDiagnosis(
            club=club, score=score, grade=grade_from_score(score),
            confidence=confidence, shots=len(today_part),
            baseline_sessions=baseline_sessions, goal=goal,
            goal_label=str(target.get("goal_label", goal)),
            performance_score=round(performance, 1),
            consistency_score=round(consistency, 1),
            trend_score=round(trend, 1),
            metrics=today_metrics, baseline=baseline_metrics, target=target,
            strengths=tuple(x.message for x in strengths[:3]),
            improvements=tuple(x.message for x in improvements[:3]),
            tasks=tuple(tasks[:2]),
        ))

    if not club_results:
        return empty("클럽별 진단에 필요한 샷 수가 부족합니다.")

    weights = [max(1.0, np.sqrt(item.shots)) for item in club_results]
    def wavg(attr: str) -> float:
        return float(np.average([getattr(item, attr) for item in club_results], weights=weights))

    overall = wavg("score")
    performance = wavg("performance_score")
    consistency = wavg("consistency_score")
    trend = wavg("trend_score")
    confidence = int(round(np.average([item.confidence for item in club_results], weights=weights)))

    def unique(items: list[Insight], limit: int) -> list[Insight]:
        out, seen = [], set()
        for item in sorted(items, key=lambda x: x.priority, reverse=True):
            key = (item.club, item.metric)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= limit:
                break
        return out

    top_strengths = unique(all_strengths, 3)
    top_improvements = unique(all_improvements, 3)
    tasks = []
    for insight in top_improvements:
        if club_family(insight.club) == "wedge":
            task = _wedge_task(insight)
        else:
            task = practice_task_from_insight(insight)
        if task not in tasks:
            tasks.append(task)
    if not tasks:
        weakest = sorted(club_results, key=lambda item: item.score)
        for item in weakest[:3]:
            tasks.append(f"{korean_club(item.club)}: 현재 리듬을 유지하며 중심 타격 10구")

    label = club_results[0].goal_label if club_results else goal
    sorted_clubs = sorted(club_results, key=lambda item: item.score, reverse=True)
    best_club = sorted_clubs[0]
    focus_club = sorted_clubs[-1]
    category_scores, category_stars = _category_summary(club_results)
    coaching_summary = _coaching_summary(
        label,
        overall,
        best_club,
        focus_club,
    )

    # v6: 상단 요약은 개별 metric이 아니라 클럽 중심 코칭으로 구성합니다.
    daily_strengths, daily_improvements, daily_tasks = _daily_coaching_summary_blocks(
        club_results
    )

    return PracticeDiagnosis(
        date=str(selected_date),
        score=round(overall, 1),
        grade=grade_from_score(overall),
        confidence=confidence,
        total_shots=len(day),
        goal=goal,
        goal_label=label,
        goal_description=goal_description(goal),
        performance_score=round(performance, 1),
        consistency_score=round(consistency, 1),
        trend_score=round(trend, 1),
        category_scores=category_scores,
        category_stars=category_stars,
        best_club=best_club.club,
        best_club_score=round(best_club.score, 1),
        focus_club=focus_club.club,
        focus_club_score=round(focus_club.score, 1),
        coaching_summary=coaching_summary,
        clubs=tuple(sorted_clubs),
        headline=_headline(overall, confidence, label, performance, consistency, trend),
        strengths=daily_strengths,
        improvements=daily_improvements,
        tasks=daily_tasks,
    )
