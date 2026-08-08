from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .ai_score import numeric


@dataclass(frozen=True)
class Insight:
    kind: str               # strength / improvement
    title: str
    message: str
    priority: float
    metric: str
    club: str


def club_family(club: str) -> str:
    club = str(club)
    if club == "Driver":
        return "driver"
    if "Wood" in club:
        return "wood"
    if "Hybrid" in club:
        return "hybrid"
    if "Wedge" in club:
        return "wedge"
    if "Iron" in club:
        return "iron"
    return "other"


def korean_club(club: str) -> str:
    mapping = {
        "Driver": "드라이버",
        "3Wood": "3번 우드",
        "5Wood": "5번 우드",
        "7Wood": "7번 우드",
        "2Hybrid": "2번 유틸리티",
        "3Hybrid": "3번 유틸리티",
        "4Hybrid": "4번 유틸리티",
        "5Hybrid": "5번 유틸리티",
        "4Iron": "4번 아이언",
        "5Iron": "5번 아이언",
        "6Iron": "6번 아이언",
        "7Iron": "7번 아이언",
        "8Iron": "8번 아이언",
        "9Iron": "9번 아이언",
        "PitchingWedge": "피칭 웨지",
        "GapWedge": "갭 웨지",
        "50Wedge": "50도 웨지",
        "52Wedge": "52도 웨지",
        "56Wedge": "56도 웨지",
        "SandWedge": "샌드 웨지",
    }
    return mapping.get(str(club), str(club))


def _pct(today: float | None, baseline: float | None) -> float | None:
    t, b = numeric(today), numeric(baseline)
    if t is None or b is None or abs(b) < 1e-9:
        return None
    return (t - b) / abs(b) * 100.0


def _fmt(value: float | None, nd: int = 1, comma: bool = False) -> str:
    value = numeric(value)
    if value is None:
        return "-"
    if comma:
        return f"{value:,.0f}"
    return f"{value:.{nd}f}"


def build_metric_insights(
    *,
    club: str,
    today: dict[str, Any],
    baseline: dict[str, Any],
) -> list[Insight]:
    """오늘 vs 자기 과거 기준에서 설명력이 높은 진단 문장을 생성합니다."""
    name = korean_club(club)
    insights: list[Insight] = []

    def add_relative(
        metric: str,
        label: str,
        unit: str,
        *,
        higher_is_better: bool,
        good_threshold: float,
        bad_threshold: float,
        nd: int = 1,
    ) -> None:
        t = numeric(today.get(metric))
        b = numeric(baseline.get(metric))
        change = _pct(t, b)
        if t is None or b is None or change is None:
            return

        signed_good = change if higher_is_better else -change
        if signed_good >= good_threshold:
            insights.append(
                Insight(
                    "strength",
                    f"{name} {label}",
                    f"{label}이 {_fmt(t, nd)}{unit}로 최근 기준({_fmt(b, nd)}{unit})보다 "
                    f"{abs(change):.1f}% {'높아졌습니다' if higher_is_better else '줄었습니다'}.",
                    min(100.0, abs(change) * 3.0 + 20.0),
                    metric,
                    club,
                )
            )
        elif signed_good <= -bad_threshold:
            insights.append(
                Insight(
                    "improvement",
                    f"{name} {label}",
                    f"{label}이 {_fmt(t, nd)}{unit}로 최근 기준({_fmt(b, nd)}{unit}) 대비 "
                    f"{abs(change):.1f}% {'낮습니다' if higher_is_better else '커졌습니다'}.",
                    min(100.0, abs(change) * 3.0 + 20.0),
                    metric,
                    club,
                )
            )

    family = club_family(club)

    # 웨지는 여러 목표 거리를 섞어 연습할 수 있으므로 전체 Carry/Carry std를 평가하지 않습니다.
    if family != "wedge":
        add_relative("carry", "캐리", "m", higher_is_better=True, good_threshold=2.0, bad_threshold=4.0)
        add_relative("carry_std", "캐리 편차", "m", higher_is_better=False, good_threshold=10.0, bad_threshold=20.0)

    add_relative("ball_speed", "볼 스피드", "m/s", higher_is_better=True, good_threshold=1.5, bad_threshold=3.0)
    add_relative("smash", "스매시 팩터", "", higher_is_better=True, good_threshold=1.2, bad_threshold=2.5, nd=2)
    add_relative("abs_side", "좌우 편차", "m", higher_is_better=False, good_threshold=8.0, bad_threshold=15.0)

    # 런치/스핀은 높다고 무조건 좋은 지표가 아니므로 '평소 범위 이탈'만 경고합니다.
    for metric, label, unit, threshold, nd in [
        ("launch", "런치 앵글", "°", 18.0, 1),
        ("spin", "스핀량", "rpm", 25.0, 0),
    ]:
        t, b = numeric(today.get(metric)), numeric(baseline.get(metric))
        change = _pct(t, b)
        if t is None or b is None or change is None:
            continue
        if abs(change) >= threshold:
            insights.append(
                Insight(
                    "improvement",
                    f"{name} {label}",
                    f"{label}이 {_fmt(t, nd)}{unit}로 최근 기준({_fmt(b, nd)}{unit})에서 "
                    f"{abs(change):.1f}% 벗어났습니다. 탄도와 임팩트 조건을 함께 확인해 보세요.",
                    min(95.0, abs(change) * 2.2),
                    metric,
                    club,
                )
            )

    # 자기 기준이 부족할 때도 최소한의 절대 기준으로 피드백을 제공합니다.
    smash = numeric(today.get("smash"))
    side = numeric(today.get("abs_side"))
    family = club_family(club)
    smash_target = {
        "driver": 1.44, "wood": 1.40, "hybrid": 1.38,
        "iron": 1.32, "wedge": 1.20, "other": 1.30,
    }[family]
    side_target = {
        "driver": 13.0, "wood": 12.0, "hybrid": 11.0,
        "iron": 9.0, "wedge": 7.0, "other": 10.0,
    }[family]

    if family != "wedge" and smash is not None and not insights:
        if smash >= smash_target:
            insights.append(
                Insight(
                    "strength",
                    f"{name} 임팩트 효율",
                    f"스매시 팩터 {_fmt(smash, 2)}로 임팩트 효율이 안정적입니다.",
                    35.0,
                    "smash",
                    club,
                )
            )
        elif smash < smash_target - 0.06:
            insights.append(
                Insight(
                    "improvement",
                    f"{name} 임팩트 효율",
                    f"스매시 팩터가 {_fmt(smash, 2)}입니다. 타점과 힘 전달 효율을 우선 확인해 보세요.",
                    45.0,
                    "smash",
                    club,
                )
            )

    if side is not None and side <= side_target * 0.75:
        insights.append(
            Insight(
                "strength",
                f"{name} 방향성",
                f"평균 좌우 편차가 {_fmt(side, 1)}m로 방향성이 좋았습니다.",
                40.0,
                "abs_side",
                club,
            )
        )
    elif side is not None and side >= side_target * 1.5:
        insights.append(
            Insight(
                "improvement",
                f"{name} 방향성",
                f"평균 좌우 편차가 {_fmt(side, 1)}m입니다. 거리보다 페이스/패스 안정에 우선순위를 두세요.",
                55.0,
                "abs_side",
                club,
            )
        )

    return insights


def practice_task_from_insight(insight: Insight) -> str:
    name = korean_club(insight.club)
    metric = insight.metric
    if metric == "smash":
        return f"{name}: 70~80% 스윙 10구로 타점과 스매시 팩터를 먼저 안정화"
    if metric == "abs_side":
        return f"{name}: 목표선 기준 좌우 편차를 줄이는 방향성 샷 10구"
    if metric == "carry_std":
        return f"{name}: 같은 리듬·같은 피니시로 거리 편차 확인 10구"
    if metric in {"launch", "spin"}:
        return f"{name}: 탄도 5구씩 확인하며 볼 위치·임팩트 로프트를 일정하게 유지"
    if metric in {"carry", "ball_speed"}:
        return f"{name}: 힘을 더 쓰기보다 동일 리듬으로 중심 타격 10구"
    return f"{name}: 오늘 가장 흔들린 지표를 기준으로 품질 위주 10구"
