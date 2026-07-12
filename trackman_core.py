from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

OUTPUT_COLUMNS = [
    "Date", "GroupId", "GroupClub", "StrokeNo", "StrokeId", "StrokeTime",
    "Club", "Ball", "MeasurementKind",
    "ClubSpeed_mps", "BallSpeed_mps", "SmashFactor", "Carry_m", "Total_m", "Run_m",
    "CarrySide_m", "TotalSide_m", "AbsTotalSide_m",
    "AttackAngle_deg", "ClubPath_deg", "FaceAngle_deg", "FaceToPath_deg",
    "LaunchAngle_deg", "LaunchDirection_deg", "SpinRate_rpm", "SpinAxis_deg",
    "MaxHeight_m", "LandingAngle_deg", "HangTime_s",
    "DynamicLoft_deg", "SpinLoft_deg", "SwingPlane_deg", "SwingDirection_deg",
    "LowPointDistance_m", "LowPointHeight_m", "LowPointSide_m",
    "ImpactOffset_mm", "ImpactHeight_mm", "DynamicLie_deg",
]

SUMMARY_COLUMNS = [
    "Club", "Shots",
    "Avg_Carry_m", "Std_Carry_m", "Avg_Total_m", "Std_Total_m", "Avg_Run_m",
    "Avg_AbsSide_m", "Avg_TotalSide_m",
    "Avg_BallSpeed_mps", "Avg_ClubSpeed_mps", "Avg_Smash",
    "Avg_Attack_deg", "Avg_Path_deg", "Avg_Face_deg", "Avg_FaceToPath_deg",
    "Avg_DynamicLoft_deg", "Avg_SpinLoft_deg", "Avg_Launch_deg", "Avg_LaunchDir_deg",
    "Avg_Spin_rpm", "Avg_SpinAxis_deg", "Avg_MaxHeight_m", "Avg_LandingAngle_deg",
    "Avg_SwingDirection_deg", "Avg_SwingPlane_deg",
    "Avg_LowPointDistance_m", "Avg_LowPointHeight_m", "Avg_LowPointSide_m",
    "Avg_ImpactOffset_mm", "Avg_ImpactHeight_mm", "Avg_DynamicLie_deg",
]

CLUB_ORDER = {
    "Driver": 1, "3Wood": 2, "5Wood": 3, "7Wood": 4,
    "2Hybrid": 5, "3Hybrid": 6, "4Hybrid": 7, "5Hybrid": 8,
    "4Iron": 9, "5Iron": 10, "6Iron": 11, "7Iron": 12,
    "8Iron": 13, "9Iron": 14, "PitchingWedge": 15, "GapWedge": 16,
    "50Wedge": 17, "52Wedge": 18, "56Wedge": 19, "SandWedge": 20,
}

CLUB_LABELS = {
    "Driver": "Dr", "3Wood": "3W", "5Wood": "5W", "7Wood": "7W",
    "2Hybrid": "2H", "3Hybrid": "3H", "4Hybrid": "4H", "5Hybrid": "5H",
    "4Iron": "4i", "5Iron": "5i", "6Iron": "6i", "7Iron": "7i", "8Iron": "8i", "9Iron": "9i",
    "PitchingWedge": "PW", "GapWedge": "GW", "50Wedge": "50°", "52Wedge": "52°", "56Wedge": "56°", "SandWedge": "SW",
}


def club_sort_key(club: str) -> tuple[int, str]:
    return (CLUB_ORDER.get(club, 999), club)


def club_label(club: str) -> str:
    return CLUB_LABELS.get(club, club)


def num(x: Any) -> float | None:
    return x if isinstance(x, (int, float)) and math.isfinite(x) else None


def rnd(x: Any, nd: int = 3) -> float | None:
    x = num(x)
    return round(x, nd) if x is not None else None


def avg(values: Iterable[Any]) -> float | None:
    vals = [num(v) for v in values]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def sd(values: Iterable[Any]) -> float | None:
    vals = [num(v) for v in values]
    vals = [v for v in vals if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else None


def load_trackman_json(path_or_obj: str | Path | dict) -> dict:
    if isinstance(path_or_obj, dict):
        return path_or_obj
    path = Path(path_or_obj)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_activities(data: dict) -> list[dict]:
    items = data.get("data", {}).get("me", {}).get("activities", {}).get("items", [])
    out = []
    for item in items:
        link = item.get("reportLink")
        if not link:
            continue
        out.append({
            "id": item.get("id"),
            "kind": item.get("kind"),
            "time": item.get("time"),
            "date": (item.get("time") or "")[:10],
            "strokeCount": item.get("strokeCount"),
            "reportLink": link,
        })
    return out


def find_activity_by_date(activities: list[dict], date: str) -> dict | None:
    matches = [a for a in activities if a.get("date") == date]
    if not matches:
        return None
    return sorted(matches, key=lambda a: a.get("strokeCount") or 0, reverse=True)[0]


def parse_trackman_report(data: dict, measurement_key: str = "NormalizedMeasurement") -> list[dict]:
    rows: list[dict] = []
    for group in data.get("StrokeGroups", []):
        group_id = group.get("Id")
        date = group.get("Date")
        group_club = group.get("Club")
        for i, stroke in enumerate(group.get("Strokes", []), start=1):
            measurement = stroke.get(measurement_key) or stroke.get("Measurement") or {}
            if not measurement:
                continue
            impact = stroke.get("ImpactLocation") or {}
            carry = num(measurement.get("Carry"))
            total = num(measurement.get("Total"))
            total_side = num(measurement.get("TotalSide"))
            impact_offset = measurement.get("ImpactOffset", impact.get("ImpactOffset"))
            impact_height = measurement.get("ImpactHeight", impact.get("ImpactHeight"))
            row = {
                "Date": date,
                "GroupId": group_id,
                "GroupClub": group_club,
                "StrokeNo": i,
                "StrokeId": stroke.get("Id"),
                "StrokeTime": stroke.get("Time"),
                "Club": stroke.get("Club") or group_club,
                "Ball": stroke.get("Ball") or group.get("Ball"),
                "MeasurementKind": measurement.get("Kind"),
                "ClubSpeed_mps": rnd(measurement.get("ClubSpeed"), 3),
                "BallSpeed_mps": rnd(measurement.get("BallSpeed"), 3),
                "SmashFactor": rnd(measurement.get("SmashFactor"), 3),
                "Carry_m": rnd(carry, 3),
                "Total_m": rnd(total, 3),
                "Run_m": rnd(total - carry, 3) if carry is not None and total is not None else None,
                "CarrySide_m": rnd(measurement.get("CarrySide"), 3),
                "TotalSide_m": rnd(total_side, 3),
                "AbsTotalSide_m": rnd(abs(total_side), 3) if total_side is not None else None,
                "AttackAngle_deg": rnd(measurement.get("AttackAngle"), 3),
                "ClubPath_deg": rnd(measurement.get("ClubPath"), 3),
                "FaceAngle_deg": rnd(measurement.get("FaceAngle"), 3),
                "FaceToPath_deg": rnd(measurement.get("FaceToPath"), 3),
                "LaunchAngle_deg": rnd(measurement.get("LaunchAngle"), 3),
                "LaunchDirection_deg": rnd(measurement.get("LaunchDirection"), 3),
                "SpinRate_rpm": rnd(measurement.get("SpinRate"), 0),
                "SpinAxis_deg": rnd(measurement.get("SpinAxis"), 3),
                "MaxHeight_m": rnd(measurement.get("MaxHeight"), 3),
                "LandingAngle_deg": rnd(measurement.get("LandingAngle"), 3),
                "HangTime_s": rnd(measurement.get("HangTime"), 3),
                "DynamicLoft_deg": rnd(measurement.get("DynamicLoft"), 3),
                "SpinLoft_deg": rnd(measurement.get("SpinLoft"), 3),
                "SwingPlane_deg": rnd(measurement.get("SwingPlane"), 3),
                "SwingDirection_deg": rnd(measurement.get("SwingDirection"), 3),
                "LowPointDistance_m": rnd(measurement.get("LowPointDistance"), 3),
                "LowPointHeight_m": rnd(measurement.get("LowPointHeight"), 3),
                "LowPointSide_m": rnd(measurement.get("LowPointSide"), 3),
                "ImpactOffset_mm": rnd(impact_offset * 1000, 1) if num(impact_offset) is not None else None,
                "ImpactHeight_mm": rnd(impact_height * 1000, 1) if num(impact_height) is not None else None,
                "DynamicLie_deg": rnd(measurement.get("DynamicLie"), 3),
            }
            rows.append(row)
    return rows


def make_summary(rows: list[dict]) -> list[dict]:
    by_club: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("Club"):
            by_club[row["Club"]].append(row)
    summary = []
    for club in sorted(by_club, key=club_sort_key):
        r = by_club[club]
        summary.append({
            "Club": club,
            "Shots": len(r),
            "Avg_Carry_m": rnd(avg(x.get("Carry_m") for x in r), 1),
            "Std_Carry_m": rnd(sd(x.get("Carry_m") for x in r), 1),
            "Avg_Total_m": rnd(avg(x.get("Total_m") for x in r), 1),
            "Std_Total_m": rnd(sd(x.get("Total_m") for x in r), 1),
            "Avg_Run_m": rnd(avg(x.get("Run_m") for x in r), 1),
            "Avg_AbsSide_m": rnd(avg(x.get("AbsTotalSide_m") for x in r), 1),
            "Avg_TotalSide_m": rnd(avg(x.get("TotalSide_m") for x in r), 1),
            "Avg_BallSpeed_mps": rnd(avg(x.get("BallSpeed_mps") for x in r), 1),
            "Avg_ClubSpeed_mps": rnd(avg(x.get("ClubSpeed_mps") for x in r), 1),
            "Avg_Smash": rnd(avg(x.get("SmashFactor") for x in r), 2),
            "Avg_Attack_deg": rnd(avg(x.get("AttackAngle_deg") for x in r), 1),
            "Avg_Path_deg": rnd(avg(x.get("ClubPath_deg") for x in r), 1),
            "Avg_Face_deg": rnd(avg(x.get("FaceAngle_deg") for x in r), 1),
            "Avg_FaceToPath_deg": rnd(avg(x.get("FaceToPath_deg") for x in r), 1),
            "Avg_DynamicLoft_deg": rnd(avg(x.get("DynamicLoft_deg") for x in r), 1),
            "Avg_SpinLoft_deg": rnd(avg(x.get("SpinLoft_deg") for x in r), 1),
            "Avg_Launch_deg": rnd(avg(x.get("LaunchAngle_deg") for x in r), 1),
            "Avg_LaunchDir_deg": rnd(avg(x.get("LaunchDirection_deg") for x in r), 1),
            "Avg_Spin_rpm": rnd(avg(x.get("SpinRate_rpm") for x in r), 0),
            "Avg_SpinAxis_deg": rnd(avg(x.get("SpinAxis_deg") for x in r), 1),
            "Avg_MaxHeight_m": rnd(avg(x.get("MaxHeight_m") for x in r), 1),
            "Avg_LandingAngle_deg": rnd(avg(x.get("LandingAngle_deg") for x in r), 1),
            "Avg_SwingDirection_deg": rnd(avg(x.get("SwingDirection_deg") for x in r), 1),
            "Avg_SwingPlane_deg": rnd(avg(x.get("SwingPlane_deg") for x in r), 1),
            "Avg_LowPointDistance_m": rnd(avg(x.get("LowPointDistance_m") for x in r), 3),
            "Avg_LowPointHeight_m": rnd(avg(x.get("LowPointHeight_m") for x in r), 3),
            "Avg_LowPointSide_m": rnd(avg(x.get("LowPointSide_m") for x in r), 3),
            "Avg_ImpactOffset_mm": rnd(avg(x.get("ImpactOffset_mm") for x in r), 1),
            "Avg_ImpactHeight_mm": rnd(avg(x.get("ImpactHeight_mm") for x in r), 1),
            "Avg_DynamicLie_deg": rnd(avg(x.get("DynamicLie_deg") for x in r), 1),
        })
    return summary


def write_csv(path: str | Path, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(report_json: dict, out_dir: str | Path, prefix: str | None = None, raw: bool = False) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = report_json.get("StrokeGroups") or []
    date = groups[0].get("Date") if groups else None
    prefix = prefix or f"trackman_{date or 'report'}"
    rows = parse_trackman_report(report_json, "Measurement" if raw else "NormalizedMeasurement")
    summary = make_summary(rows)
    shots_path = out_dir / f"{prefix}_shots.csv"
    summary_path = out_dir / f"{prefix}_club_summary.csv"
    write_csv(shots_path, rows, OUTPUT_COLUMNS)
    write_csv(summary_path, summary, SUMMARY_COLUMNS)
    return shots_path, summary_path
