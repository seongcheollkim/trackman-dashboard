from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


REFERENCE_FILE = Path(__file__).with_name("ai_reference.json")


@lru_cache(maxsize=1)
def load_reference_database() -> dict[str, Any]:
    with REFERENCE_FILE.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def goal_options() -> dict[str, str]:
    db = load_reference_database()
    return {
        key: profile["label"]
        for key, profile in db["goal_profiles"].items()
    }


def goal_description(goal: str) -> str:
    db = load_reference_database()
    profile = db["goal_profiles"].get(goal, db["goal_profiles"]["80s"])
    return str(profile.get("description", ""))


def _base_reference_for_club(club: str) -> dict[str, Any] | None:
    db = load_reference_database()
    refs = db["club_reference"]
    if club in refs:
        return dict(refs[club])

    if "Iron" in str(club):
        return dict(refs["7Iron"])
    if "Wedge" in str(club):
        return dict(refs["PitchingWedge"])
    if "Hybrid" in str(club):
        return dict(refs["4Hybrid"])
    if "Wood" in str(club):
        return dict(refs["5Wood"])
    return None


def club_target(club: str, goal: str) -> dict[str, Any] | None:
    """싱글 기준 레퍼런스를 목표 단계별 허용 범위로 변환합니다."""
    db = load_reference_database()
    profile = db["goal_profiles"].get(goal, db["goal_profiles"]["80s"])
    base = _base_reference_for_club(club)
    if base is None:
        return None

    result = dict(base)
    result["goal"] = goal
    result["goal_label"] = profile["label"]

    result["smash_min"] = max(
        0.8,
        float(base["smash_min"]) - float(profile["efficiency_relax"])
    )
    result["abs_side_max_m"] = float(base["abs_side_max_m"]) * float(profile["direction_multiplier"])
    result["face_to_path_abs_max_deg"] = float(base["face_to_path_abs_max_deg"]) * float(profile["path_multiplier"])
    result["carry_cv_max_pct"] = float(base["carry_cv_max_pct"]) * float(profile["consistency_multiplier"])

    launch_lo, launch_hi = [float(x) for x in base["launch_range_deg"]]
    launch_extra = float(profile["launch_extra"])
    result["launch_range_deg"] = [launch_lo - launch_extra, launch_hi + launch_extra]

    spin_lo, spin_hi = [float(x) for x in base["spin_range_rpm"]]
    result["spin_range_rpm"] = [
        spin_lo * float(profile["spin_multiplier_low"]),
        spin_hi * float(profile["spin_multiplier_high"]),
    ]

    attack_lo, attack_hi = [float(x) for x in base["attack_range_deg"]]
    attack_extra = float(profile["attack_extra"])
    result["attack_range_deg"] = [attack_lo - attack_extra, attack_hi + attack_extra]
    return result



def wedge_benchmark(goal: str) -> dict[str, Any]:
    """DODOS wedge dispersion provisional benchmark를 반환합니다."""
    db = load_reference_database()
    bench = db.get("wedge_benchmark", {})
    goals = bench.get("goals", {})
    goal_values = goals.get(goal, goals.get("80s", {}))
    result = dict(goal_values)
    result["bucket_size_m"] = int(bench.get("bucket_size_m", 10))
    result["min_shots_per_bucket"] = int(bench.get("min_shots_per_bucket", 3))
    result["performance_weights"] = dict(bench.get("performance_weights", {}))
    result["consistency_weights"] = dict(bench.get("consistency_weights", {}))
    result["trend_weights"] = dict(bench.get("trend_weights", {}))
    result["status"] = bench.get("status", "DODOS provisional benchmark")
    return result
