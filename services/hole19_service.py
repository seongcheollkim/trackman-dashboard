from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests


HOLE19_HOSTS = {"www.hole19golf.com", "hole19golf.com"}


@dataclass(frozen=True)
class Hole19Round:
    source_round_id: str
    source_url: str
    course_name: str
    played_at: str | None
    scoring_mode: str | None
    course_par: int | None
    holes_number: int
    playing_hcp: int | None
    holes: list[dict[str, Any]]


def extract_round_id(url: str) -> str:
    value = str(url or "").strip()
    parsed = urlparse(value)
    if parsed.netloc.lower() not in HOLE19_HOSTS:
        raise ValueError("Hole19 라운드 URL만 입력할 수 있습니다.")
    match = re.search(r"/performance/rounds/([^/?#]+)", parsed.path)
    if not match:
        raise ValueError("Hole19 라운드 URL 형식이 아닙니다. /performance/rounds/{id} 형식이어야 합니다.")
    return match.group(1)


def _walk_for_round(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if isinstance(value.get("data"), dict):
            candidate = value["data"]
            if "course_name" in candidate and isinstance(candidate.get("holes"), list):
                return candidate
        if "course_name" in value and isinstance(value.get("holes"), list):
            return value
        for child in value.values():
            found = _walk_for_round(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _walk_for_round(child)
            if found:
                return found
    return None


def parse_hole19_html(source_html: str) -> dict[str, Any]:
    """Extract the server-rendered JSON payload embedded in a Hole19 round page."""
    text = html.unescape(source_html)
    scripts = re.findall(
        r'<script[^>]+type=["\']application/json["\'][^>]*>(.*?)</script>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # The current Hole19 page uses an application/json script containing a `data` object.
    # We intentionally inspect every JSON script so the importer is resilient to React
    # component ordering changes.
    for raw in scripts:
        candidate = raw.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        found = _walk_for_round(payload)
        if found:
            return found

    # Fallback for pages where the script tag is not easy to match because of attribute order.
    for match in re.finditer(r'\{"data":\s*\{.*?"holes":\s*\[', text, flags=re.DOTALL):
        start = match.start()
        # Try progressively larger windows around the embedded object.
        for end in (len(text), min(len(text), start + 250000)):
            fragment = text[start:end]
            try:
                payload, _ = json.JSONDecoder().raw_decode(fragment)
            except json.JSONDecodeError:
                continue
            found = _walk_for_round(payload)
            if found:
                return found

    raise ValueError("Hole19 라운드 JSON 데이터를 페이지에서 찾지 못했습니다.")


def _to_int(value: Any) -> int | None:
    try:
        return None if value is None else int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def normalize_hole19_round(data: dict[str, Any], *, source_url: str, source_round_id: str) -> Hole19Round:
    holes: list[dict[str, Any]] = []
    for raw in data.get("holes") or []:
        tee = raw.get("hole_tee") or {}
        score = raw.get("hole_score") or {}
        holes.append(
            {
                "hole_number": _to_int(raw.get("sequence")),
                "par": _to_int(tee.get("par")),
                "stroke_index": _to_int(tee.get("stroke_index")),
                "distance_m": _to_float(tee.get("distance")),
                "score": _to_int(score.get("total_of_strokes")),
                "putts": _to_int(score.get("total_of_putts")),
                "fir": score.get("fairway_hit"),
                "gir": bool(score.get("green_in_regulation")) if score.get("green_in_regulation") is not None else None,
                "sand_shots": _to_int(score.get("total_of_sand_shots")),
                "penalties": _to_int(score.get("total_of_penalties")),
                "extra_strokes": _to_int(raw.get("extra_strokes")),
                "net_score": _to_int(score.get("net_score")),
                "stableford_points": _to_int(score.get("stableford_points")),
            }
        )

    holes.sort(key=lambda item: item.get("hole_number") or 999)
    return Hole19Round(
        source_round_id=source_round_id,
        source_url=source_url,
        course_name=str(data.get("course_name") or ""),
        played_at=data.get("played_at"),
        scoring_mode=data.get("scoring_mode"),
        course_par=_to_int(data.get("course_par")),
        holes_number=_to_int(data.get("holes_number")) or len(holes),
        playing_hcp=_to_int(data.get("playing_hcp")),
        holes=holes,
    )


def fetch_hole19_round(url: str, *, timeout: int = 15) -> Hole19Round:
    source_round_id = extract_round_id(url)
    normalized_url = f"https://www.hole19golf.com/performance/rounds/{source_round_id}"
    response = requests.get(
        normalized_url,
        timeout=timeout,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/150 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
        },
    )
    response.raise_for_status()
    data = parse_hole19_html(response.text)
    return normalize_hole19_round(data, source_url=normalized_url, source_round_id=source_round_id)
