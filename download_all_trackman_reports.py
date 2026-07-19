#!/usr/bin/env python3
"""
TrackMan Bulk Downloader v3

지원 기능
- macOS Chrome의 $'...' ANSI-C quoting cURL 지원
- 일반 '...' / "..." / --data / --data-raw / --data-binary 지원
- getPlayerActivities take/skip 페이지네이션
- SHOT_ANALYSIS만 선별
- reportLink의 ?a=UUID 추출
- getactivityreport UUID 자동 치환
- 재실행 시 기존 파일 자동 건너뜀
- 실패 목록 저장 및 재시도
- dry-run / limit / force 지원
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests


@dataclass
class CurlRequest:
    method: str
    url: str
    headers: dict[str, str]
    data: str | None = None


@dataclass
class Activity:
    activity_id: str
    report_uuid: str
    time: str
    stroke_count: int
    kind: str
    report_link: str


DATA_FLAGS = {
    "-d",
    "--data",
    "--data-raw",
    "--data-binary",
    "--data-ascii",
    "--data-urlencode",
}


def normalize_multiline_curl(text: str) -> str:
    """백슬래시 줄바꿈을 하나의 명령으로 합친다."""
    return re.sub(r"\\\r?\n", " ", text).strip()


def decode_ansi_c_quoted(value: str) -> str:
    """
    Bash ANSI-C quoting 형태인 $'...'를 Python 문자열로 변환.
    예: $'{"query":"\\n ..."}'
    """
    if not (value.startswith("$'") and value.endswith("'")):
        return value

    inner = value[2:-1]

    # Python 문자열 리터럴로 안전하게 해석한다.
    # Bash와 Python escape 규칙이 대부분 호환된다.
    try:
        return ast.literal_eval("'" + inner.replace("'", "\\'") + "'")
    except Exception:
        pass

    # fallback
    try:
        return bytes(inner, "utf-8").decode("unicode_escape")
    except Exception:
        return inner


def tokenize_curl(text: str) -> list[str]:
    """
    shlex가 $'...'를 완전히 처리하지 못하는 문제를 보정한다.
    먼저 일반 shlex 파싱을 시도하고, data 플래그 뒤의 ANSI-C 문자열은
    원본 명령에서 별도로 찾아 교체한다.
    """
    normalized = normalize_multiline_curl(text)
    tokens = shlex.split(normalized, posix=True)

    # shlex 결과에서 "$" + "..."로 쪼개지는 경우를 보정
    repaired: list[str] = []
    i = 0
    while i < len(tokens):
        if (
            tokens[i] in DATA_FLAGS
            and i + 2 < len(tokens)
            and tokens[i + 1] == "$"
        ):
            repaired.extend([tokens[i], "$'" + tokens[i + 2] + "'"])
            i += 3
            continue
        repaired.append(tokens[i])
        i += 1

    return repaired


def extract_raw_data_argument(text: str) -> str | None:
    """
    원본 cURL에서 data 플래그 뒤 값을 직접 추출.
    특히 --data-raw $'...' 형식을 정확히 처리한다.
    """
    normalized = normalize_multiline_curl(text)

    patterns = [
        r"(?:--data-raw|--data-binary|--data-ascii|--data|-d)\s+(\$'(?:\\.|[^'])*')",
        r'(?:--data-raw|--data-binary|--data-ascii|--data|-d)\s+("(?:\\.|[^"])*")',
        r"(?:--data-raw|--data-binary|--data-ascii|--data|-d)\s+('(?:\\.|[^'])*')",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.DOTALL)
        if not match:
            continue

        raw = match.group(1)

        if raw.startswith("$'"):
            return decode_ansi_c_quoted(raw)

        if raw.startswith("'") and raw.endswith("'"):
            return raw[1:-1]

        if raw.startswith('"') and raw.endswith('"'):
            try:
                return ast.literal_eval(raw)
            except Exception:
                return raw[1:-1]

    return None


def parse_curl(text: str) -> CurlRequest:
    normalized = normalize_multiline_curl(text)
    tokens = tokenize_curl(text)

    if not tokens or tokens[0] != "curl":
        raise ValueError("cURL 파일이 curl 명령으로 시작하지 않습니다.")

    method = "GET"
    url: str | None = None
    headers: dict[str, str] = {}
    data: str | None = extract_raw_data_argument(text)

    i = 1
    while i < len(tokens):
        token = tokens[i]

        if token in ("-X", "--request"):
            i += 1
            if i >= len(tokens):
                raise ValueError("cURL의 요청 메서드 값이 없습니다.")
            method = tokens[i].upper()

        elif token in ("-H", "--header"):
            i += 1
            if i >= len(tokens):
                raise ValueError("cURL 헤더 값이 없습니다.")
            raw = tokens[i]
            if ":" in raw:
                name, value = raw.split(":", 1)
                headers[name.strip()] = value.strip()

        elif token in DATA_FLAGS:
            i += 1
            if data is None and i < len(tokens):
                candidate = tokens[i]
                data = decode_ansi_c_quoted(candidate)

            if method == "GET":
                method = "POST"

        elif token.startswith(("http://", "https://")):
            url = token

        i += 1

    if not url:
        raise ValueError("cURL에서 URL을 찾지 못했습니다.")

    # requests가 직접 계산하는 편이 안전한 헤더
    for key in list(headers):
        if key.lower() in {
            "content-length",
            "host",
            "connection",
            "accept-encoding",
        }:
            headers.pop(key, None)

    if data is not None and method == "GET":
        method = "POST"

    return CurlRequest(method=method, url=url, headers=headers, data=data)


def read_curl(path: Path) -> CurlRequest:
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"파일이 비어 있습니다: {path}")

    return parse_curl(text)


def decode_graphql_body(data: str | None) -> dict[str, Any]:
    if not data:
        raise ValueError(
            "활동 목록 cURL에 GraphQL Request Payload가 없습니다."
        )

    body = data.strip()

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        preview = body[:180].replace("\n", "\\n")
        raise ValueError(
            "GraphQL Payload를 JSON으로 해석하지 못했습니다.\n"
            f"Payload 시작 부분: {preview}"
        ) from exc

    if isinstance(payload, list):
        if not payload:
            raise ValueError("GraphQL Payload 배열이 비어 있습니다.")
        payload = payload[0]

    if not isinstance(payload, dict):
        raise ValueError("GraphQL Payload가 JSON 객체가 아닙니다.")

    variables = payload.get("variables")

    if isinstance(variables, str):
        try:
            variables = json.loads(variables)
            payload["variables"] = variables
        except json.JSONDecodeError as exc:
            raise ValueError(
                "GraphQL variables 문자열을 JSON으로 해석하지 못했습니다."
            ) from exc

    if not isinstance(variables, dict):
        raise ValueError("GraphQL Payload에 variables 객체가 없습니다.")

    required = ("take", "skip")
    missing = [key for key in required if key not in variables]
    if missing:
        raise ValueError(
            "GraphQL variables에 필수 항목이 없습니다: "
            + ", ".join(missing)
        )

    return payload


def send_json(
    session: requests.Session,
    req: CurlRequest,
    timeout: int = 90,
) -> Any:
    response = session.request(
        method=req.method,
        url=req.url,
        headers=req.headers,
        data=req.data,
        timeout=timeout,
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except requests.JSONDecodeError as exc:
        preview = response.text[:300].replace("\n", " ")
        raise RuntimeError(
            f"응답이 JSON이 아닙니다. HTTP {response.status_code}: {preview}"
        ) from exc

    if isinstance(payload, dict) and payload.get("errors"):
        raise RuntimeError(
            "GraphQL 오류: "
            + json.dumps(payload["errors"], ensure_ascii=False)
        )

    return payload


def extract_report_uuid(report_link: str) -> str | None:
    try:
        values = parse_qs(urlparse(report_link).query).get("a")
        if values and values[0]:
            return values[0]
    except Exception:
        pass

    match = re.search(
        r"[?&]a=([0-9a-fA-F-]{36})",
        report_link,
    )
    return match.group(1) if match else None


def parse_activities(payload: Any) -> tuple[list[Activity], bool, int]:
    try:
        activities_node = payload["data"]["me"]["activities"]
        items = activities_node["items"]
        page_info = activities_node["pageInfo"]
        total_count = int(activities_node.get("totalCount") or 0)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "getPlayerActivities 응답 구조가 예상과 다릅니다."
        ) from exc

    result: list[Activity] = []

    for item in items:
        if not isinstance(item, dict):
            continue

        if item.get("kind") != "SHOT_ANALYSIS":
            continue

        report_link = item.get("reportLink")
        if not report_link:
            continue

        report_uuid = extract_report_uuid(str(report_link))
        if not report_uuid:
            continue

        result.append(
            Activity(
                activity_id=str(item.get("id") or ""),
                report_uuid=report_uuid,
                time=str(item.get("time") or ""),
                stroke_count=int(item.get("strokeCount") or 0),
                kind=str(item.get("kind") or ""),
                report_link=str(report_link),
            )
        )

    return (
        result,
        bool(page_info.get("hasNextPage")),
        total_count,
    )


def fetch_all_activities(
    session: requests.Session,
    list_req: CurlRequest,
    page_size: int,
) -> tuple[list[Activity], int]:
    base_payload = decode_graphql_body(list_req.data)
    base_variables = dict(base_payload["variables"])

    all_activities: list[Activity] = []
    seen: set[str] = set()
    skip = 0
    server_total = 0
    page_number = 1

    while True:
        payload = dict(base_payload)
        variables = dict(base_variables)
        variables["take"] = page_size
        variables["skip"] = skip
        payload["variables"] = variables

        page_req = CurlRequest(
            method=list_req.method,
            url=list_req.url,
            headers=dict(list_req.headers),
            data=json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

        print(
            f"[목록 {page_number}] skip={skip}, take={page_size}"
        )

        response_payload = send_json(session, page_req)
        page_activities, has_next, total_count = parse_activities(
            response_payload
        )

        if total_count:
            server_total = total_count

        new_count = 0
        for activity in page_activities:
            if activity.report_uuid in seen:
                continue
            seen.add(activity.report_uuid)
            all_activities.append(activity)
            new_count += 1

        print(
            f"  Shot Analysis {new_count}개 추가 "
            f"(누적 {len(all_activities)}개 / 전체 활동 {server_total or '?'})"
        )

        if not has_next:
            break

        if new_count == 0 and page_activities:
            print(
                "경고: 다음 페이지가 있지만 모든 UUID가 중복입니다. "
                "무한 반복을 방지하기 위해 중단합니다."
            )
            break

        skip += page_size
        page_number += 1

    return all_activities, server_total


def find_reference_uuid(report_req: CurlRequest) -> str | None:
    haystack = report_req.url + "\n" + (report_req.data or "")

    match = re.search(
        r"(?:[?&]a=|activityId[\"'=:%20]+)"
        r"([0-9a-fA-F-]{36})",
        haystack,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)

    match = re.search(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}\b",
        haystack,
    )
    return match.group(0) if match else None


def replace_report_uuid(
    req: CurlRequest,
    reference_uuid: str,
    target_uuid: str,
) -> CurlRequest:
    return CurlRequest(
        method=req.method,
        url=req.url.replace(reference_uuid, target_uuid),
        headers=dict(req.headers),
        data=(
            req.data.replace(reference_uuid, target_uuid)
            if req.data
            else None
        ),
    )


def activity_date(activity: Activity) -> str:
    match = re.match(r"(20\d{2}-\d{2}-\d{2})", activity.time)
    return match.group(1) if match else "unknown-date"


def safe_filename(activity: Activity, duplicate_index: int) -> str:
    if duplicate_index == 1:
        return f"{activity_date(activity)}.json"

    return f"{activity_date(activity)}_{duplicate_index:02d}.json"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_progress(
    current: int,
    total: int,
    completed: int,
    skipped: int,
    failed: int,
) -> None:
    width = 28
    ratio = current / total if total else 1
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"[{bar}] {current}/{total} "
        f"완료:{completed} 건너뜀:{skipped} 실패:{failed}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TrackMan Shot Analysis 전체 JSON 일괄 다운로드 v3"
    )
    parser.add_argument(
        "--list-curl",
        default="activity_list.curl",
    )
    parser.add_argument(
        "--report-curl",
        default="activity_report.curl",
    )
    parser.add_argument(
        "--output",
        default="data/trackman_reports",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.6,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="이전 실패 UUID만 다시 시도",
    )
    parser.add_argument(
        "--reference-id",
        default=None,
    )
    args = parser.parse_args()

    if args.page_size <= 0:
        raise ValueError("--page-size는 1 이상이어야 합니다.")

    print("cURL 파일 분석 중...")
    list_req = read_curl(Path(args.list_curl))
    report_req = read_curl(Path(args.report_curl))

    # 진단 출력
    list_payload = decode_graphql_body(list_req.data)
    variables = list_payload["variables"]
    print(
        "활동 목록 요청 확인: "
        f"take={variables.get('take')}, "
        f"skip={variables.get('skip')}, "
        f"kind={variables.get('kind')}"
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    print("\n전체 활동 목록을 가져오는 중...")
    activities, server_total = fetch_all_activities(
        session,
        list_req,
        page_size=args.page_size,
    )

    print()
    print(f"전체 활동 수: {server_total or '-'}")
    print(f"다운로드 가능한 Shot Analysis: {len(activities)}개")

    index_payload = [
        {
            "activityId": activity.activity_id,
            "reportUuid": activity.report_uuid,
            "time": activity.time,
            "date": activity_date(activity),
            "strokeCount": activity.stroke_count,
            "kind": activity.kind,
            "reportLink": activity.report_link,
        }
        for activity in activities
    ]
    save_json(output_dir / "_activities_index.json", index_payload)

    if not activities:
        print("SHOT_ANALYSIS 활동을 찾지 못했습니다.")
        return 2

    failed_path = output_dir / "_failed.json"
    previous_failed = load_json(failed_path, [])

    if args.retry_failed:
        failed_ids = {
            item.get("reportUuid")
            for item in previous_failed
            if isinstance(item, dict)
        }
        activities = [
            activity
            for activity in activities
            if activity.report_uuid in failed_ids
        ]
        print(f"이전 실패 항목 재시도: {len(activities)}개")

    if args.limit is not None:
        activities = activities[: max(args.limit, 0)]

    if args.dry_run:
        print("\n[DRY RUN]")
        for index, activity in enumerate(activities, start=1):
            print(
                f"{index:4d}. {activity_date(activity)} | "
                f"{activity.stroke_count:3d} shots | "
                f"{activity.report_uuid}"
            )
        return 0

    reference_uuid = args.reference_id or find_reference_uuid(report_req)

    if not reference_uuid:
        print(
            "activity_report.curl에서 기준 UUID를 찾지 못했습니다.\n"
            "--reference-id UUID 옵션으로 직접 지정하세요."
        )
        return 2

    print(f"\n상세 요청 기준 UUID: {reference_uuid}")

    manifest_path = output_dir / "_manifest.json"
    manifest = load_json(manifest_path, {"downloaded": {}})
    if not isinstance(manifest, dict):
        manifest = {"downloaded": {}}
    manifest.setdefault("downloaded", {})
    downloaded: dict[str, Any] = manifest["downloaded"]

    date_counts: dict[str, int] = {}
    completed = 0
    skipped = 0
    failed = 0
    failed_items: list[dict[str, Any]] = []

    total = len(activities)

    for index, activity in enumerate(activities, start=1):
        report_uuid = activity.report_uuid
        date = activity_date(activity)
        date_counts[date] = date_counts.get(date, 0) + 1

        if not args.force and report_uuid in downloaded:
            skipped += 1
            print(
                f"\n[{index}/{total}] 건너뜀: "
                f"{date} ({activity.stroke_count} shots)"
            )
            print_progress(index, total, completed, skipped, failed)
            continue

        request = replace_report_uuid(
            report_req,
            reference_uuid,
            report_uuid,
        )

        try:
            report_payload = send_json(session, request)
            filename = safe_filename(
                activity,
                date_counts[date],
            )
            destination = output_dir / filename
            save_json(destination, report_payload)

            downloaded[report_uuid] = {
                "file": filename,
                "date": date,
                "time": activity.time,
                "strokeCount": activity.stroke_count,
                "activityId": activity.activity_id,
                "reportLink": activity.report_link,
                "downloadedAt": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }
            save_json(manifest_path, manifest)

            completed += 1
            print(
                f"\n[{index}/{total}] 저장 완료: "
                f"{date} / {activity.stroke_count} shots"
            )

        except requests.HTTPError as exc:
            failed += 1
            status = (
                exc.response.status_code
                if exc.response is not None
                else None
            )
            error_text = f"HTTP {status or '?'}"

            failed_items.append(
                {
                    "reportUuid": report_uuid,
                    "date": date,
                    "time": activity.time,
                    "strokeCount": activity.stroke_count,
                    "error": error_text,
                }
            )

            print(
                f"\n[{index}/{total}] 실패: "
                f"{date} / {report_uuid} / {error_text}"
            )

            if status in (401, 403):
                save_json(failed_path, failed_items)
                print(
                    "인증 토큰 또는 쿠키가 만료됐습니다. "
                    "Chrome에서 cURL 파일을 다시 복사하세요."
                )
                break

        except Exception as exc:
            failed += 1
            failed_items.append(
                {
                    "reportUuid": report_uuid,
                    "date": date,
                    "time": activity.time,
                    "strokeCount": activity.stroke_count,
                    "error": str(exc),
                }
            )
            print(
                f"\n[{index}/{total}] 실패: "
                f"{date} / {report_uuid} / {exc}"
            )

        save_json(failed_path, failed_items)
        print_progress(index, total, completed, skipped, failed)
        time.sleep(max(args.delay, 0))

    save_json(failed_path, failed_items)

    log_payload = {
        "finishedAt": datetime.now().isoformat(timespec="seconds"),
        "totalActivities": server_total,
        "shotAnalysisCount": len(index_payload),
        "attempted": total,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "output": str(output_dir.resolve()),
    }
    save_json(output_dir / "_last_run.json", log_payload)

    print("\n다운로드 작업 종료")
    print(f"- 새로 저장: {completed}개")
    print(f"- 기존 항목 건너뜀: {skipped}개")
    print(f"- 실패: {failed}개")
    print(f"- 저장 위치: {output_dir.resolve()}")

    if failed:
        print(
            "- 실패 항목 재시도: "
            "python download_all_trackman_reports_v3.py --retry-failed"
        )

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됐습니다.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n오류: {exc}", file=sys.stderr)
        raise SystemExit(1)
