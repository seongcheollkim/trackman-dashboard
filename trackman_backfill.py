from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dodos_supabase import (
    DodosSupabaseConfig,
    create_dodos_client,
    test_connection,
)
from trackman_db_archive import DodosTrackmanArchive


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT_DIR = PROJECT_DIR / "data" / "trackman_reports"


def _print_precheck(report_dir: Path, user_email: str) -> None:
    config = DodosSupabaseConfig.load()
    client = create_dodos_client(config)
    status = test_connection(client, email=user_email)

    files = sorted(
        p for p in report_dir.glob("*.json")
        if p.is_file() and not p.name.startswith("_")
    )

    print("=" * 64)
    print("DODOS DB BACKFILL PRECHECK")
    print("=" * 64)
    print(f"User       : {status['user']['email']}")
    print(f"Report dir : {report_dir}")
    print(f"JSON files : {len(files)}")
    print(f"Bucket     : {config.bucket}")
    print(f"Prefix     : {config.cloud_prefix}")
    print("")
    print("Current DB:")
    for table, count in status["counts"].items():
        print(f"  {table:<32} {count:,}")
    print("=" * 64)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="기존 TrackMan JSON을 DODOS Database v1.0으로 backfill"
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="TrackMan JSON 폴더",
    )
    parser.add_argument(
        "--user-email",
        default="",
        help="DODOS 사용자 이메일. 생략 시 DODOS_USER_EMAIL 사용",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="실제 DB 저장. 생략하면 precheck만 실행",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 검증 완료된 파일도 다시 upsert",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir).expanduser().resolve()
    config = DodosSupabaseConfig.load()
    user_email = (args.user_email or config.user_email).lower().strip()

    if not report_dir.exists():
        print(f"보고서 폴더가 없습니다: {report_dir}", file=sys.stderr)
        return 2

    if not user_email:
        print(
            "--user-email 또는 DODOS_USER_EMAIL 설정이 필요합니다.",
            file=sys.stderr,
        )
        return 2

    _print_precheck(report_dir, user_email)

    if not args.commit:
        print("")
        print("PRECHECK만 완료했습니다. DB에는 아무 것도 저장하지 않았습니다.")
        print("정상이면 같은 명령에 --commit을 추가하세요.")
        return 0

    archive = DodosTrackmanArchive(user_email=user_email)

    files = sorted(
        p for p in report_dir.glob("*.json")
        if p.is_file() and not p.name.startswith("_")
    )

    total = len(files)
    ok = 0
    skipped = 0
    failed = 0
    total_shots = 0

    print("")
    print("Backfill 시작")
    print("-" * 64)

    for index, path in enumerate(files, start=1):
        result = archive.archive_file(path, force=args.force)

        if result.ok:
            ok += 1
            total_shots += result.archived_shots
            if result.skipped:
                skipped += 1
                state = "SKIP"
            else:
                state = "OK"
            print(
                f"[{index:>3}/{total}] {state:<4} "
                f"{result.practice_date}  {result.file_name}  "
                f"{result.archived_shots}/{result.expected_shots} shots"
            )
        else:
            failed += 1
            print(
                f"[{index:>3}/{total}] FAIL {result.file_name}: "
                f"{result.error}"
            )

    print("-" * 64)
    print("Backfill 종료")
    print(f"  파일       : {total}")
    print(f"  성공/검증  : {ok}")
    print(f"  기존 건너뜀: {skipped}")
    print(f"  실패       : {failed}")
    print(f"  DB 샷      : {total_shots:,}")
    print("")

    if failed:
        print("실패 파일이 있으므로 다음 자동화 단계로 넘어가지 마세요.")
        return 1

    print("모든 JSON이 DB에 정상 검증되었습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
