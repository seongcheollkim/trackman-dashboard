#!/usr/bin/env python3
"""
DODOS Golf Solution - Scheduled TrackMan -> DB -> AI pipeline v2

기존 trackman_backup_wrapper.sh가 넘기는 CLI 인자를 그대로 호환합니다.

기존 흐름:
    wrapper
      -> scheduled_trackman_backup.py
         -> TrackMan 다운로드 + Supabase Storage

변경 흐름:
    wrapper
      -> scheduled_dodos_pipeline.py
         -> scheduled_trackman_backup.py
         -> trackman_backfill.py --commit
         -> DODOS DB + AI snapshot

안전 원칙:
- 기존 scheduled_trackman_backup.py는 수정하지 않음
- TrackMan 수집 실패 시 DB 적재 실행 안 함
- DB 적재 실패 시 원본 JSON 삭제 안 함
- lock으로 중복 실행 방지
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def now_text() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def banner(text: str) -> None:
    print("=" * 70, flush=True)
    print(f"[{now_text()}] {text}", flush=True)
    print("=" * 70, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TrackMan 자동 수집 후 DODOS DB + AI 자동 동기화"
    )

    # 기존 wrapper / scheduled_trackman_backup.py 호환 인자
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parent),
        help="TrackMan Dashboard 프로젝트 루트",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="TrackMan 자동 수집 재시도 횟수",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=60,
        help="TrackMan 자동 수집 재시도 간격(초)",
    )

    # 수동 점검용
    parser.add_argument(
        "--db-only",
        action="store_true",
        help="TrackMan 수집은 건너뛰고 DB + AI 적재만 실행",
    )
    parser.add_argument(
        "--trackman-only",
        action="store_true",
        help="TrackMan 자동 수집만 실행",
    )

    return parser.parse_args()


def run_command(
    args: list[str],
    *,
    cwd: Path,
    label: str,
    timeout: int,
) -> int:
    print(f"[{now_text()}] {label} 시작", flush=True)
    print("  $ " + " ".join(args), flush=True)

    try:
        result = subprocess.run(
            args,
            cwd=str(cwd),
            env=os.environ.copy(),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[{now_text()}] ERROR: {label} timeout ({timeout}초)",
            flush=True,
        )
        return 124
    except Exception as exc:
        print(
            f"[{now_text()}] ERROR: {label} 실행 실패: {exc}",
            flush=True,
        )
        return 125

    print(
        f"[{now_text()}] {label} 종료 코드: {result.returncode}",
        flush=True,
    )
    return int(result.returncode)


def main() -> int:
    args = parse_args()

    if args.db_only and args.trackman_only:
        print("--db-only와 --trackman-only는 동시에 사용할 수 없습니다.")
        return 2

    project_dir = Path(args.project_dir).expanduser().resolve()
    python = project_dir / ".venv" / "bin" / "python"
    trackman_job = project_dir / "scheduled_trackman_backup.py"
    db_job = project_dir / "trackman_backfill.py"

    log_dir = project_dir / "logs"
    lock_file = log_dir / "dodos_pipeline.lock"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not python.exists():
        print(f"가상환경 Python을 찾을 수 없습니다: {python}")
        return 10

    with lock_file.open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"[{now_text()}] 다른 DODOS 파이프라인이 실행 중입니다. "
                "이번 실행은 건너뜁니다.",
                flush=True,
            )
            return 0

        banner("DODOS 자동 데이터 파이프라인 시작")

        # ----------------------------------------------------------
        # 1. 기존 TrackMan 자동 수집
        # ----------------------------------------------------------
        if not args.db_only:
            if not trackman_job.exists():
                print(f"예약 실행 파일을 찾을 수 없습니다: {trackman_job}")
                return 11

            trackman_cmd = [
                str(python),
                "-u",
                str(trackman_job),
                "--project-dir",
                str(project_dir),
                "--retries",
                str(args.retries),
                "--retry-delay",
                str(args.retry_delay),
            ]

            rc = run_command(
                trackman_cmd,
                cwd=project_dir,
                label="Phase 1 TrackMan 자동 수집",
                timeout=30 * 60,
            )

            if rc != 0:
                print(
                    f"[{now_text()}] TrackMan 자동 수집 실패. "
                    "DB + AI 적재는 실행하지 않습니다.",
                    flush=True,
                )
                return rc

            if args.trackman_only:
                banner("DODOS 자동 데이터 파이프라인 정상 종료")
                return 0

        # ----------------------------------------------------------
        # 2. JSON -> DODOS DB + AI
        # ----------------------------------------------------------
        if not db_job.exists():
            print(f"DB 적재 파일을 찾을 수 없습니다: {db_job}")
            return 12

        db_cmd = [
            str(python),
            "-u",
            str(db_job),
            "--commit",
        ]

        rc = run_command(
            db_cmd,
            cwd=project_dir,
            label="DODOS DB + AI 동기화",
            timeout=45 * 60,
        )

        if rc != 0:
            print(
                f"[{now_text()}] DB + AI 동기화 실패. "
                "TrackMan JSON은 그대로 보존됩니다.",
                flush=True,
            )
            return rc

        banner("DODOS 자동 데이터 파이프라인 정상 종료")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
