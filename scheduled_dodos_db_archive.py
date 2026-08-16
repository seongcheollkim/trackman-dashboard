from __future__ import annotations

"""
기존 scheduled_trackman_backup.py를 당장 교체하지 않고,
기존 자동 백업 성공 후 이 파일을 추가 실행하는 안전한 전환용 스크립트입니다.

동작:
  local data/trackman_reports/*.json
    -> dodos_* DB에 미적재 파일만 자동 archive

몇 번 실행해도 검증 완료 파일은 SKIP됩니다.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dodos_supabase import DodosSupabaseConfig
from trackman_db_archive import DodosTrackmanArchive


PROJECT_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_DIR / "data" / "trackman_reports"
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "dodos_db_archive.log"


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    line = f"[{stamp}] {message}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    try:
        config = DodosSupabaseConfig.load()
        if not config.user_email:
            raise RuntimeError(
                "DODOS_USER_EMAIL이 없습니다. "
                "LaunchAgent 환경 또는 .streamlit/secrets.toml에 설정하세요."
            )

        archive = DodosTrackmanArchive(
            config=config,
            user_email=config.user_email,
        )

        files = sorted(
            p for p in REPORT_DIR.glob("*.json")
            if p.is_file() and not p.name.startswith("_")
        )
        log(f"DB archive 시작 | JSON {len(files)}개")

        archived = 0
        skipped = 0
        failed = 0
        shots = 0

        for path in files:
            result = archive.archive_file(path)

            if result.ok:
                shots += result.archived_shots
                if result.skipped:
                    skipped += 1
                else:
                    archived += 1
                    log(
                        f"신규 DB 저장 | {result.practice_date} | "
                        f"{result.file_name} | {result.archived_shots}샷"
                    )
            else:
                failed += 1
                log(f"DB 저장 실패 | {result.file_name} | {result.error}")

        log(
            "DB archive 종료 | "
            f"신규 {archived} | 건너뜀 {skipped} | 실패 {failed} | "
            f"검증샷 {shots}"
        )

        return 1 if failed else 0

    except Exception as exc:
        log(f"DB archive 치명적 오류 | {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
