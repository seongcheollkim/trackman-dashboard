from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from trackman_storage import TrackmanStorage
from trackman_sync import sync_trackman_reports


PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"
LOCK_FILE = PROJECT_DIR / ".trackman_sync.lock"
RESULT_FILE = LOG_DIR / "last_scheduled_sync.json"


def write_result(payload: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def acquire_lock() -> bool:
    try:
        with LOCK_FILE.open("x", encoding="utf-8") as file:
            file.write(datetime.now().isoformat())
        return True
    except FileExistsError:
        return False


def release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def main() -> int:
    started_at = datetime.now()

    if not acquire_lock():
        write_result({
            "ok": False,
            "skipped": True,
            "reason": "이미 TrackMan 동기화가 실행 중입니다.",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
        })
        print("이미 TrackMan 동기화가 실행 중입니다.")
        return 0

    try:
        storage = TrackmanStorage(PROJECT_DIR)
        result = sync_trackman_reports(
            project_dir=PROJECT_DIR,
            storage=storage,
            page_size=50,
            delay=0.6,
            timeout=900,
            auth_timeout=300,
        )

        payload = {
            "ok": result.ok,
            "return_code": result.return_code,
            "downloaded_count": result.downloaded_count,
            "before_count": result.before_count,
            "after_count": result.after_count,
            "cloud_uploaded": result.cloud.uploaded if result.cloud else 0,
            "cloud_errors": list(result.cloud.errors) if result.cloud and result.cloud.errors else [],
            "stdout_tail": (result.stdout or "")[-3000:],
            "stderr_tail": (result.stderr or "")[-3000:],
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
        }
        write_result(payload)

        if result.ok:
            print(
                f"TrackMan 예약 동기화 완료: 신규 {result.downloaded_count}개, "
                f"Supabase 업로드 {payload['cloud_uploaded']}개"
            )
            return 0

        print(result.stderr or result.stdout or "알 수 없는 오류", file=sys.stderr)
        return result.return_code or 1

    except Exception as exc:
        write_result({
            "ok": False,
            "return_code": 1,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
        })
        traceback.print_exc()
        return 1

    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
