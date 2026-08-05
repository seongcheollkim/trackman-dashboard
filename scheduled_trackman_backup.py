from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from trackman_sync_auto_auth import SyncResult, sync_trackman_reports


DEFAULT_PROJECT_DIR = Path(
    "/Users/justin/Desktop/python/training_golf/trackman_dashboard_project"
)


def log(message: str) -> None:
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{stamp}] {message}", flush=True)


def load_streamlit_secrets(project_dir: Path) -> None:
    """launchd 환경에서도 .streamlit/secrets.toml 값을 환경변수로 전달합니다."""
    secrets_path = project_dir / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        log(f"secrets.toml 없음: {secrets_path}")
        return

    try:
        import tomllib
    except ImportError as exc:
        raise RuntimeError("Python 3.11 이상이 필요합니다.") from exc

    with secrets_path.open("rb") as fp:
        payload = tomllib.load(fp)

    for key in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_BUCKET"):
        value = payload.get(key)
        if value is not None and not os.getenv(key):
            os.environ[key] = str(value)

    missing = [
        key
        for key in ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_BUCKET")
        if not os.getenv(key)
    ]
    if missing:
        raise RuntimeError(
            "Supabase 설정 누락: " + ", ".join(missing)
        )


def run_once(
    project_dir: Path,
    *,
    page_size: int,
    delay: float,
    timeout: int,
    auth_timeout: int,
) -> SyncResult:
    return sync_trackman_reports(
        project_dir=project_dir,
        page_size=page_size,
        delay=delay,
        timeout=timeout,
        auth_timeout=auth_timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="DODOS Golf Solution 예약 백업 실행기"
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=int, default=60)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--auth-timeout", type=int, default=180)
    args = parser.parse_args()

    project_dir = args.project_dir.expanduser().resolve()
    if not project_dir.exists():
        log(f"프로젝트 폴더 없음: {project_dir}")
        return 2

    os.chdir(project_dir)
    load_streamlit_secrets(project_dir)

    log("DODOS Golf Solution 자동 백업 시작")
    log(f"프로젝트: {project_dir}")

    total_attempts = max(1, args.retries + 1)
    last_result: SyncResult | None = None

    for attempt in range(1, total_attempts + 1):
        log(f"동기화 시도 {attempt}/{total_attempts}")
        last_result = run_once(
            project_dir,
            page_size=args.page_size,
            delay=args.delay,
            timeout=args.timeout,
            auth_timeout=args.auth_timeout,
        )

        if last_result.stdout:
            print(last_result.stdout, flush=True)
        if last_result.stderr:
            print(last_result.stderr, file=sys.stderr, flush=True)

        if last_result.ok:
            cloud = last_result.cloud
            log(
                "자동 백업 성공 | "
                f"신규 다운로드 {last_result.downloaded_count}개 | "
                f"Supabase 업로드 {cloud.uploaded if cloud else 0}개 | "
                f"건너뜀 {cloud.skipped if cloud else 0}개"
            )
            return 0

        log(f"자동 백업 실패 | 종료 코드 {last_result.return_code}")
        if attempt < total_attempts:
            log(f"{args.retry_delay}초 후 재시도")
            time.sleep(args.retry_delay)

    log("모든 재시도 실패")
    return last_result.return_code if last_result else 1


if __name__ == "__main__":
    raise SystemExit(main())
