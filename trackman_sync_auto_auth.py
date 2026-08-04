from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from trackman_storage import CloudSyncResult, TrackmanStorage


@dataclass(frozen=True)
class SyncResult:
    ok: bool
    return_code: int
    stdout: str
    stderr: str
    before_count: int
    after_count: int
    cloud: CloudSyncResult | None = None

    @property
    def downloaded_count(self) -> int:
        return max(0, self.after_count - self.before_count)


def _run_command(
    command: list[str],
    *,
    root: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def sync_trackman_reports(
    project_dir: str | Path | None = None,
    *,
    storage: TrackmanStorage | None = None,
    page_size: int = 50,
    delay: float = 0.6,
    timeout: int = 900,
    auth_timeout: int = 180,
) -> SyncResult:
    """TrackMan 인증 갱신 → 신규 보고서 다운로드 → Supabase 백업."""
    root = (Path(project_dir) if project_dir else Path(__file__).resolve().parent).resolve()
    storage = storage or TrackmanStorage(root)
    report_dir = storage.report_dir

    list_curl = root / "activity_list.curl"
    report_curl = root / "activity_report.curl"
    downloader = root / "download_all_trackman_reports.py"
    auth_refresher = root / "trackman_auth_refresh.py"
    browser_profile = root / ".trackman_browser"

    required = (list_curl, report_curl, downloader, auth_refresher)
    missing = [path.name for path in required if not path.exists()]
    if missing:
        return SyncResult(
            ok=False,
            return_code=2,
            stdout="",
            stderr="필수 파일이 없습니다: " + ", ".join(missing),
            before_count=0,
            after_count=0,
        )

    before_files = {path.name for path in storage.report_files()}

    auth_command = [
        sys.executable,
        str(auth_refresher),
        "--list-curl",
        str(list_curl),
        "--report-curl",
        str(report_curl),
        "--profile-dir",
        str(browser_profile),
        "--timeout",
        str(auth_timeout),
    ]

    download_command = [
        sys.executable,
        str(downloader),
        "--list-curl",
        str(list_curl),
        "--report-curl",
        str(report_curl),
        "--output",
        str(report_dir),
        "--page-size",
        str(page_size),
        "--delay",
        str(delay),
    ]

    try:
        auth_result = _run_command(
            auth_command,
            root=root,
            timeout=auth_timeout + 30,
        )
        if auth_result.returncode != 0:
            return SyncResult(
                ok=False,
                return_code=auth_result.returncode,
                stdout=auth_result.stdout,
                stderr=auth_result.stderr,
                before_count=len(before_files),
                after_count=len(before_files),
            )

        completed = _run_command(
            download_command,
            root=root,
            timeout=timeout,
        )

        after_files = {path.name for path in storage.report_files()}
        new_paths = [
            storage.report_dir / name
            for name in sorted(after_files - before_files)
        ]

        cloud_result: CloudSyncResult | None = None
        if completed.returncode == 0:
            if storage.cloud_configured:
                # 신규 파일이 있으면 신규 파일만, 없으면 누락된 로컬 파일 전체를 점검합니다.
                cloud_result = storage.upload_local_reports(
                    new_paths if new_paths else None
                )
            else:
                cloud_result = CloudSyncResult(
                    errors=("Supabase 설정이 없습니다.",)
                )

        cloud_failed = bool(cloud_result and cloud_result.errors)

        if completed.returncode == 0 and not cloud_failed:
            storage.invalidate_cache()
            storage.write_last_sync(
                source="scheduled_trackman_backup",
                details={
                    "downloaded": len(new_paths),
                    "cloud_uploaded": cloud_result.uploaded if cloud_result else 0,
                    "cloud_skipped": cloud_result.skipped if cloud_result else 0,
                    "auth_refreshed": True,
                },
            )

        combined_stdout = "\n".join(
            part.strip()
            for part in (auth_result.stdout, completed.stdout)
            if part and part.strip()
        )
        combined_stderr = "\n".join(
            part.strip()
            for part in (
                auth_result.stderr,
                completed.stderr,
                "\n".join(cloud_result.errors) if cloud_failed and cloud_result else "",
            )
            if part and part.strip()
        )

        return SyncResult(
            ok=completed.returncode == 0 and not cloud_failed,
            return_code=completed.returncode if not cloud_failed else 3,
            stdout=combined_stdout,
            stderr=combined_stderr,
            before_count=len(before_files),
            after_count=len(after_files),
            cloud=cloud_result,
        )

    except subprocess.TimeoutExpired as exc:
        after_count = len(storage.report_files())
        captured_stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        captured_stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return SyncResult(
            ok=False,
            return_code=124,
            stdout=captured_stdout,
            stderr=(
                captured_stderr
                + f"\n동기화 제한 시간({timeout}초)을 초과했습니다."
            ).strip(),
            before_count=len(before_files),
            after_count=after_count,
        )
    except Exception as exc:
        return SyncResult(
            ok=False,
            return_code=1,
            stdout="",
            stderr=f"예상하지 못한 동기화 오류: {exc}",
            before_count=len(before_files),
            after_count=len(storage.report_files()),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DODOS Golf Solution TrackMan 자동 동기화"
    )
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.6)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--auth-timeout", type=int, default=180)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = sync_trackman_reports(
        project_dir=args.project_dir,
        page_size=args.page_size,
        delay=args.delay,
        timeout=args.timeout,
        auth_timeout=args.auth_timeout,
    )

    print(
        f"[결과] ok={result.ok}, "
        f"downloaded={result.downloaded_count}, "
        f"local={result.after_count}"
    )
    if result.cloud is not None:
        print(
            f"[Supabase] uploaded={result.cloud.uploaded}, "
            f"skipped={result.cloud.skipped}, "
            f"errors={len(result.cloud.errors)}"
        )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return 0 if result.ok else result.return_code or 1


if __name__ == "__main__":
    raise SystemExit(main())
