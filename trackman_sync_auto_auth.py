from __future__ import annotations

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
    """TrackMan 인증을 자동 갱신한 뒤 신규 보고서를 내려받고 Supabase에 백업합니다."""
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
            False,
            2,
            "",
            "필수 파일이 없습니다: " + ", ".join(missing),
            0,
            0,
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

        cloud_result = None
        if completed.returncode == 0 and storage.cloud_configured:
            cloud_result = storage.upload_local_reports(
                new_paths if new_paths else None
            )

        if completed.returncode == 0:
            storage.invalidate_cache()
            storage.write_last_sync(
                source="trackman_downloader",
                details={
                    "downloaded": len(after_files - before_files),
                    "cloud_uploaded": cloud_result.uploaded if cloud_result else 0,
                    "auth_refreshed": True,
                },
            )

        cloud_failed = bool(cloud_result and cloud_result.errors)
        combined_stdout = "\n".join(
            part for part in (auth_result.stdout, completed.stdout) if part
        )

        return SyncResult(
            ok=completed.returncode == 0 and not cloud_failed,
            return_code=completed.returncode if not cloud_failed else 3,
            stdout=combined_stdout,
            stderr=(
                completed.stderr
                + (
                    "\n" + "\n".join(cloud_result.errors)
                    if cloud_failed
                    else ""
                )
            ).strip(),
            before_count=len(before_files),
            after_count=len(after_files),
            cloud=cloud_result,
        )

    except subprocess.TimeoutExpired as exc:
        after_count = len(storage.report_files())
        return SyncResult(
            False,
            124,
            exc.stdout or "",
            f"동기화 제한 시간({timeout}초)을 초과했습니다.",
            len(before_files),
            after_count,
        )
