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


def sync_trackman_reports(
    project_dir: str | Path | None = None,
    *,
    storage: TrackmanStorage | None = None,
    page_size: int = 50,
    delay: float = 0.6,
    timeout: int = 900,
) -> SyncResult:
    """신규 보고서를 내려받고 로컬 저장 후 Supabase Storage에 자동 백업합니다."""
    root = (Path(project_dir) if project_dir else Path(__file__).resolve().parent).resolve()
    storage = storage or TrackmanStorage(root)
    report_dir = storage.report_dir

    list_curl = root / "activity_list.curl"
    report_curl = root / "activity_report.curl"
    downloader = root / "download_all_trackman_reports.py"
    missing = [p.name for p in (list_curl, report_curl, downloader) if not p.exists()]
    if missing:
        return SyncResult(False, 2, "", "필수 파일이 없습니다: " + ", ".join(missing), 0, 0)

    before_files = {p.name for p in storage.report_files()}
    command = [
        sys.executable, str(downloader),
        "--list-curl", str(list_curl),
        "--report-curl", str(report_curl),
        "--output", str(report_dir),
        "--page-size", str(page_size),
        "--delay", str(delay),
    ]

    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
        after_files = {p.name for p in storage.report_files()}
        new_paths = [storage.report_dir / name for name in sorted(after_files - before_files)]
        cloud_result = None
        if completed.returncode == 0 and storage.cloud_configured:
            cloud_result = storage.upload_local_reports(new_paths if new_paths else None)
        if completed.returncode == 0:
            storage.invalidate_cache()
            storage.write_last_sync(source="trackman_downloader", details={
                "downloaded": len(after_files - before_files),
                "cloud_uploaded": cloud_result.uploaded if cloud_result else 0,
            })
        cloud_failed = bool(cloud_result and cloud_result.errors)
        return SyncResult(
            ok=completed.returncode == 0 and not cloud_failed,
            return_code=completed.returncode if not cloud_failed else 3,
            stdout=completed.stdout,
            stderr=(completed.stderr + ("\n" + "\n".join(cloud_result.errors) if cloud_failed else "")).strip(),
            before_count=len(before_files),
            after_count=len(after_files),
            cloud=cloud_result,
        )
    except subprocess.TimeoutExpired as exc:
        after_count = len(storage.report_files())
        return SyncResult(False, 124, exc.stdout or "", f"동기화 제한 시간({timeout}초)을 초과했습니다.", len(before_files), after_count)
