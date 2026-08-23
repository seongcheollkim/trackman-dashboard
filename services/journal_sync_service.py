from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping


@dataclass(frozen=True)
class JournalDbSyncResult:
    ok: bool
    returncode: int
    message: str
    stdout: str = ""
    stderr: str = ""


def sync_journal_db_after_pull(
    *,
    project_dir: Path,
    user_email: str,
    secrets: Mapping[str, str],
    timeout_seconds: int = 900,
) -> JournalDbSyncResult:
    """
    Supabase Storage에서 TrackMan JSON을 내려받은 직후
    DODOS DB / AI snapshot을 동기화합니다.

    기존 운영 로직(trackman_backfill.py --commit)을 그대로 재사용하므로
    이미 DB에 적재된 세션은 SKIP되고 신규 세션만 추가됩니다.

    `sys.executable`을 사용하므로 로컬 .venv와 Streamlit Cloud에서
    현재 앱이 실행 중인 동일 Python 환경을 사용합니다.
    """
    project_dir = Path(project_dir).resolve()
    backfill_script = project_dir / "trackman_backfill.py"

    if not backfill_script.exists():
        return JournalDbSyncResult(
            ok=False,
            returncode=2,
            message=f"DB 동기화 파일을 찾을 수 없습니다: {backfill_script}",
        )

    env = os.environ.copy()

    # CLI로 실행되는 backfill이 Streamlit 런타임 밖에서도
    # 동일한 Supabase/DODOS 설정을 읽도록 명시적으로 전달합니다.
    for name in (
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_BUCKET",
        "DODOS_USER_EMAIL",
    ):
        value = str(secrets.get(name, "") or "").strip()
        if value:
            env[name] = value

    if user_email:
        env["DODOS_USER_EMAIL"] = str(user_email).strip()

    missing = [
        name
        for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "DODOS_USER_EMAIL")
        if not env.get(name)
    ]
    if missing:
        return JournalDbSyncResult(
            ok=False,
            returncode=3,
            message="DB 동기화 필수 설정 누락: " + ", ".join(missing),
        )

    cmd = [
        sys.executable,
        "-u",
        str(backfill_script),
        "--commit",
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(project_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return JournalDbSyncResult(
            ok=False,
            returncode=124,
            message=f"DB/AI 동기화가 {timeout_seconds}초를 초과했습니다.",
            stdout=str(exc.stdout or ""),
            stderr=str(exc.stderr or ""),
        )
    except Exception as exc:
        return JournalDbSyncResult(
            ok=False,
            returncode=125,
            message=f"DB/AI 동기화 실행 실패: {exc}",
        )

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode != 0:
        return JournalDbSyncResult(
            ok=False,
            returncode=int(completed.returncode),
            message=f"DB/AI 동기화 실패 (exit={completed.returncode})",
            stdout=stdout,
            stderr=stderr,
        )

    return JournalDbSyncResult(
        ok=True,
        returncode=0,
        message="DODOS DB / AI / 연습일지 동기화 완료",
        stdout=stdout,
        stderr=stderr,
    )
