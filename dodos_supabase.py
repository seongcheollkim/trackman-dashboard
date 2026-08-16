from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from supabase import Client, create_client


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _read_toml_secrets() -> dict[str, Any]:
    """
    CLI에서도 Streamlit secrets를 사용할 수 있게
    .streamlit/secrets.toml을 직접 읽습니다.
    """
    candidates = [
        _project_root() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def setting(name: str, default: str = "") -> str:
    """
    우선순위:
      1) 환경변수
      2) .streamlit/secrets.toml
      3) default
    """
    value = os.getenv(name)
    if value:
        return str(value).strip()

    secrets = _read_toml_secrets()
    value = secrets.get(name, default)
    return str(value or default).strip()


@dataclass(frozen=True)
class DodosSupabaseConfig:
    url: str
    service_role_key: str
    bucket: str = "trackman-reports"
    cloud_prefix: str = "reports"
    user_email: str = ""

    @classmethod
    def load(cls) -> "DodosSupabaseConfig":
        url = setting("SUPABASE_URL")
        service_role_key = (
            setting("SUPABASE_SERVICE_ROLE_KEY")
            or setting("SUPABASE_SERVICE_KEY")
        )
        bucket = setting("SUPABASE_BUCKET", "trackman-reports")
        prefix = setting("SUPABASE_PREFIX", "reports")
        user_email = (
            setting("DODOS_USER_EMAIL")
            or setting("AUTH_EMAIL")
            or setting("USER_EMAIL")
        )

        missing = []
        if not url:
            missing.append("SUPABASE_URL")
        if not service_role_key:
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
        if missing:
            raise RuntimeError(
                "필수 Supabase 설정이 없습니다: " + ", ".join(missing)
            )

        return cls(
            url=url.rstrip("/"),
            service_role_key=service_role_key,
            bucket=bucket,
            cloud_prefix=prefix.strip("/") or "reports",
            user_email=user_email.lower().strip(),
        )


def create_dodos_client(config: DodosSupabaseConfig | None = None) -> Client:
    config = config or DodosSupabaseConfig.load()
    return create_client(config.url, config.service_role_key)


def get_dodos_user(
    client: Client,
    *,
    email: str,
) -> dict[str, Any]:
    email = email.lower().strip()
    if not email:
        raise RuntimeError(
            "사용자 이메일이 없습니다. "
            "DODOS_USER_EMAIL 환경변수를 설정하거나 --user-email을 사용하세요."
        )

    result = (
        client.table("dodos_users")
        .select("id,email,display_name")
        .eq("email", email)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise RuntimeError(
            f"dodos_users에서 {email} 사용자를 찾지 못했습니다. "
            "03_BOOTSTRAP_FIRST_USER.sql 실행 결과를 확인하세요."
        )
    return result.data[0]


def test_connection(
    client: Client,
    *,
    email: str,
) -> dict[str, Any]:
    user = get_dodos_user(client, email=email)

    tables = [
        "dodos_practice_sessions",
        "dodos_shots",
        "dodos_raw_files",
        "dodos_session_club_summary",
    ]
    counts: dict[str, int] = {}
    for table in tables:
        result = client.table(table).select("id", count="exact").limit(1).execute()
        counts[table] = int(result.count or 0)

    return {
        "user": user,
        "counts": counts,
    }
