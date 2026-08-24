from __future__ import annotations

from typing import Any

import streamlit as st

from ui.round_record import render_round_record


st.set_page_config(
    page_title="DODOS · 라운드 기록",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _allowed_emails() -> set[str]:
    try:
        raw = st.secrets.get("ALLOWED_EMAILS", [])
    except Exception:
        raw = []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    return {str(email).strip().lower() for email in raw if str(email).strip()}


def _require_user() -> str:
    try:
        if not st.user.is_logged_in:
            st.info("Google 로그인 후 라운드 기록을 사용할 수 있습니다.")
            st.button("Google 계정으로 로그인", type="primary", on_click=st.login)
            st.stop()
    except Exception:
        st.error("Google 로그인 설정을 읽지 못했습니다.")
        st.stop()

    user: dict[str, Any] = st.user.to_dict()
    email = str(user.get("email", "")).strip().lower()
    allowed = _allowed_emails()
    if not email or email not in allowed:
        st.error("이 Google 계정은 앱 사용 권한이 없습니다.")
        st.stop()
    return email


AUTH_EMAIL = _require_user()
render_round_record(user_email=AUTH_EMAIL)
