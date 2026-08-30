from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import matplotlib

# Streamlit 서버에서는 GUI 백엔드가 필요하지 않습니다.
# macOS에서 발생할 수 있는 Matplotlib 관련 충돌을 방지합니다.
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Shot list: AG Grid dark interactive table
from st_aggrid import AgGrid, GridOptionsBuilder

_FONT_CONFIGURED = False

def configure_matplotlib_korean_font() -> None:
    """로컬 macOS와 Streamlit Cloud에서 한글 폰트를 한 번만 적용합니다."""
    global _FONT_CONFIGURED

    if _FONT_CONFIGURED:
        return

    preferred_fonts = [
        "NanumGothic",          # Streamlit Cloud: packages.txt로 설치
        "Apple SD Gothic Neo",  # macOS
        "AppleGothic",          # macOS
        "Arial Unicode MS",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "Malgun Gothic",        # Windows
    ]

    installed_fonts = {
        font.name for font in fm.fontManager.ttflist
    }

    selected_font = next(
        (name for name in preferred_fonts if name in installed_fonts),
        None,
    )

    if selected_font is None:
        print("⚠️ Matplotlib 한글 폰트를 찾지 못했습니다.")
        print(
            sorted(
                name
                for name in installed_fonts
                if any(
                    keyword in name.lower()
                    for keyword in ["nanum", "gothic", "noto", "malgun"]
                )
            )
        )
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        selected_font,
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    _FONT_CONFIGURED = True


configure_matplotlib_korean_font()

def safe_dataframe_for_streamlit(source_df: pd.DataFrame) -> pd.DataFrame:
    """Streamlit의 PyArrow 변환 전에 혼합 자료형을 안전하게 정리합니다."""
    if source_df is None:
        return pd.DataFrame()

    result = source_df.copy()

    def normalize_value(value: Any) -> Any:
        if value is None:
            return None

        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist(), ensure_ascii=False)

        if isinstance(value, (dict, list, tuple, set)):
            serializable = list(value) if isinstance(value, set) else value
            try:
                return json.dumps(serializable, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(value)

        if isinstance(value, (pd.Timestamp, datetime, date)):
            if pd.isna(value):
                return None
            return value.strftime("%Y-%m-%d %H:%M:%S")

        if isinstance(value, time):
            return value.strftime("%H:%M:%S")

        return value

    for column in result.columns:
        series = result[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            result[column] = series.dt.strftime("%Y-%m-%d %H:%M:%S")
            result[column] = result[column].replace("NaT", None)
            continue

        if series.dtype == "object":
            result[column] = series.map(normalize_value)
            non_null_values = result[column].dropna()
            if not non_null_values.empty:
                value_types = non_null_values.map(type).unique()
                if len(value_types) > 1:
                    result[column] = result[column].map(
                        lambda value: None if value is None else str(value)
                    )

    return result


from trackman_storage import TrackmanStorage
from trackman_sync import sync_trackman_reports

from trackman_core import (
    OUTPUT_COLUMNS,
    SUMMARY_COLUMNS,
    club_sort_key,
    make_summary,
    parse_trackman_report,
)

from ai import AI_CSS

from ui.common import (
    classify_face_to_path,
    classify_path,
    fmt,
    fmt_int,
    mean_existing_column,
    render_top_metrics,
    side_text,
)
from ui.ai_summary import render_ai_summary
from ui.ai_diagnosis import render_ai_club_detail
from ui.shot_analysis import render_single_shot_analysis
from ui.average_analysis import render_average_analysis
from charts.club_visuals import club_path_fig, impact_face_fig, loft_spin_fig
from ui.practice_journal import render_practice_journal_tab
from ui.round_record import render_round_record
from services.journal_sync_service import sync_journal_db_after_pull

st.set_page_config(
    page_title="TRACKMAN DASHBOARD",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Google OIDC 로그인 + 허용 이메일 제한
# -----------------------------------------------------------------------------
def _allowed_emails() -> set[str]:
    """Secrets의 ALLOWED_EMAILS를 소문자 이메일 집합으로 변환합니다."""
    try:
        raw = st.secrets.get("ALLOWED_EMAILS", [])
    except Exception:
        raw = []

    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]

    return {
        str(email).strip().lower()
        for email in raw
        if str(email).strip()
    }


def _render_login_screen() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { display: none; }
        .tm-login-wrap {
            max-width: 520px;
            margin: 12vh auto 0 auto;
            padding: 38px 40px;
            border: 1px solid #263548;
            border-radius: 18px;
            background: linear-gradient(180deg, #121f2d, #0b141e);
            text-align: center;
            box-shadow: 0 18px 60px rgba(0, 0, 0, .28);
        }
        .tm-login-logo {
            color: #f3f7fb;
            font-size: 1.45rem;
            font-weight: 850;
            letter-spacing: .02em;
            margin-bottom: 18px;
        }
        .tm-login-orange { color: #ff6b35; }
        .tm-login-title {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .tm-login-sub {
            color: #aab7c7;
            line-height: 1.7;
            margin-bottom: 22px;
        }
        
.tm-shot-section-title{
  font-size:1.03rem;
  font-weight:850;
  color:#f3f7fb;
  margin:12px 0 8px;
  padding-top:4px;
}


/* DODOS unified dark UI - tables and Streamlit controls */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  background: #0d1722 !important;
  color: #ffffff !important;
}
[data-testid="stDataFrame"] *, [data-testid="stDataEditor"] * {
  color: #ffffff !important;
}
[data-testid="stDataFrame"] table,
[data-testid="stDataFrame"] thead,
[data-testid="stDataFrame"] tbody,
[data-testid="stDataFrame"] tr,
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
  background-color: #0d1722 !important;
  color: #ffffff !important;
  border-color: #263548 !important;
}
[data-baseweb="select"] *,
[data-baseweb="input"] *,
[data-baseweb="textarea"] * {
  color: #ffffff !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label,
.stNumberInput label, .stDateInput label, .stMultiSelect label,
.stRadio label, .stCheckbox label, .stSlider label,
.stFileUploader label, .stButton button, .stMarkdown,
.stCaption, [data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
  color: #ffffff !important;
}

</style>
        <div class="tm-login-wrap">
          <div class="tm-login-logo"><span class="tm-login-orange">▰</span> TRACKMAN DASHBOARD</div>
          <div class="tm-login-title">개인 전용 대시보드</div>
          <div class="tm-login-sub">
            등록된 Google 계정으로 로그인해야<br>
            TrackMan 연습 데이터를 확인할 수 있습니다.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button(
        "Google 계정으로 로그인",
        type="primary",
        use_container_width=True,
        on_click=st.login,
    )


def require_authorized_google_user() -> dict[str, Any]:
    """로그인 및 이메일 화이트리스트 검사를 통과한 사용자 정보만 반환합니다."""
    try:
        is_logged_in = bool(st.user.is_logged_in)
    except Exception:
        st.error("Google 로그인 설정을 읽지 못했습니다. Streamlit Secrets의 [auth] 설정을 확인해 주세요.")
        st.stop()

    if not is_logged_in:
        _render_login_screen()
        st.stop()

    user = st.user.to_dict()
    email = str(user.get("email", "")).strip().lower()
    allowed = _allowed_emails()

    if not allowed:
        st.error("ALLOWED_EMAILS가 설정되지 않아 접근을 차단했습니다.")
        st.caption("Streamlit Secrets에 ALLOWED_EMAILS = [\"your-email@gmail.com\"] 형식으로 추가해 주세요.")
        st.button("로그아웃", on_click=st.logout)
        st.stop()

    if not email or email not in allowed:
        st.error("이 Google 계정은 앱 사용 권한이 없습니다.")
        if email:
            st.caption(f"로그인 계정: {email}")
        st.button("다른 계정으로 로그인", on_click=st.logout)
        st.stop()

    return user


AUTH_USER = require_authorized_google_user()
AUTH_EMAIL = str(AUTH_USER.get("email", ""))
AUTH_NAME = str(AUTH_USER.get("name", "사용자"))
CSS = """
<style>

.tm-kpi-grid {
  display: grid;
  grid-template-columns: repeat(9, minmax(125px, 1fr));
  gap: 14px;
  width: 100%;
  margin: 12px 0 28px 0;
}

.tm-kpi-card {
  min-width: 0;
  min-height: 108px;
  border: 1px solid #263548;
  border-radius: 10px;
  padding: 15px 16px;
  background: linear-gradient(180deg, #121f2d, #0d1721);
  box-sizing: border-box;
}

.tm-kpi-label {
  color: #b8c7d8;
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 12px;
  white-space: nowrap;
}

.tm-kpi-value {
  color: #f3f8ff;
  font-size: 28px;
  font-weight: 800;
  line-height: 1.1;
  white-space: nowrap;
}

.tm-kpi-unit {
  color: #c2cfdd;
  font-size: 14px;
  font-weight: 600;
  margin-left: 4px;
}

@media (max-width: 1450px) {
  .tm-kpi-grid {
    grid-template-columns: repeat(5, minmax(135px, 1fr));
  }
}

@media (max-width: 900px) {
  .tm-kpi-grid {
    grid-template-columns: repeat(3, minmax(130px, 1fr));
  }

  .block-container {
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
  }
}
:root {
  --bg: #07101a; --panel:#101b27; --panel2:#0d1722; --line:#243244;
  --accent:#ff6b1a; --text:#eef4fb; --muted:#9aaabd; --blue:#3d94ff; --red:#ff4b3e;
}
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at 40% 0%, #102033 0%, #07101a 42%, #050a10 100%); }
[data-testid="stSidebar"] { background: #0b1520; border-right:1px solid #1e2c3c; }
.block-container {
  padding-top: 1.5rem !important;
  padding-bottom: 1.5rem !important;
  padding-left: 3rem !important;
  padding-right: 3rem !important;
  max-width: 1800px !important;
}

/* Streamlit 기본 상단 헤더 완전히 숨김 */
header[data-testid="stHeader"] {
  display: block !important;
  visibility: visible !important;
  background: transparent !important;
}

[data-testid="stSidebarCollapsedControl"] {
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  pointer-events: auto !important;
  z-index: 999999 !important;
}

[data-testid="stDecoration"] {
  display: none !important;
}

#MainMenu, footer {
  visibility: hidden;
}

/* 사이드바도 위쪽 여백 확보 */
section[data-testid="stSidebar"] > div {
  padding-top: 1.5rem !important;
}

hr { border-color:#223044; margin: 1.2rem 0; }
.tm-topbar {
  display:flex;
  align-items:center;
  flex-wrap:wrap;
  gap:14px;
  min-height:48px;
  border-bottom:1px solid #1e2c3c;
  padding:4px 0 12px 0;
  margin:0 0 14px 0;
  position:relative;
  z-index:10;
  overflow:visible;
}
.tm-logo {font-weight:800; font-size:1.25rem; letter-spacing:.5px; color:#f5f7fb;}
.tm-orange {color:#ff6b1a;}
.tm-pill { display:inline-flex; align-items:center; gap:6px; padding: 7px 12px; border-radius: 9px; background:#111c29; border:1px solid #263548; color:#dce8f5; font-size:0.88rem; margin-right:6px; }
.tm-pill.active { border-color:#ff6b1a; color:#fff; box-shadow: inset 0 -2px 0 #ff6b1a; }
.tm-title {font-size:1.55rem; font-weight:800; margin: 6px 0 12px 0; color:#f7fbff;}
.tm-card { border:1px solid #263548; border-radius:12px; padding:14px 16px; background:linear-gradient(180deg,#111e2b,#0d1721); min-height:136px; }
.tm-card h4 { margin:0 0 10px 0; font-size:1.02rem; }
.tm-card .big { font-size:1.55rem; font-weight:800; color:#f4f7fb; line-height:1.1; }
.tm-card .blue { color:#4aa3ff; font-weight:800; }
.tm-card .small { color:#aab7c7; font-size:.82rem; line-height:1.55; }
.tm-panel { border:1px solid #263548; border-radius:12px; padding:14px 16px; background:linear-gradient(180deg,#101b27,#0c151f); height:100%; }
.tm-panel-title {font-weight:800; color:#f5f8fd; font-size:1.05rem; margin-bottom:8px;}
.tm-muted {color:#9aaabd; font-size:.87rem;}
.tm-shot-card { border:1px solid #263548; border-radius:12px; background:linear-gradient(180deg,#111e2b,#0d1721); padding:15px 18px; }
.tm-shot-detail-shell {border:1px solid #263548;border-radius:12px;background:linear-gradient(180deg,#111e2b,#0d1721);padding:10px 12px;margin:6px 0 8px;}
.tm-shot-detail-shell .tm-shot-card {border:0;background:transparent;padding:4px 0;}
.tm-shot-grid {
  width:100%;
  background:transparent;
}
.tm-shot-row {
  display:grid;
  grid-template-columns:repeat(9, minmax(108px, 1fr));
  align-items:stretch;
  width:100%;
}
.tm-shot-row + .tm-shot-row { margin-top:-1px; }
.tm-shot-row .tm-shot-item {
  border:1px solid #3a4f67 !important;
  margin-right:-1px;
  background:linear-gradient(180deg,#111e2b,#0d1721);
}
.tm-shot-row:first-child .tm-shot-item:first-child { border-top-left-radius:9px; }
.tm-shot-row:first-child .tm-shot-item:last-child { border-top-right-radius:9px; }
.tm-shot-row:last-child .tm-shot-item:first-child { border-bottom-left-radius:9px; }
.tm-shot-row:last-child .tm-shot-item:last-child { border-bottom-right-radius:9px; }
.tm-shot-club-only {
  display:flex;
  align-items:center;
  justify-content:center;
  min-height:72px;
}
.tm-detail-filter-hint{color:#93a5b8;font-size:.72rem;text-align:center;margin-top:5px;line-height:1.35;}
.tm-dark-table-wrap{border:1px solid #263548;border-radius:12px;overflow:auto;max-height:560px;background:#0d1721;}
.tm-dark-table{width:100%;border-collapse:collapse;min-width:980px;color:#e8eef6;font-size:.82rem;}
.tm-dark-table th{position:sticky;top:0;z-index:2;background:#142232;color:#c9d5e3;padding:9px 10px;border-right:1px solid #263548;border-bottom:1px solid #33465e;white-space:nowrap;text-align:center;}
.tm-dark-table td{padding:8px 10px;border-right:1px solid #223044;border-bottom:1px solid #1d2b3b;white-space:nowrap;text-align:right;}
.tm-dark-table td:first-child,.tm-dark-table td:nth-child(2),.tm-dark-table td:nth-child(3),.tm-dark-table td:nth-child(4){text-align:left;}
.tm-dark-table tbody tr:nth-child(even){background:#101c29;}
.tm-dark-table tbody tr:hover{background:#17283a;}
.tm-dark-table tbody tr.tm-selected-row{background:#17395d!important;box-shadow:inset 4px 0 0 #ff6b1a;}
.tm-dark-table tbody tr.tm-selected-row td{color:#ffffff;font-weight:700;}
.tm-filter-group-title{font-size:.95rem;font-weight:900;color:#ff7a29!important;margin:14px 0 7px;padding:9px 2px 2px;border-top:1px solid #4a5d72;letter-spacing:.01em;}
.tm-range-caption{color:#91a4b7;font-size:.75rem;margin-top:-6px;margin-bottom:6px;}

.tm-shot-item {
  border:0 !important;
  padding:8px 10px;
  min-height:72px;
  min-width:0;
  overflow:hidden;
  box-sizing:border-box;
  background:linear-gradient(180deg,#111e2b,#0d1721);
}
.tm-shot-label {
  color:#aab7c7;
  font-size:.76rem;
  line-height:1.25;
  margin-bottom:7px;
  white-space:normal;
  overflow-wrap:anywhere;
}
.tm-shot-value {
  color:#f4f8ff;
  font-size:1.27rem;
  font-weight:800;
  line-height:1.15;
  white-space:nowrap;
  letter-spacing:-.02em;
}
.tm-shot-unit {color:#aebdcd; font-size:.72rem; font-weight:650; margin-left:4px; vertical-align:baseline;}
.tm-shot-sub {color:#9fb0c2; font-size:.78rem; margin-top:2px;}
.tm-blue {color:#3d94ff!important;} .tm-red {color:#ff5c4e!important;} .tm-green{color:#52d273!important;}
.sidebar-box {border:1px solid #223044; background:#101b27; border-radius:10px; padding:12px; margin:12px 0;}
.stTabs [data-baseweb="tab-list"] { gap:0; background:#0c1520; border:1px solid #243244; border-radius:10px; overflow:hidden; }
.stTabs [data-baseweb="tab"] { height:46px; padding:0 28px; background:#0d1722; border-right:1px solid #243244; color:#b5c4d7; }
.stTabs [aria-selected="true"] { color:#ff7a29; border-bottom:3px solid #ff6b1a; background:#101b27; }
[data-testid="stDataFrame"] { border:1px solid #223044; border-radius:10px; overflow:hidden; }

.tm-compare-grid{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:12px;margin:10px 0 18px}
.tm-compare-card{border:1px solid #263548;border-radius:10px;background:linear-gradient(180deg,#111e2b,#0d1721);padding:13px 14px}
.tm-compare-title{font-weight:800;color:#f3f7fb;font-size:.95rem;margin-bottom:10px}
.tm-compare-values{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;align-items:end}
.tm-compare-value{font-size:1.35rem;font-weight:800;line-height:1.1}.tm-day{color:#3d94ff}.tm-month{color:#67cf45}.tm-year{color:#aa76f2}
.tm-compare-label{color:#93a5b8;font-size:.72rem;margin-top:4px}.tm-deltas{display:flex;justify-content:space-between;gap:8px;margin-top:10px;padding-top:8px;border-top:1px solid #223044;font-size:.78rem}
.tm-good{color:#62d45a}.tm-bad{color:#ff5a50}.tm-neutral{color:#aab7c7}.tm-auto-summary{border:1px solid #263548;border-radius:12px;padding:16px 18px;background:linear-gradient(180deg,#101b27,#0c151f);line-height:1.75;color:#d8e3ee}
.tm-legend{display:flex;gap:18px;justify-content:flex-end;color:#c8d4e1;font-size:.85rem;margin-bottom:10px}.tm-dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}.tm-dot-day{background:#3d94ff}.tm-dot-month{background:#67cf45}.tm-dot-year{background:#aa76f2}
@media(max-width:1200px){.tm-compare-grid{grid-template-columns:repeat(2,minmax(180px,1fr))}}


.tm-shot-nav {
  display:grid;
  grid-template-columns:120px minmax(220px,1fr) 120px;
  gap:12px;
  align-items:end;
  margin:8px 0 16px;
}
.tm-shot-heading {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  flex-wrap:wrap;
  margin:4px 0 10px;
}
.tm-shot-heading-title {font-size:1.18rem;font-weight:800;color:#f4f8ff}
.tm-shot-heading-sub {font-size:.85rem;color:#9fb0c2}

/* AI club detail: match the dashboard's bright text hierarchy */
.st-key-ai_club_detail h3,
.st-key-ai_club_detail h4,
.st-key-ai_club_detail [data-testid="stMarkdownContainer"] p,
.st-key-ai_club_detail [data-testid="stMetricLabel"] p,
.st-key-ai_club_detail [data-testid="stMetricValue"],
.st-key-ai_club_detail [data-testid="stMetricValue"] > div {
  color:#f4f8ff !important;
  opacity:1 !important;
}
.st-key-ai_club_detail [data-testid="stMetricLabel"] { opacity:1 !important; }
.st-key-ai_club_detail [data-testid="stCaptionContainer"],
.st-key-ai_club_detail .stCaption { color:#c3d0de !important; opacity:1 !important; }
.st-key-ai_club_detail [data-baseweb="select"] > div {
  background:#0d1721 !important;
  color:#f4f8ff !important;
  border-color:#31445a !important;
}
.st-key-ai_club_detail [data-baseweb="select"] span,
.st-key-ai_club_detail [data-baseweb="select"] input {
  color:#f4f8ff !important;
}
.ai-club-detail-title {
  color:#f4f8ff !important;
  font-size:1.45rem;
  font-weight:850;
  margin:18px 0 10px;
}
.tm-shot-compare-grid{display:grid;grid-template-columns:repeat(4,minmax(190px,1fr));gap:12px;margin:10px 0 18px}
.tm-shot-compare-card{border:1px solid #263548;border-radius:10px;background:linear-gradient(180deg,#111e2b,#0d1721);padding:13px 14px}
.tm-shot-compare-title{font-weight:800;color:#f3f7fb;font-size:.95rem;margin-bottom:10px}
.tm-shot-compare-values{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;align-items:end}
.tm-shot-compare-value{font-size:1.14rem;font-weight:800;line-height:1.1}
.tm-shot-current{color:#ff8a32}.tm-shot-day{color:#3d94ff}.tm-shot-month{color:#67cf45}.tm-shot-year{color:#aa76f2}
.tm-shot-compare-label{color:#93a5b8;font-size:.68rem;margin-top:4px}
.tm-shot-deltas{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;margin-top:10px;padding-top:8px;border-top:1px solid #223044;font-size:.72rem}
@media(max-width:1200px){.tm-shot-compare-grid{grid-template-columns:repeat(2,minmax(190px,1fr))}}


/* ============================================================
   Mobile responsive layout (v3.0)
   ============================================================ */
.tm-shot-card { overflow-x: auto; -webkit-overflow-scrolling: touch; }

@media (max-width: 768px) {
  /* Main page spacing */
  .block-container {
    padding-top: .75rem !important;
    padding-bottom: 1.25rem !important;
    padding-left: .75rem !important;
    padding-right: .75rem !important;
    max-width: 100% !important;
  }

  /* Sidebar: larger touch targets and compact spacing */
  section[data-testid="stSidebar"] > div {
    padding-top: .75rem !important;
  }
  [data-testid="stSidebar"] .stButton > button,
  [data-testid="stSidebar"] [data-testid="stDownloadButton"] > button {
    min-height: 46px !important;
    font-size: .92rem !important;
    border-radius: 10px !important;
  }

  /* Stack all Streamlit columns vertically on phones */
  [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap !important;
    gap: .65rem !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 0 !important;
  }

  /* Header / navigation */
  .tm-topbar {
    gap: 8px;
    min-height: 40px;
    padding-bottom: 8px;
    margin-bottom: 10px;
  }
  .tm-logo { font-size: 1.05rem !important; }
  .tm-title { font-size: 1.25rem !important; margin: 4px 0 10px !important; }
  .tm-pill { padding: 6px 9px; font-size: .78rem; margin-right: 2px; }

  /* KPI cards: 2 columns on normal phones */
  .tm-kpi-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 8px !important;
    margin: 8px 0 18px !important;
  }
  .tm-kpi-card {
    min-height: 88px;
    padding: 11px 12px;
    border-radius: 9px;
  }
  .tm-kpi-label {
    font-size: 12px;
    margin-bottom: 8px;
    white-space: normal;
  }
  .tm-kpi-value { font-size: 22px; }
  .tm-kpi-unit { font-size: 11px; margin-left: 2px; }

  /* Custom comparison cards */
  .tm-compare-grid {
    grid-template-columns: 1fr !important;
    gap: 8px !important;
  }
  .tm-compare-card { padding: 12px; }
  .tm-compare-value { font-size: 1.18rem; }
  .tm-legend {
    justify-content: flex-start;
    flex-wrap: wrap;
    gap: 8px 14px;
    font-size: .76rem;
  }

  /* Club cards and panels */
  .tm-card {
    min-height: auto;
    padding: 12px 13px;
  }
  .tm-card .big { font-size: 1.35rem; }
  .tm-panel { padding: 12px; }

  /* Tabs become horizontally scrollable instead of squeezed */
  .stTabs [data-baseweb="tab-list"] {
    overflow-x: auto !important;
    overflow-y: hidden !important;
    flex-wrap: nowrap !important;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
  }
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
  .stTabs [data-baseweb="tab"] {
    flex: 0 0 auto !important;
    height: 42px;
    padding: 0 16px;
    font-size: .82rem;
    white-space: nowrap;
  }

  /* Selects, radios and buttons: comfortable touch size */
  [data-baseweb="select"] > div { min-height: 44px !important; }
  .stButton > button { min-height: 44px; }
  [data-testid="stRadio"] label { min-height: 38px; align-items: center; }

  /* Wide shot detail table stays usable via horizontal swipe */
  .tm-shot-card {
    padding: 12px;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .tm-shot-grid { min-width: 0; }
  .tm-shot-row { grid-template-columns:repeat(2, minmax(0, 1fr)); }
  .tm-shot-item { padding: 5px 9px; }

  /* Charts/images fill phone width without overflowing */
  [data-testid="stImage"] img,
  [data-testid="stPlotlyChart"],
  [data-testid="stVegaLiteChart"],
  [data-testid="stPyplotGlobalUse"] {
    max-width: 100% !important;
  }

  /* Dataframe horizontal swipe */
  [data-testid="stDataFrame"] {
    max-width: 100% !important;
    overflow-x: auto !important;
  }

  /* Reduce heading sizes */
  h1 { font-size: 1.45rem !important; }
  h2 { font-size: 1.25rem !important; }
  .tm-shot-nav { grid-template-columns:1fr 1fr; }
  .tm-shot-nav > :nth-child(2) { grid-column:1 / -1; grid-row:1; }
  .tm-shot-compare-grid { grid-template-columns:1fr !important; gap:8px !important; }
  .tm-shot-compare-value { font-size:1rem; }
  .tm-shot-deltas { grid-template-columns:1fr; }
  h3 { font-size: 1.05rem !important; margin-top: 1rem !important; }
}

@media (max-width: 390px) {
  .tm-kpi-grid { grid-template-columns: 1fr !important; }
  .tm-kpi-card { min-height: 78px; }
  .tm-kpi-label { margin-bottom: 5px; }
  .tm-kpi-value { font-size: 21px; }
  .tm-compare-values { gap: 4px; }
  .tm-compare-value { font-size: 1.05rem; }
  .block-container {
    padding-left: .55rem !important;
    padding-right: .55rem !important;
  }
}

.tm-distribution-header {
  height: 34px;
  display: flex;
  align-items: flex-end;
}
.tm-distribution-header .tm-panel-title {
  margin: 0 !important;
}
.tm-distribution-control-spacer {
  height: 38px;
}



/* Sidebar: text, controls and action buttons must stay readable on dark UI. */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
  color:#eef4fb !important;
}
[data-testid="stSidebar"] details summary,
[data-testid="stSidebar"] details summary p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  color:#ff8a32 !important;
  font-weight:850 !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color:#b5c4d6 !important;
}
[data-testid="stSidebar"] [role="radiogroup"] label p,
[data-testid="stSidebar"] [data-baseweb="checkbox"] p {
  color:#f4f7fb !important;
  font-weight:650 !important;
}
/* Prevent white buttons with white text (logout / reset included). */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
  background:linear-gradient(180deg,#ff812d,#ef5d0e) !important;
  color:#ffffff !important;
  border:1px solid #ff9b54 !important;
  font-weight:800 !important;
  box-shadow:0 3px 12px rgba(255,107,26,.18) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background:linear-gradient(180deg,#ff9348,#ff6b1a) !important;
  color:#ffffff !important;
  border-color:#ffc08b !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
  background:#263548 !important;
  color:#9fb0c2 !important;
  border-color:#405066 !important;
}

/* BaseWeb popover is mounted outside the Streamlit app tree. Style the portal itself. */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] [role="dialog"],
[data-testid="stPopoverBody"] {
  background:#0d1721 !important;
  color:#eef4fb !important;
  border-color:#31445a !important;
}
div[data-baseweb="popover"] h1,
div[data-baseweb="popover"] h2,
div[data-baseweb="popover"] h3,
div[data-baseweb="popover"] h4,
div[data-baseweb="popover"] p,
div[data-baseweb="popover"] label,
div[data-baseweb="popover"] span,
div[data-baseweb="popover"] [data-testid="stMarkdownContainer"],
[data-testid="stPopoverBody"] h4,
[data-testid="stPopoverBody"] p,
[data-testid="stPopoverBody"] label {
  color:#f4f7fb !important;
}
div[data-baseweb="popover"] [data-baseweb="checkbox"] p,
div[data-baseweb="popover"] [data-baseweb="checkbox"] span,
[data-testid="stPopoverBody"] [data-baseweb="checkbox"] p {
  color:#ffffff !important;
  font-weight:700 !important;
  opacity:1 !important;
}
div[data-baseweb="popover"] .tm-filter-group-title,
[data-testid="stPopoverBody"] .tm-filter-group-title {
  color:#ff7a29 !important;
  border-top:1px solid #40546b !important;
  font-weight:900 !important;
  background:#101c28 !important;
  padding:9px 8px !important;
  margin:10px -4px 5px !important;
  border-radius:6px !important;
}
div[data-baseweb="popover"] .stButton > button,
[data-testid="stPopoverBody"] .stButton > button {
  background:#172638 !important;
  color:#ffffff !important;
  border:1px solid #4b6078 !important;
  font-weight:800 !important;
}
div[data-baseweb="popover"] .stButton > button:hover {
  border-color:#ff7a29 !important;
  color:#ff9a55 !important;
}



/* ============================================================
   Phase 3-1 · Practice Journal
   ============================================================ */
.journal-title{font-size:1.55rem;font-weight:850;color:#f7fbff!important;margin:6px 0 4px;letter-spacing:-.02em}
.journal-sub{color:#aab7c7!important;font-size:.87rem;margin:0 0 16px}
.journal-card{border:1px solid #263548;border-radius:12px;padding:16px 17px;background:linear-gradient(180deg,#111e2b,#0d1721);min-height:100%;box-sizing:border-box}
.journal-summary-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}
.journal-kpi{border:1px solid #263548;border-radius:10px;padding:12px 13px;background:#0c1620}
.journal-kpi-label{color:#aab7c7!important;font-size:.75rem;font-weight:650;margin-bottom:6px}
.journal-kpi-value{color:#f4f8ff!important;font-size:1.18rem;font-weight:850;line-height:1.2}
.journal-kpi-value.accent{color:#ff8a32!important}
.journal-ai-title{color:#f5f8fd!important;font-size:1.05rem;font-weight:850;margin-bottom:9px}
.journal-ai-line{color:#d8e3ee!important;font-size:.87rem;line-height:1.65;margin:4px 0}
.journal-ai-coach{color:#aab7c7!important;font-size:.81rem;line-height:1.6;margin-top:10px;padding-top:9px;border-top:1px solid #223044}
.journal-section-title{color:#f5f8fd!important;font-size:1.03rem;font-weight:850;margin:16px 0 8px}
.journal-history-card{border:1px solid #263548;border-radius:10px;padding:10px 12px;background:linear-gradient(180deg,#101c29,#0c151f);margin:7px 0}
.journal-history-date{color:#f3f7fb!important;font-size:.88rem;font-weight:800}
.journal-history-sub{color:#93a5b8!important;font-size:.73rem;margin-top:3px}
.st-key-journal_scope [data-baseweb="select"]>div,.st-key-journal_scope textarea,.st-key-journal_scope input{background:#0d1721!important;color:#f4f8ff!important;border-color:#31445a!important}
.st-key-journal_scope textarea{
  field-sizing:content!important;
  min-height:110px!important;
  line-height:1.55!important;
  overflow-y:hidden!important;
  resize:none!important;
  white-space:pre-wrap!important;
  overflow-wrap:anywhere!important;
}
.st-key-journal_scope textarea:focus,.st-key-journal_scope input:focus{border-color:#ff7a29!important;box-shadow:0 0 0 1px #ff7a29!important}
.st-key-journal_scope label,.st-key-journal_scope [data-testid="stWidgetLabel"] p,.st-key-journal_scope [data-testid="stMarkdownContainer"] p{color:#eef4fb!important}
.st-key-journal_scope [data-testid="stCaptionContainer"]{color:#aab7c7!important}
.st-key-journal_scope [data-testid="stBaseButton-primary"]{background:linear-gradient(180deg,#ff812d,#ef5d0e)!important;color:#fff!important;border:1px solid #ff9b54!important;font-weight:850!important}
@media(max-width:900px){.journal-summary-grid{grid-template-columns:1fr}}

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(AI_CSS, unsafe_allow_html=True)

CLUB_COLORS = {
    "Driver": "#2388ff", "3Wood": "#8b5a3c", "5Wood": "#9b6a50", "7Wood": "#72c7ff",
    "4Hybrid": "#6655ff", "5Hybrid": "#6f7cff",
    "4Iron": "#b08968", "5Iron": "#a16d52", "6Iron": "#c13bd8", "7Iron": "#ff8a00",
    "8Iron": "#ff9f1a", "9Iron": "#ffd23f", "PitchingWedge": "#ff4b4b",
    "50Wedge": "#55d4ff", "52Wedge": "#28c7bb", "56Wedge": "#ff4d6d", "SandWedge": "#ff3b68",
}
SHORT_CLUB = {
    "Driver": "Dr", "5Wood": "5W", "3Wood": "3W", "4Hybrid": "4H", "5Iron": "5i", "6Iron": "6i", "7Iron": "7i", "8Iron": "8i", "9Iron": "9i", "PitchingWedge": "Pw", "50Wedge": "50W", "56Wedge": "56W", "SandWedge": "SW"
}


APP_DIR = Path(__file__).resolve().parent
def load_uploaded_files(files) -> tuple[list[dict], list[str]]:
    all_rows, errors = [], []
    for file in files or []:
        name = file.name.lower()
        try:
            if name.endswith(".csv"):
                df_csv = pd.read_csv(file)
                all_rows.extend(df_csv.to_dict("records"))
            else:
                data = json.loads(file.getvalue().decode("utf-8"))
                if "StrokeGroups" not in data:
                    errors.append(f"{file.name}: StrokeGroups가 없습니다. getactivityreport Response JSON을 올려주세요.")
                    continue
                all_rows.extend(parse_trackman_report(data))
        except Exception as e:
            errors.append(f"{file.name}: {e}")
    return all_rows, errors


def prepare_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    numeric_cols = [c for c in df.columns if c.endswith(("_m", "_mps", "_deg", "_rpm", "_mm", "_s")) or c in ["SmashFactor", "StrokeNo", "AbsTotalSide_m", "Run_m"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "StrokeTime" in df.columns:
        dt = pd.to_datetime(df["StrokeTime"], errors="coerce")
        df["ShotTimeLocal"] = dt.dt.strftime("%H:%M:%S")
    if "TotalSide_m" in df.columns:
        df["SideText"] = df["TotalSide_m"].apply(side_text)
    return df


def card_html(row: pd.Series) -> str:
    club = row["Club"]
    color = CLUB_COLORS.get(club, "#4aa3ff")
    return f"""
    <div class="tm-card">
      <h4 style="color:{color};">{club} <span class="tm-muted">{int(row['Shots'])} Shots</span></h4>
      <div class="small">Carry&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Total</div>
      <div style="display:flex;gap:26px;align-items:baseline;">
        <div class="big">{fmt_int(row['Avg_Carry_m'])}<span class="small"> m</span></div>
        <div class="big">{fmt_int(row['Avg_Total_m'])}<span class="small"> m</span></div>
      </div>
      <div class="small" style="margin-top:10px;">
        Run&nbsp;&nbsp;{fmt_int(row['Avg_Run_m'])}m&nbsp;&nbsp;&nbsp;&nbsp; Smash&nbsp;&nbsp;{fmt(row['Avg_Smash'],2)}<br>
        Spin&nbsp;&nbsp;{fmt_int(row['Avg_Spin_rpm'], comma=True)}rpm
      </div>
    </div>
    """


def render_club_cards(summary: pd.DataFrame, max_cards: int = 6) -> None:
    st.markdown("### 클럽별 평균 (선택 클럽)")
    show = summary.head(max_cards)
    cols = st.columns(min(max_cards, max(1, len(show))))
    for idx, (_, row) in enumerate(show.iterrows()):
        with cols[idx % len(cols)]:
            st.markdown(card_html(row), unsafe_allow_html=True)
    remain = len(summary) - len(show)
    if remain > 0:
        st.caption(f"+ {remain}개 클럽은 아래 표와 리스트에서 확인")


def trajectory_mini_fig(row):
    carry = row.get("Carry_m"); total = row.get("Total_m"); h = row.get("MaxHeight_m")
    if pd.isna(carry): carry = 100
    if pd.isna(total): total = carry
    if pd.isna(h): h = 20
    fig, ax = plt.subplots(figsize=(3.2,1.15), dpi=150)
    fig.patch.set_facecolor("#101b27"); ax.set_facecolor("#101b27")
    ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    x = np.linspace(.08,.78,80); y = .18 + .55*np.sin(np.pi*(x-.08)/.70)
    ax.plot([.05,.9],[.17,.17], color="#6f9a4e", lw=8, alpha=.9)
    ax.plot(x,y,color="#ff7a00",lw=1.6)
    ax.scatter([.06],[.17],s=65,color="#e8edf2",edgecolors="#cfd6e0")
    ax.plot([.78,.86],[.17,.35],color="#2f86ff",lw=1)
    ax.text(.86,.52,f"높이\n{fmt(h,1)} m",color="#cbd6e5",fontsize=7,ha="center")
    ax.text(.86,.04,f"런\n{fmt((total or 0)-(carry or 0),1)} m",color="#cbd6e5",fontsize=7,ha="center")
    return fig


def render_metric_bar(label, avg, std, max_val=None):
    if pd.isna(avg): avg=0
    if max_val is None: max_val = max(abs(avg)*1.4, 1)
    width = int(min(100, max(3, abs(float(avg))/max_val*100)))
    return f"""
    <div style='display:grid;grid-template-columns:120px 80px 1fr 70px;align-items:center;gap:8px;margin:5px 0;'>
      <div class='tm-muted'>{label}</div><div>{fmt(avg,1)}</div>
      <div style='height:10px;background:#1b2838;border-radius:5px;overflow:hidden;'><div style='height:10px;width:{width}%;background:#3d79ff;border-radius:5px;'></div></div>
      <div class='tm-muted'>± {fmt(std,1)}</div>
    </div>"""


def render_distance_chart(summary):
    chart = summary[["Club","Avg_Carry_m","Avg_Total_m"]].dropna()
    st.bar_chart(chart.set_index("Club")[["Avg_Carry_m","Avg_Total_m"]], use_container_width=True)


SHOT_DETAIL_DEFAULT_FIELDS = [
    "Carry_m", "Total_m", "BallSpeed_mps", "ClubSpeed_mps", "SmashFactor",
    "LaunchAngle_deg", "SpinRate_rpm", "TotalSide_m", "ClubPath_deg",
    "FaceAngle_deg", "FaceToPath_deg",
]

SHOT_DETAIL_FIELD_META: dict[str, tuple[str, str, int, str]] = {
    "StrokeNo": ("샷 번호", "", 0, "number"),
    "Carry_m": ("캐리", "m", 1, "number"),
    "Total_m": ("토탈", "m", 1, "number"),
    "Run_m": ("런", "m", 1, "number"),
    "BallSpeed_mps": ("볼 스피드", "m/s", 1, "number"),
    "ClubSpeed_mps": ("클럽 스피드", "m/s", 1, "number"),
    "SmashFactor": ("스매시 팩터", "", 2, "number"),
    "SpinRate_rpm": ("스핀량", "rpm", 0, "comma"),
    "LaunchAngle_deg": ("발사각", "°", 1, "number"),
    "AttackAngle_deg": ("어택 앵글", "°", 1, "number"),
    "ClubPath_deg": ("클럽 패스", "°", 1, "signed_angle"),
    "FaceAngle_deg": ("페이스 앵글", "°", 1, "signed_angle"),
    "FaceToPath_deg": ("페이스 투 패스", "°", 1, "signed_angle"),
    "DynamicLoft_deg": ("다이나믹 로프트", "°", 1, "number"),
    "DynamicLoftAngle_deg": ("다이나믹 로프트", "°", 1, "number"),
    "SpinLoft_deg": ("스핀 로프트", "°", 1, "number"),
    "SpinLoftAngle_deg": ("스핀 로프트", "°", 1, "number"),
    "TotalSide_m": ("사이드", "m", 1, "side"),
    "AbsTotalSide_m": ("절대 좌우 편차", "m", 1, "number"),
    "MaxHeight_m": ("최고점", "m", 1, "number"),
    "ImpactOffset_mm": ("임팩트 좌우", "mm", 1, "number"),
    "ImpactHeight_mm": ("임팩트 높이", "mm", 1, "number"),
}

SHOT_DETAIL_EXCLUDED_FIELDS = {
    "SideText", "AbsTotalSide_m", "StrokeGroups", "Raw", "raw", "SourceFile",
    # 화면 분석에 불필요한 식별자/메타데이터
    "Date", "StrokeTime", "ShotTimeLocal", "Club", "ClubName",
    "GroupID", "GroupId", "groupid", "GroupClub", "GroupClup", "groupclub", "groupclup",
    "StrokeID", "StrokeId", "strokeid",
    "MeasurementKind", "measurementkind",
}

SHOT_DETAIL_EXCLUDED_NORMALIZED = {
    "date", "time", "stroktime", "shottimelocal", "club", "clubname",
    "groupid", "groupclub", "groupclup", "strokeid", "measurementkind",
}


def _shot_detail_label(column: str) -> str:
    if column in SHOT_DETAIL_FIELD_META:
        return SHOT_DETAIL_FIELD_META[column][0]
    label = column.replace("_mps", "").replace("_rpm", "").replace("_deg", "")
    label = label.replace("_mm", "").replace("_m", "").replace("_s", "")
    return label.replace("_", " ")


def _shot_detail_available_fields(row: pd.Series) -> list[str]:
    """현재 샷에 실제 값이 존재하는 상세 데이터 컬럼을 화면 표시 순서로 반환합니다."""
    preferred = list(SHOT_DETAIL_FIELD_META)
    remaining = [str(column) for column in row.index if str(column) not in preferred]
    candidates = preferred + remaining
    fields: list[str] = []
    for column in candidates:
        normalized_column = "".join(ch for ch in str(column).lower() if ch.isalnum())
        if (
            column in SHOT_DETAIL_EXCLUDED_FIELDS
            or normalized_column in SHOT_DETAIL_EXCLUDED_NORMALIZED
            or column not in row.index
        ):
            continue
        value = row.get(column)
        try:
            if value is None or pd.isna(value):
                continue
        except (TypeError, ValueError):
            if value is None:
                continue
        if isinstance(value, (dict, list, tuple, set, np.ndarray)):
            continue
        fields.append(column)
    return fields


def _format_shot_detail_field(row: pd.Series, column: str) -> tuple[str, str, str]:
    label, unit, nd, display_type = SHOT_DETAIL_FIELD_META.get(
        column,
        (_shot_detail_label(column), "", 1, "auto"),
    )
    value = row.get(column)
    if display_type == "text":
        return label, str(value), unit
    if display_type == "side":
        return label, side_text(value), unit
    if display_type == "comma":
        return label, fmt_int(value, comma=True), unit
    if display_type == "signed_angle":
        number = _safe_numeric(value)
        if pd.isna(number):
            return label, "-", unit
        direction = "R" if number > 0.05 else "L" if number < -0.05 else ""
        return label, f"{abs(number):.{nd}f}{direction}", unit

    if display_type == "number":
        return label, fmt(value, nd), unit

    # 메타데이터가 없는 숫자 컬럼도 접미사에 따라 단위를 자동 추론합니다.
    if column.endswith("_mps"):
        unit = "m/s"
    elif column.endswith("_rpm"):
        unit = "rpm"
    elif column.endswith("_deg"):
        unit = "°"
    elif column.endswith("_mm"):
        unit = "mm"
    elif column.endswith("_m"):
        unit = "m"
    elif column.endswith("_s"):
        unit = "s"

    number = _safe_numeric(value)
    if not pd.isna(number):
        if unit == "rpm":
            return label, f"{int(round(number)):,}", unit
        return label, f"{number:.1f}", unit
    return label, str(value), unit


def shot_detail_html(row: pd.Series, selected_fields: list[str]) -> str:
    """상세 수치를 9칸 단위로 표시하고, 첫 카드는 클럽 약어만 보여줍니다."""
    vals = [_format_shot_detail_field(row, column) for column in selected_fields if column in row.index]

    club = row.get("Club")
    club_item = (
        "<div class='tm-shot-item tm-shot-club-only'>"
        f"<div style='font-weight:900;color:{CLUB_COLORS.get(club, '#4aa3ff')};font-size:1.55rem;line-height:1;'>"
        f"{SHORT_CLUB.get(club, club)}</div>"
        "</div>"
    )

    metric_items = [
        "<div class='tm-shot-item'>"
        f"<div class='tm-shot-label'>{label}</div>"
        "<div class='tm-shot-value'>"
        f"{value}"
        + (f"<span class='tm-shot-unit'>{unit}</span>" if unit else "")
        + "</div></div>"
        for label, value, unit in vals
    ]

    # 첫 행에는 클럽 약어 카드가 포함됩니다. 이후 행에는 실제 항목만 배치하여
    # 남는 칸이 별도 색으로 채워지지 않도록 합니다.
    all_items = [club_item] + metric_items
    rows = []
    for start in range(0, len(all_items), 9):
        rows.append("<div class='tm-shot-row'>" + "".join(all_items[start:start + 9]) + "</div>")

    return "<div class='tm-shot-card'><div class='tm-shot-grid'>" + "".join(rows) + "</div></div>"


SHOT_DETAIL_FIELD_GROUPS = {
    "거리": ["Carry_m", "Total_m", "Run_m", "TotalSide_m", "AbsTotalSide_m"],
    "스피드": ["BallSpeed_mps", "ClubSpeed_mps", "SmashFactor"],
    "볼 비행": ["LaunchAngle_deg", "SpinRate_rpm", "MaxHeight_m"],
    "클럽/스윙 방향": ["ClubPath_deg", "FaceAngle_deg", "FaceToPath_deg"],
    "임팩트 / 기타": [
        "AttackAngle_deg", "DynamicLoft_deg", "DynamicLoftAngle_deg",
        "SpinLoft_deg", "SpinLoftAngle_deg", "ImpactOffset_mm", "ImpactHeight_mm",
    ],
}


def _detail_field_group(field: str) -> str:
    for group, fields in SHOT_DETAIL_FIELD_GROUPS.items():
        if field in fields:
            return group
    return "기타 데이터"


def render_dark_dataframe(
    source_df: pd.DataFrame,
    max_rows: int = 500,
    selected_row: int | None = None,
) -> None:
    """Streamlit 테마와 일치하는 다크 HTML 테이블을 렌더링합니다."""
    import html as html_lib

    if source_df is None or source_df.empty:
        st.info("표시할 데이터가 없습니다.")
        return

    safe_df = safe_dataframe_for_streamlit(source_df.head(max_rows)).copy()

    def display_value(value: Any) -> str:
        if value is None:
            return "-"
        try:
            if pd.isna(value):
                return "-"
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)

    headers = "".join(
        f"<th>{html_lib.escape(str(column))}</th>"
        for column in safe_df.columns
    )
    body_rows: list[str] = []
    for row_position, (_, row) in enumerate(safe_df.iterrows()):
        selected_class = " class='tm-selected-row'" if selected_row == row_position else ""
        cells = "".join(
            f"<td>{html_lib.escape(display_value(row[column]))}</td>"
            for column in safe_df.columns
        )
        body_rows.append(f"<tr{selected_class}>{cells}</tr>")

    table_html = (
        "<div class='tm-dark-table-wrap'>"
        "<table class='tm-dark-table'>"
        f"<thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
    if len(source_df) > max_rows:
        st.caption(
            f"성능을 위해 앞의 {max_rows:,}개 행만 표시했습니다. "
            "전체 행은 CSV 다운로드로 확인할 수 있습니다."
        )


def _numeric_range(series: pd.Series) -> tuple[float, float] | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    low, high = float(values.min()), float(values.max())
    if low == high:
        pad = max(abs(low) * .05, 1.0)
        return low - pad, high + pad
    return low, high


def _apply_numeric_range(df_source: pd.DataFrame, column: str, selected_range: tuple[float, float]) -> pd.DataFrame:
    if column not in df_source.columns:
        return df_source
    values = pd.to_numeric(df_source[column], errors="coerce")
    low, high = selected_range
    return df_source[values.between(low, high, inclusive="both")]


def _detail_checkbox_key(state_suffix: str, field: str) -> str:
    return f"detail_check::{state_suffix}::{field}"


def _sync_detail_selection_from_checkboxes(
    state_suffix: str,
    selection_key: str,
    available_fields: list[str],
) -> None:
    """체크박스 상태를 상세 카드 선택 목록에 즉시 반영합니다."""
    st.session_state[selection_key] = [
        field
        for field in available_fields
        if bool(st.session_state.get(_detail_checkbox_key(state_suffix, field), False))
    ]


def _set_detail_selection(
    state_suffix: str,
    selection_key: str,
    available_fields: list[str],
    selected_fields: list[str],
) -> None:
    """기본값/모두 표시 버튼에서 체크박스와 카드 상태를 동시에 갱신합니다."""
    selected_set = set(selected_fields)
    st.session_state[selection_key] = [
        field for field in available_fields if field in selected_set
    ]
    for field in available_fields:
        st.session_state[_detail_checkbox_key(state_suffix, field)] = field in selected_set


def render_shot_detail_panel(row: pd.Series, state_suffix: str) -> None:
    """상세 데이터 카드와 이퀄라이저 모양의 표시 항목 설정 팝오버를 렌더링합니다."""
    available_fields = _shot_detail_available_fields(row)
    default_fields = [field for field in SHOT_DETAIL_DEFAULT_FIELDS if field in available_fields]
    selection_key = f"shot_detail_fields::{state_suffix}"

    # 선택 목록을 현재 샷에서 실제 사용할 수 있는 필드로 정리합니다.
    existing_selection = st.session_state.get(selection_key)
    if existing_selection is None:
        current_selection = list(default_fields)
    else:
        current_selection = [field for field in existing_selection if field in available_fields]
        if not current_selection:
            current_selection = list(default_fields)
    st.session_state[selection_key] = current_selection

    # 체크박스는 value 인자를 사용하지 않고 Session State로만 초기화합니다.
    # 이렇게 해야 "default value + Session State" 중복 경고가 발생하지 않습니다.
    selected_set = set(current_selection)
    for field in available_fields:
        checkbox_key = _detail_checkbox_key(state_suffix, field)
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = field in selected_set

    with st.container(border=False):
        st.markdown("<div class='tm-shot-detail-shell'>", unsafe_allow_html=True)
        detail_col, filter_col = st.columns([12.2, 1.05], vertical_alignment="center")

        with detail_col:
            # 체크박스 on_change 콜백이 rerun 전에 selection_key를 갱신하므로
            # 클릭 직후 현재 선택 항목이 빠짐없이 표시됩니다.
            selected_fields = list(st.session_state.get(selection_key, default_fields))
            st.markdown(shot_detail_html(row, selected_fields), unsafe_allow_html=True)

        with filter_col:
            with st.popover("☷", help="상세 데이터 표시 항목 설정", width="stretch"):
                st.markdown(
                    "<h4 style='color:#ff8a32;margin:0 0 6px 0'>상세 데이터 필터</h4>",
                    unsafe_allow_html=True,
                )
                st.caption("체크한 데이터만 상세 카드에 표시됩니다.")

                action_left, action_right = st.columns(2)
                with action_left:
                    st.button(
                        "기본값",
                        key=f"detail_default::{state_suffix}",
                        width="stretch",
                        on_click=_set_detail_selection,
                        args=(state_suffix, selection_key, available_fields, default_fields),
                    )
                with action_right:
                    st.button(
                        "모두 표시",
                        key=f"detail_all::{state_suffix}",
                        width="stretch",
                        on_click=_set_detail_selection,
                        args=(state_suffix, selection_key, available_fields, available_fields),
                    )

                grouped: dict[str, list[str]] = {}
                for field in available_fields:
                    grouped.setdefault(_detail_field_group(field), []).append(field)

                group_order = [
                    "거리",
                    "스피드",
                    "볼 비행",
                    "클럽/스윙 방향",
                    "임팩트 / 기타",
                    "기타 데이터",
                ]
                for group_name in group_order:
                    fields = grouped.get(group_name, [])
                    if not fields:
                        continue
                    st.markdown(
                        f"<div class='tm-filter-group-title'>{group_name}</div>",
                        unsafe_allow_html=True,
                    )
                    for field in fields:
                        st.checkbox(
                            _shot_detail_label(field),
                            key=_detail_checkbox_key(state_suffix, field),
                            on_change=_sync_detail_selection_from_checkboxes,
                            args=(state_suffix, selection_key, available_fields),
                        )

                selected_count = len(st.session_state.get(selection_key, []))
                st.caption(f"현재 {selected_count}개 / 전체 {len(available_fields)}개 표시")

            st.markdown(
                "<div class='tm-detail-filter-hint'>상세 데이터<br>추가·삭제</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)



def _summary(df_period: pd.DataFrame, club: str, mode: str) -> pd.Series:
    part=df_period[df_period['Club']==club].copy()
    if part.empty: return pd.Series(dtype='object')
    if mode=='전체 샷 평균':
        s=pd.DataFrame(make_summary(part.to_dict('records')))
        return s.iloc[0] if not s.empty else pd.Series(dtype='object')
    daily=[]
    for _,g in part.groupby('Date'):
        s=pd.DataFrame(make_summary(g.to_dict('records')))
        if not s.empty: daily.append(s.iloc[0])
    if not daily: return pd.Series(dtype='object')
    d=pd.DataFrame(daily); result={'Club':club,'Shots':len(part),'Sessions':part['Date'].nunique()}
    for c in d.columns:
        if c=='Club': continue
        v=pd.to_numeric(d[c],errors='coerce')
        if v.notna().any(): result[c]=float(v.mean())
    return pd.Series(result)

def _raw_avg(df_period, club, candidates, mode):
    part=df_period[df_period['Club']==club].copy()
    for c in candidates:
        if c not in part.columns: continue
        part[c]=pd.to_numeric(part[c],errors='coerce')
        if mode=='연습일 평균':
            v=part.groupby('Date')[c].mean().dropna()
        else: v=part[c].dropna()
        if not v.empty: return float(v.mean())
    return float('nan')

def _periods(df_all, selected_date, exclude):
    ts=pd.to_datetime(selected_date); ds=pd.to_datetime(df_all['Date'],errors='coerce')
    day=df_all[ds.dt.date==ts.date()].copy()
    month=df_all[(ds.dt.year==ts.year)&(ds.dt.month==ts.month)].copy()
    year=df_all[ds.dt.year==ts.year].copy()
    if exclude:
        month=month[month['Date']!=selected_date].copy(); year=year[year['Date']!=selected_date].copy()
    return day,month,year,f'{ts.year}-{ts.month:02d}',str(ts.year)

def _card(title,d,m,y,unit='',nd=1,lower=False):
    def f(v):
        if v is None or pd.isna(v): return '-'
        return f'{int(round(float(v))):,}' if nd==0 else f'{float(v):.{nd}f}'
    def delta(base,val):
        if base is None or val is None or pd.isna(base) or pd.isna(val): return '-', 'tm-neutral'
        x=float(val)-float(base); good=x<0 if lower else x>0
        css='tm-good' if good else 'tm-bad' if abs(x)>1e-9 else 'tm-neutral'; sign='+' if x>0 else ''
        txt=f'{sign}{int(round(x)):,}{unit}' if nd==0 else f'{sign}{x:.{nd}f}{unit}'
        return txt,css
    dm,cm=delta(m,d); dy,cy=delta(y,d)
    return f"<div class='tm-compare-card'><div class='tm-compare-title'>{title}<span style='float:right;color:#9fb0c2;font-weight:500'>{unit}</span></div><div class='tm-compare-values'><div><div class='tm-compare-value tm-day'>{f(d)}</div><div class='tm-compare-label'>선택일</div></div><div><div class='tm-compare-value tm-month'>{f(m)}</div><div class='tm-compare-label'>월간 평균</div></div><div><div class='tm-compare-value tm-year'>{f(y)}</div><div class='tm-compare-label'>연간 평균</div></div></div><div class='tm-deltas'><span>vs 월간 <b class='{cm}'>{dm}</b></span><span>vs 연간 <b class='{cy}'>{dy}</b></span></div></div>"

def _render_compare_cards(day,month,year):
    items=[('캐리','Avg_Carry_m','m',0,False),('토탈','Avg_Total_m','m',0,False),('볼 스피드','Avg_BallSpeed_mps','m/s',1,False),('클럽 스피드','Avg_ClubSpeed_mps','m/s',1,False),('스매시 팩터','Avg_Smash','',2,False),('스핀량','Avg_Spin_rpm','rpm',0,False),('발사각','Avg_Launch_deg','°',1,False),('좌우 편차','Avg_AbsSide_m','m',1,True)]
    html=''.join(_card(t,day.get(c),month.get(c),year.get(c),u,n,l) for t,c,u,n,l in items)
    st.markdown("<div class='tm-compare-grid'>"+html+"</div>",unsafe_allow_html=True)

def _period_row(raw, summary, club, mode):
    return pd.Series({
        'Club': club,
        'ClubPath_deg': summary.get('Avg_Path_deg'),
        'FaceAngle_deg': summary.get('Avg_Face_deg'),
        'FaceToPath_deg': summary.get('Avg_FaceToPath_deg'),
        'AttackAngle_deg': summary.get('Avg_Attack_deg'),
        'LaunchAngle_deg': summary.get('Avg_Launch_deg'),
        'DynamicLoft_deg': _raw_avg(
            raw, club,
            ['DynamicLoft_deg', 'DynamicLoft', 'DynamicLoftAngle_deg'],
            mode,
        ),
        'SpinLoft_deg': _raw_avg(
            raw, club,
            ['SpinLoft_deg', 'SpinLoft', 'SpinLoftAngle_deg'],
            mode,
        ),
        'SpinRate_rpm': summary.get('Avg_Spin_rpm'),
    })

def _distance_chart(day, month, year, club, metric="Carry_m"):
    label_map = {"Carry_m": "Carry", "Total_m": "Total"}
    metric_label = label_map.get(metric, metric)
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=150)
    fig.patch.set_facecolor('#101b27')
    ax.set_facecolor('#101b27')
    data = [('선택일', day, '#3d94ff'), ('월간', month, '#67cf45'), ('연간', year, '#aa76f2')]
    vals = []
    for _, d, _ in data:
        if metric in d.columns:
            vals.extend(pd.to_numeric(d[d['Club'] == club][metric], errors='coerce').dropna().tolist())
    if not vals:
        st.info(f'{metric_label} 데이터 없음')
        plt.close(fig)
        return
    xs = np.linspace(min(vals) - 8, max(vals) + 8, 220)
    for label, d, color in data:
        if metric not in d.columns:
            continue
        v = pd.to_numeric(d[d['Club'] == club][metric], errors='coerce').dropna()
        if len(v) < 2:
            continue
        mu = float(v.mean())
        sd = max(float(v.std(ddof=0)), 2)
        y = np.exp(-.5 * ((xs - mu) / sd) ** 2) / sd
        ax.plot(xs, y, color=color, lw=2, label=label)
        ax.axvline(mu, color=color, ls='--', lw=.8, alpha=.7)
    ax.legend(frameon=False, labelcolor='#cbd6e5', fontsize=8)
    ax.set_yticks([])
    ax.tick_params(colors='#aab7c7', labelsize=7)
    ax.set_xlabel(f'{metric_label} (m)', color='#aab7c7')
    for spine in ax.spines.values():
        spine.set_color('#263548')
    st.pyplot(fig, clear_figure=True)

def _side_chart(day, month, year, club, metric="Carry_m"):
    """선택 거리 기준으로 선택일/월간/연간 탄착군을 비교합니다."""
    metric_label = "캐리" if metric == "Carry_m" else "토탈"
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=150)
    fig.patch.set_facecolor('#101b27')
    ax.set_facecolor('#101b27')

    has_data = False
    for label, data, color in [
        ('선택일', day, '#3d94ff'),
        ('월간', month, '#67cf45'),
        ('연간', year, '#aa76f2'),
    ]:
        if metric not in data.columns or 'TotalSide_m' not in data.columns:
            continue
        part = data[data['Club'] == club][['TotalSide_m', metric]].copy()
        part['TotalSide_m'] = pd.to_numeric(part['TotalSide_m'], errors='coerce')
        part[metric] = pd.to_numeric(part[metric], errors='coerce')
        part = part.dropna()
        if part.empty:
            continue
        has_data = True
        ax.scatter(part['TotalSide_m'], part[metric], s=14, color=color, alpha=.42, label=label)

        if len(part) >= 2:
            x_mean = float(part['TotalSide_m'].mean())
            y_mean = float(part[metric].mean())
            x_std = max(float(part['TotalSide_m'].std(ddof=0)), 1.5)
            y_std = max(float(part[metric].std(ddof=0)), 2.0)
            ax.add_patch(
                patches.Ellipse(
                    (x_mean, y_mean),
                    width=x_std * 4.0,
                    height=y_std * 4.0,
                    fill=False,
                    edgecolor=color,
                    lw=1.1,
                    alpha=.9,
                )
            )

    if not has_data:
        st.info(f'{metric_label} 탄착군 데이터 없음')
        plt.close(fig)
        return

    ax.axvline(0, color='#d8e0ea', lw=1.0, alpha=.75)
    x_values = []
    for data in (day, month, year):
        if 'TotalSide_m' in data.columns:
            x_values.extend(pd.to_numeric(data[data['Club'] == club]['TotalSide_m'], errors='coerce').dropna().tolist())
    x_limit = max(10, int(math.ceil(max([abs(v) for v in x_values] or [10]) / 5.0) * 5))
    ax.set_xlim(-x_limit, x_limit)
    ax.set_xticks(range(-x_limit, x_limit + 1, 5))
    ax.grid(True, color='#314154', alpha=.38, linestyle='--', linewidth=.7)
    ax.set_xlabel('좌우 편차 (m)', color='#aab7c7')
    ax.set_ylabel(f'{metric_label} 거리 (m)', color='#aab7c7')
    ax.tick_params(colors='#aab7c7', labelsize=7)
    ax.legend(frameon=False, labelcolor='#cbd6e5', fontsize=7)
    for spine in ax.spines.values():
        spine.set_color('#263548')
    st.pyplot(fig, clear_figure=True)

def _trend(df_all, club, year, mode, metric="Carry_m"):
    """선택한 캐리/토탈 기준으로 월별 거리 추세를 표시합니다."""
    metric_label = "캐리" if metric == "Carry_m" else "토탈"
    summary_column = "Avg_Carry_m" if metric == "Carry_m" else "Avg_Total_m"
    ds = pd.to_datetime(df_all['Date'], errors='coerce')
    part = df_all[(df_all['Club'] == club) & (ds.dt.year == year)].copy()
    rec = []
    for month_number in range(1, 13):
        month_part = part[pd.to_datetime(part['Date']).dt.month == month_number]
        if month_part.empty:
            continue
        value = _summary(month_part, club, mode).get(summary_column)
        if value is not None and not pd.isna(value):
            rec.append((month_number, float(value)))
    if not rec:
        st.info(f'{metric_label} 추세 데이터 없음')
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.2), dpi=150)
    fig.patch.set_facecolor('#101b27')
    ax.set_facecolor('#101b27')
    ax.plot([x for x, _ in rec], [y for _, y in rec], marker='o', lw=2, color='#ff7a29')
    for x, y in rec:
        ax.text(x, y + .7, fmt_int(y), color='#e8eef5', fontsize=7, ha='center')
    ax.set_xticks([x for x, _ in rec], [f'{x}월' for x, _ in rec])
    ax.tick_params(colors='#aab7c7', labelsize=7)
    ax.set_ylabel(f'{metric_label} (m)', color='#aab7c7')
    ax.grid(axis='y', color='#263548', alpha=.45)
    for spine in ax.spines.values():
        spine.set_color('#263548')
    st.pyplot(fig, clear_figure=True)

def _auto_text(day,month,year,club):
    def d(col,b):
        a=day.get(col); z=b.get(col)
        return None if a is None or z is None or pd.isna(a) or pd.isna(z) else float(a)-float(z)
    cm=d('Avg_Carry_m',month); cy=d('Avg_Carry_m',year); sm=d('Avg_Smash',month); side=d('Avg_AbsSide_m',month); spin=d('Avg_Spin_rpm',year)
    t=[f'<b>{club} 선택일 분석</b>']
    if cm is not None and cy is not None: t.append(f'캐리는 월간 대비 <b>{cm:+.0f}m</b>, 연간 대비 <b>{cy:+.0f}m</b>입니다.')
    if sm is not None: t.append(f'스매시 팩터는 월간 대비 <b>{sm:+.2f}</b>입니다.')
    if side is not None: t.append(f'좌우 편차는 월간 대비 <b>{side:+.1f}m</b>입니다.')
    if spin is not None: t.append(f'스핀량은 연간 대비 <b>{spin:+.0f}rpm</b>입니다.')
    return ' '.join(t)



def _safe_numeric(value: Any) -> float:
    """샷/평균 비교에 사용할 값을 안전하게 float로 변환합니다."""
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _shot_sort_columns(df_shots: pd.DataFrame) -> list[str]:
    """데이터에 실제로 존재하는 컬럼만 사용해 샷 순서를 정합니다."""
    return [column for column in ["Date", "StrokeTime", "StrokeNo"] if column in df_shots.columns]


def _shot_display_number(row: pd.Series, fallback_index: int) -> str:
    stroke_no = _safe_numeric(row.get("StrokeNo"))
    if not pd.isna(stroke_no):
        return str(int(round(stroke_no)))
    return str(fallback_index + 1)


def _shot_value(row: pd.Series, candidates: list[str]) -> float:
    for column in candidates:
        if column in row.index:
            value = _safe_numeric(row.get(column))
            if not pd.isna(value):
                return value
    return float("nan")


def _shot_metric_items(row: pd.Series) -> list[tuple[str, str, str]]:
    """선택한 한 샷의 핵심 지표를 상단 KPI 형식으로 반환합니다."""
    return [
        ("캐리", fmt_int(row.get("Carry_m")), "m"),
        ("토탈", fmt_int(row.get("Total_m")), "m"),
        ("런", fmt_int(row.get("Run_m")), "m"),
        ("볼 스피드", fmt(row.get("BallSpeed_mps"), 1), "m/s"),
        ("클럽 스피드", fmt(row.get("ClubSpeed_mps"), 1), "m/s"),
        ("스매시 팩터", fmt(row.get("SmashFactor"), 2), ""),
        ("스핀량", fmt_int(row.get("SpinRate_rpm"), comma=True), "rpm"),
        ("발사각", fmt(row.get("LaunchAngle_deg"), 1), "°"),
        ("사이드", side_text(row.get("TotalSide_m")), "m"),
    ]


def _format_compare_value(value: Any, nd: int) -> str:
    number = _safe_numeric(value)
    if pd.isna(number):
        return "-"
    if nd == 0:
        return f"{int(round(number)):,}"
    return f"{number:.{nd}f}"


def _format_shot_delta(shot_value: Any, average_value: Any, unit: str, nd: int, lower_is_better: bool) -> tuple[str, str]:
    shot_number = _safe_numeric(shot_value)
    average_number = _safe_numeric(average_value)
    if pd.isna(shot_number) or pd.isna(average_number):
        return "-", "tm-neutral"

    delta = shot_number - average_number
    if abs(delta) < 1e-9:
        css = "tm-neutral"
    else:
        improved = delta < 0 if lower_is_better else delta > 0
        css = "tm-good" if improved else "tm-bad"

    sign = "+" if delta > 0 else ""
    if nd == 0:
        return f"{sign}{int(round(delta)):,}{unit}", css
    return f"{sign}{delta:.{nd}f}{unit}", css


def _shot_compare_card(
    title: str,
    shot_value: Any,
    day_value: Any,
    month_value: Any,
    year_value: Any,
    unit: str = "",
    nd: int = 1,
    lower_is_better: bool = False,
) -> str:
    day_delta, day_css = _format_shot_delta(shot_value, day_value, unit, nd, lower_is_better)
    month_delta, month_css = _format_shot_delta(shot_value, month_value, unit, nd, lower_is_better)
    year_delta, year_css = _format_shot_delta(shot_value, year_value, unit, nd, lower_is_better)
    return (
        "<div class='tm-shot-compare-card'>"
        f"<div class='tm-shot-compare-title'>{title}<span style='float:right;color:#9fb0c2;font-weight:500'>{unit}</span></div>"
        "<div class='tm-shot-compare-values'>"
        f"<div><div class='tm-shot-compare-value tm-shot-current'>{_format_compare_value(shot_value, nd)}</div><div class='tm-shot-compare-label'>선택 샷</div></div>"
        f"<div><div class='tm-shot-compare-value tm-shot-day'>{_format_compare_value(day_value, nd)}</div><div class='tm-shot-compare-label'>당일 평균</div></div>"
        f"<div><div class='tm-shot-compare-value tm-shot-month'>{_format_compare_value(month_value, nd)}</div><div class='tm-shot-compare-label'>월간 평균</div></div>"
        f"<div><div class='tm-shot-compare-value tm-shot-year'>{_format_compare_value(year_value, nd)}</div><div class='tm-shot-compare-label'>연간 평균</div></div>"
        "</div>"
        "<div class='tm-shot-deltas'>"
        f"<span>vs 당일 <b class='{day_css}'>{day_delta}</b></span>"
        f"<span>vs 월간 <b class='{month_css}'>{month_delta}</b></span>"
        f"<span>vs 연간 <b class='{year_css}'>{year_delta}</b></span>"
        "</div></div>"
    )


def _render_shot_compare_cards(row: pd.Series, day_summary: pd.Series, month_summary: pd.Series, year_summary: pd.Series) -> None:
    """Step 3: 선택 샷을 당일·월간·연간 평균과 비교합니다."""
    definitions = [
        ("캐리", ["Carry_m"], "Avg_Carry_m", "m", 0, False),
        ("토탈", ["Total_m"], "Avg_Total_m", "m", 0, False),
        ("볼 스피드", ["BallSpeed_mps"], "Avg_BallSpeed_mps", "m/s", 1, False),
        ("클럽 스피드", ["ClubSpeed_mps"], "Avg_ClubSpeed_mps", "m/s", 1, False),
        ("스매시 팩터", ["SmashFactor"], "Avg_Smash", "", 2, False),
        ("스핀량", ["SpinRate_rpm"], "Avg_Spin_rpm", "rpm", 0, False),
        ("발사각", ["LaunchAngle_deg"], "Avg_Launch_deg", "°", 1, False),
        ("좌우 편차", ["AbsTotalSide_m", "TotalSide_m"], "Avg_AbsSide_m", "m", 1, True),
    ]
    cards = []
    for title, raw_columns, summary_column, unit, nd, lower_is_better in definitions:
        shot_number = _shot_value(row, raw_columns)
        if title == "좌우 편차" and "AbsTotalSide_m" not in row.index:
            shot_number = abs(shot_number) if not pd.isna(shot_number) else shot_number
        cards.append(
            _shot_compare_card(
                title,
                shot_number,
                day_summary.get(summary_column),
                month_summary.get(summary_column),
                year_summary.get(summary_column),
                unit,
                nd,
                lower_is_better,
            )
        )
    st.markdown("<div class='tm-shot-compare-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)



def _robust_center_scale(series: pd.Series) -> tuple[float | None, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None, None

    center = float(values.median())
    mad = float((values - center).abs().median())
    if mad > 1e-9:
        # MAD -> sigma scale
        return center, mad * 1.4826

    std = float(values.std(ddof=0)) if len(values) >= 2 else 0.0
    return center, max(std, 1e-6)


def _score_near_center(
    value: Any,
    center: float | None,
    scale: float | None,
    *,
    floor_scale: float,
) -> float | None:
    number = _safe_numeric(value)
    if pd.isna(number) or center is None:
        return None
    spread = max(float(scale or 0.0), float(floor_scale), 1e-6)
    z = abs(float(number) - float(center)) / spread
    return max(0.0, min(100.0, 100.0 - z * 22.0))


def _score_lower_is_better(
    value: Any,
    reference: float | None,
    scale: float | None,
    *,
    floor_scale: float,
) -> float | None:
    number = _safe_numeric(value)
    if pd.isna(number):
        return None
    ref = 0.0 if reference is None else float(reference)
    spread = max(float(scale or 0.0), float(floor_scale), 1e-6)
    # reference보다 좋은 값은 100점, 커질수록 완만하게 감점
    excess = max(0.0, abs(float(number)) - abs(ref))
    return max(0.0, min(100.0, 100.0 - (excess / spread) * 24.0))


def _score_higher_is_better(
    value: Any,
    center: float | None,
    scale: float | None,
    *,
    floor_scale: float,
) -> float | None:
    number = _safe_numeric(value)
    if pd.isna(number) or center is None:
        return None
    spread = max(float(scale or 0.0), float(floor_scale), 1e-6)
    # 당일 중앙값 이상이면 충분히 좋은 것으로 보고 큰 추가 가산은 하지 않음
    if float(number) >= float(center):
        return 100.0
    deficit = float(center) - float(number)
    return max(0.0, min(100.0, 100.0 - (deficit / spread) * 24.0))


def _weighted_available_score(parts: list[tuple[float | None, float]]) -> float:
    usable = [(score, weight) for score, weight in parts if score is not None and weight > 0]
    if not usable:
        return 70.0
    total_weight = sum(weight for _, weight in usable)
    return round(sum(float(score) * weight for score, weight in usable) / total_weight, 1)


def _shot_ai_scores(shots: pd.DataFrame, club: str) -> pd.DataFrame:
    """
    개별 샷 품질을 0~100점으로 평가합니다.

    - 절대 핸디캡 점수가 아니라 '같은 날 같은 클럽 안에서의 샷 품질' 점수
    - 웨지는 여러 목표 거리를 섞어 치므로 전체 Carry 중앙값으로 평가하지 않고
      10m 거리 버킷 안에서만 Carry/Launch 안정성을 봅니다.
    """
    scored = shots.copy().reset_index(drop=True)
    if scored.empty:
        scored["AI점수"] = []
        scored["상태"] = []
        return scored

    is_wedge = ("Wedge" in str(club)) or str(club) in {"PitchingWedge", "GapWedge", "SandWedge"}

    stats: dict[str, tuple[float | None, float | None]] = {}
    for column in [
        "Carry_m", "BallSpeed_mps", "SmashFactor", "TotalSide_m",
        "FaceToPath_deg", "LaunchAngle_deg",
    ]:
        if column in scored.columns:
            stats[column] = _robust_center_scale(scored[column])

    # 웨지용 10m 버킷
    if is_wedge and "Carry_m" in scored.columns:
        carries = pd.to_numeric(scored["Carry_m"], errors="coerce")
        scored["_shot_bucket"] = (
            np.floor((carries + 5.0) / 10.0) * 10.0
        )
    else:
        scored["_shot_bucket"] = np.nan

    scores: list[float] = []
    for row_index, row in scored.iterrows():
        side_center, side_scale = stats.get("TotalSide_m", (0.0, None))
        ftp_center, ftp_scale = stats.get("FaceToPath_deg", (0.0, None))
        carry_center, carry_scale = stats.get("Carry_m", (None, None))
        ball_center, ball_scale = stats.get("BallSpeed_mps", (None, None))
        smash_center, smash_scale = stats.get("SmashFactor", (None, None))
        launch_center, launch_scale = stats.get("LaunchAngle_deg", (None, None))

        if is_wedge:
            bucket_value = row.get("_shot_bucket")
            if not pd.isna(bucket_value):
                bucket = scored[scored["_shot_bucket"] == bucket_value]
            else:
                bucket = scored.iloc[0:0]

            if len(bucket) >= 3:
                bucket_carry_center, bucket_carry_scale = _robust_center_scale(bucket["Carry_m"])
                if "LaunchAngle_deg" in bucket.columns:
                    bucket_launch_center, bucket_launch_scale = _robust_center_scale(bucket["LaunchAngle_deg"])
                else:
                    bucket_launch_center, bucket_launch_scale = None, None
            else:
                bucket_carry_center, bucket_carry_scale = None, None
                bucket_launch_center, bucket_launch_scale = None, None

            parts = [
                (
                    _score_lower_is_better(
                        row.get("TotalSide_m"),
                        0.0,
                        side_scale,
                        floor_scale=4.0,
                    ),
                    0.45,
                ),
                (
                    _score_lower_is_better(
                        row.get("FaceToPath_deg"),
                        0.0,
                        ftp_scale,
                        floor_scale=3.0,
                    ),
                    0.20,
                ),
                (
                    _score_near_center(
                        row.get("Carry_m"),
                        bucket_carry_center,
                        bucket_carry_scale,
                        floor_scale=2.5,
                    ),
                    0.20,
                ),
                (
                    _score_near_center(
                        row.get("LaunchAngle_deg"),
                        bucket_launch_center,
                        bucket_launch_scale,
                        floor_scale=2.5,
                    ),
                    0.15,
                ),
            ]
        else:
            parts = [
                (
                    _score_near_center(
                        row.get("Carry_m"),
                        carry_center,
                        carry_scale,
                        floor_scale=5.0,
                    ),
                    0.25,
                ),
                (
                    _score_lower_is_better(
                        row.get("TotalSide_m"),
                        0.0,
                        side_scale,
                        floor_scale=6.0,
                    ),
                    0.30,
                ),
                (
                    _score_higher_is_better(
                        row.get("SmashFactor"),
                        smash_center,
                        smash_scale,
                        floor_scale=0.035,
                    ),
                    0.20,
                ),
                (
                    _score_higher_is_better(
                        row.get("BallSpeed_mps"),
                        ball_center,
                        ball_scale,
                        floor_scale=1.8,
                    ),
                    0.15,
                ),
                (
                    _score_lower_is_better(
                        row.get("FaceToPath_deg"),
                        0.0,
                        ftp_scale,
                        floor_scale=3.0,
                    ),
                    0.10,
                ),
            ]

        scores.append(_weighted_available_score(parts))

    scored["AI점수"] = [int(round(value)) for value in scores]

    def status_from_score(value: int) -> str:
        if value >= 85:
            return "Excellent"
        if value >= 70:
            return "Good"
        if value >= 50:
            return "Poor"
        return "Miss"

    scored["상태"] = scored["AI점수"].map(status_from_score)
    return scored


def _shot_status_badge(score: int, status: str) -> str:
    icon = {
        "Excellent": "🟢",
        "Good": "🟡",
        "Poor": "🟠",
        "Miss": "🔴",
    }.get(status, "⚪")
    return f"{icon} {status}"


def _render_clickable_shot_distribution(
    shots: pd.DataFrame,
    selected_index: int,
    *,
    key: str,
    distance_metric: str = "Carry_m",
) -> int:
    """
    Plotly 클릭 선택형 탄착군.

    - 모든 실제 샷은 단일 trace를 유지해 shot_index 연동 안정성 보존
    - 선택 샷은 오렌지 중심점 + 2중 halo/glow + Shot 라벨로 강조
    - 0m 목표선은 노란 점선, 오늘 평균선은 초록 점선
    - 다크 대시보드와 동일한 배경/그리드 사용
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning(
            "탄착군 클릭 기능을 사용하려면 plotly가 필요합니다. "
            "`pip install plotly` 후 다시 실행하세요."
        )
        _render_blinking_direction_distribution(shots, selected_index, distance_metric)
        return selected_index

    y_metric = distance_metric if distance_metric in shots.columns else "Carry_m"
    if "TotalSide_m" not in shots.columns or y_metric not in shots.columns:
        st.info("탄착군 데이터가 없습니다.")
        return selected_index

    work = shots.copy().reset_index(drop=True)
    work["_side"] = pd.to_numeric(work["TotalSide_m"], errors="coerce")
    work["_distance"] = pd.to_numeric(work[y_metric], errors="coerce")
    work["_shot_index"] = np.arange(len(work), dtype=int)
    work = work.dropna(subset=["_side", "_distance"]).reset_index(drop=True)

    if work.empty:
        st.info("탄착군 데이터가 없습니다.")
        return selected_index

    labels = [
        f"Shot {_shot_display_number(shots.iloc[int(idx)], int(idx))}"
        for idx in work["_shot_index"]
    ]

    # 실제 샷 trace. 선택 샷도 여기 안에 그대로 존재하므로 클릭 mapping은 변하지 않습니다.
    marker_sizes = [
        12 if int(idx) == int(selected_index) else 9
        for idx in work["_shot_index"]
    ]
    marker_colors = [
        "#FF8A32" if int(idx) == int(selected_index) else "#3D94FF"
        for idx in work["_shot_index"]
    ]
    marker_line_widths = [
        2 if int(idx) == int(selected_index) else 0
        for idx in work["_shot_index"]
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["_side"],
            y=work["_distance"],
            mode="markers",
            marker={
                "size": marker_sizes,
                "color": marker_colors,
                "opacity": 0.90,
                "line": {
                    "width": marker_line_widths,
                    "color": "#FFF2D8",
                },
            },
            customdata=work["_shot_index"].astype(int).tolist(),
            text=labels,
            hovertemplate=(
                "%{text}<br>"
                "좌우 %{x:.1f}m<br>"
                "거리 %{y:.1f}m"
                "<extra></extra>"
            ),
            name="샷",
        )
    )

    # 선택 샷 static glow/halo.
    # CSS pulse 대신 Plotly 자체 trace로 만들어 Streamlit rerun/iframe 영향을 받지 않습니다.
    selected_point = work[work["_shot_index"] == selected_index]
    if not selected_point.empty:
        selected_x = float(selected_point.iloc[0]["_side"])
        selected_y = float(selected_point.iloc[0]["_distance"])
        selected_shot_no = _shot_display_number(shots.iloc[selected_index], selected_index)

        # 바깥 halo
        fig.add_trace(
            go.Scatter(
                x=[selected_x],
                y=[selected_y],
                mode="markers",
                marker={
                    "size": 32,
                    "color": "rgba(255,138,50,0.10)",
                    "line": {"width": 2, "color": "rgba(255,138,50,0.28)"},
                },
                customdata=[selected_index],
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # 안쪽 glow ring + Shot 라벨
        fig.add_trace(
            go.Scatter(
                x=[selected_x],
                y=[selected_y],
                mode="markers+text",
                marker={
                    "size": 22,
                    "color": "rgba(255,138,50,0.04)",
                    "line": {"width": 3, "color": "#FFD54F"},
                },
                text=[f"Shot {selected_shot_no}"],
                textposition="top center",
                textfont={"color": "#FFD54F", "size": 11},
                customdata=[selected_index],
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # 중앙 목표선(0m)
    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="dash",
        line_color="#FFD54F",
        opacity=0.85,
    )

    # 오늘 평균 좌우 편차선
    avg_side = float(work["_side"].mean())
    fig.add_vline(
        x=avg_side,
        line_width=2,
        line_dash="dot",
        line_color="#67CF45",
        opacity=0.90,
        annotation_text=f"평균 {avg_side:+.1f}m",
        annotation_position="top",
        annotation_font_color="#67CF45",
    )

    fig.update_layout(
        height=335,
        margin={"l": 20, "r": 15, "t": 26, "b": 25},
        paper_bgcolor="#0D1721",
        plot_bgcolor="#0D1721",
        font={"color": "#DCE8F5"},
        xaxis={
            "title": "좌우 편차 (m)",
            "gridcolor": "#263548",
            "zeroline": False,
            "tickfont": {"color": "#AAB7C7"},
            "title_font": {"color": "#C9D5E3"},
        },
        yaxis={
            "title": "캐리 (m)" if y_metric == "Carry_m" else "토탈 (m)",
            "gridcolor": "#263548",
            "zeroline": False,
            "tickfont": {"color": "#AAB7C7"},
            "title_font": {"color": "#C9D5E3"},
        },
        showlegend=False,
        dragmode=False,
        clickmode="event+select",
    )

    event = st.plotly_chart(
        fig,
        width="stretch",
        key=f"{key}::{selected_index}",
        on_select="rerun",
        selection_mode="points",
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
        },
    )

    selection = None
    try:
        selection = event.selection
    except Exception:
        try:
            selection = event.get("selection")
        except Exception:
            selection = None

    points = []
    if selection is not None:
        try:
            points = list(selection.points or [])
        except Exception:
            try:
                points = list(selection.get("points", []) or [])
            except Exception:
                points = []

    if not points:
        return selected_index

    point = points[-1]

    def _point_value(obj, name):
        try:
            value = getattr(obj, name)
            if value is not None:
                return value
        except Exception:
            pass
        try:
            return obj.get(name)
        except Exception:
            return None

    customdata = _point_value(point, "customdata")
    clicked_index = None

    if customdata is not None:
        try:
            if isinstance(customdata, (list, tuple, np.ndarray)):
                if len(customdata):
                    clicked_index = int(customdata[0])
            else:
                clicked_index = int(customdata)
        except Exception:
            clicked_index = None

    # 메인 trace 클릭의 fallback
    if clicked_index is None:
        for field in ("point_index", "point_number", "pointIndex", "pointNumber"):
            raw = _point_value(point, field)
            if raw is None:
                continue
            try:
                plot_index = int(raw)
                if 0 <= plot_index < len(work):
                    clicked_index = int(work.iloc[plot_index]["_shot_index"])
                    break
            except Exception:
                continue

    if clicked_index is None:
        return selected_index

    return max(0, min(int(clicked_index), len(shots) - 1))

def _shot_table(
    df_shots: pd.DataFrame,
    selected_index: int,
    key: str,
    *,
    sort_by_ai_low: bool = False,
) -> int:
    """AG Grid 기반 다크 샷 목록. 행 클릭 시 상세 샷을 변경합니다."""
    scored = _shot_ai_scores(
        df_shots,
        str(df_shots.iloc[0].get("Club", "")) if not df_shots.empty else "",
    )

    table_columns = [
        "StrokeNo", "ShotTimeLocal", "AI점수", "상태",
        "Carry_m", "Total_m", "BallSpeed_mps", "ClubSpeed_mps",
        "SmashFactor", "SpinRate_rpm", "LaunchAngle_deg", "AttackAngle_deg",
        "ClubPath_deg", "FaceAngle_deg", "FaceToPath_deg", "TotalSide_m",
        "ImpactOffset_mm", "ImpactHeight_mm",
    ]
    available = [column for column in table_columns if column in scored.columns]

    table_df = scored.loc[:, available].copy()
    table_df["_shot_index"] = np.arange(len(table_df), dtype=int)

    if "상태" in table_df.columns:
        table_df["상태"] = [
            _shot_status_badge(int(score), str(status))
            for score, status in zip(table_df["AI점수"], table_df["상태"])
        ]

    if sort_by_ai_low and "AI점수" in table_df.columns:
        table_df = table_df.sort_values(
            ["AI점수", "_shot_index"],
            ascending=[True, True],
            kind="stable",
        ).reset_index(drop=True)
    else:
        table_df = table_df.reset_index(drop=True)

    # 선택된 실제 shot_index가 현재 정렬된 grid에서 몇 번째 행인지 계산
    selected_display_rows = table_df.index[
        table_df["_shot_index"].astype(int) == int(selected_index)
    ].tolist()
    pre_selected_rows = selected_display_rows[:1]

    grid_df = safe_dataframe_for_streamlit(table_df)
    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(
        editable=False,
        sortable=True,
        filter=False,
        resizable=True,
        minWidth=86,
    )
    gb.configure_selection(
        selection_mode="single",
        use_checkbox=False,
        pre_selected_rows=pre_selected_rows,
    )
    gb.configure_grid_options(
        rowHeight=35,
        headerHeight=38,
        suppressCellFocus=True,
        animateRows=False,
    )
    gb.configure_column("_shot_index", hide=True)

    # 사용성이 좋은 기본 폭
    widths = {
        "StrokeNo": 78,
        "ShotTimeLocal": 92,
        "AI점수": 76,
        "상태": 105,
        "SmashFactor": 92,
        "SpinRate_rpm": 105,
    }
    for column, width in widths.items():
        if column in grid_df.columns:
            gb.configure_column(column, width=width)

    # 전체 대시보드와 동일한 dark navy theme
    grid_css = {
        ".ag-root-wrapper": {
            "background-color": "#0D1721 !important",
            "border": "1px solid #263548 !important",
            "border-radius": "10px !important",
            "overflow": "hidden !important",
        },
        ".ag-root-wrapper-body": {"background-color": "#0D1721 !important"},
        ".ag-header": {
            "background-color": "#142232 !important",
            "border-bottom": "1px solid #33465E !important",
        },
        ".ag-header-cell": {
            "background-color": "#142232 !important",
            "color": "#C9D5E3 !important",
            "border-right": "1px solid #263548 !important",
            "font-weight": "700 !important",
        },
        ".ag-header-cell-text": {"color": "#C9D5E3 !important"},
        ".ag-body-viewport": {"background-color": "#0D1721 !important"},
        ".ag-center-cols-viewport": {"background-color": "#0D1721 !important"},
        ".ag-row": {
            "background-color": "#0D1721 !important",
            "color": "#E8EEF6 !important",
            "border-bottom": "1px solid #1D2B3B !important",
        },
        ".ag-row-even": {"background-color": "#101C29 !important"},
        ".ag-row-hover": {"background-color": "#17283A !important"},
        ".ag-row-selected": {
            "background-color": "#17395D !important",
            "box-shadow": "inset 4px 0 0 #FF6B1A !important",
        },
        ".ag-cell": {
            "color": "#E8EEF6 !important",
            "border-right": "1px solid #223044 !important",
        },
        ".ag-row-selected .ag-cell": {
            "color": "#FFFFFF !important",
            "font-weight": "700 !important",
        },
        ".ag-cell-focus": {"border-color": "transparent !important"},
        ".ag-body-horizontal-scroll": {"background-color": "#0D1721 !important"},
        ".ag-body-horizontal-scroll-viewport": {"background-color": "#0D1721 !important"},
        ".ag-horizontal-left-spacer": {"background-color": "#0D1721 !important"},
        ".ag-horizontal-right-spacer": {"background-color": "#0D1721 !important"},
    }

    response = AgGrid(
        grid_df,
        gridOptions=gb.build(),
        height=min(580, max(280, 46 + 35 * min(len(grid_df), 14))),
        theme="streamlit",
        custom_css=grid_css,
        update_on=["selectionChanged"],
        allow_unsafe_jscode=False,
        enable_enterprise_modules=False,
        fit_columns_on_grid_load=False,
        key=key,
    )

    # streamlit-aggrid 1.x 반환 객체/딕셔너리 모두 처리
    selected_rows = None
    try:
        selected_rows = response.selected_rows
    except Exception:
        try:
            selected_rows = response.get("selected_rows")
        except Exception:
            selected_rows = None

    if selected_rows is None:
        return selected_index

    try:
        if isinstance(selected_rows, pd.DataFrame):
            if selected_rows.empty:
                return selected_index
            raw_index = selected_rows.iloc[-1].get("_shot_index")
        elif isinstance(selected_rows, list):
            if not selected_rows:
                return selected_index
            raw_index = selected_rows[-1].get("_shot_index")
        else:
            selected_frame = pd.DataFrame(selected_rows)
            if selected_frame.empty:
                return selected_index
            raw_index = selected_frame.iloc[-1].get("_shot_index")

        if raw_index is None or pd.isna(raw_index):
            return selected_index
        return max(0, min(int(raw_index), len(df_shots) - 1))
    except Exception:
        return selected_index

def _club_korean_name(club: str) -> str:
    """내부 클럽 코드를 화면용 한글 클럽명으로 변환합니다."""
    mapping = {
        "Driver": "드라이버",
        "3Wood": "3번 우드",
        "5Wood": "5번 우드",
        "7Wood": "7번 우드",
        "4Hybrid": "4번 유틸리티",
        "5Hybrid": "5번 유틸리티",
        "4Iron": "4번 아이언",
        "5Iron": "5번 아이언",
        "6Iron": "6번 아이언",
        "7Iron": "7번 아이언",
        "8Iron": "8번 아이언",
        "9Iron": "9번 아이언",
        "PitchingWedge": "피칭 웨지",
        "50Wedge": "50도 웨지",
        "52Wedge": "52도 웨지",
        "56Wedge": "56도 웨지",
        "SandWedge": "샌드 웨지",
    }
    return mapping.get(str(club), str(club))


def _render_single_shot_distance_distribution(shots: pd.DataFrame, selected_index: int, metric: str) -> None:
    """당일 샷의 거리 분포를 평균 분석과 같은 부드러운 곡선으로 표시합니다."""
    label = "Carry" if metric == "Carry_m" else "Total"
    if metric not in shots.columns:
        st.info(f"{label} 데이터가 없습니다.")
        return

    values = pd.to_numeric(shots[metric], errors="coerce").dropna()
    if values.empty:
        st.info(f"{label} 데이터가 없습니다.")
        return

    selected_value = _safe_numeric(shots.iloc[selected_index].get(metric))
    avg = float(values.mean())
    spread = float(values.std(ddof=0)) if len(values) >= 2 else 0.0
    bandwidth = max(spread, 2.0)

    xmin = float(values.min())
    xmax = float(values.max())
    if not pd.isna(selected_value):
        xmin = min(xmin, float(selected_value))
        xmax = max(xmax, float(selected_value))
    padding = max(8.0, (xmax - xmin) * 0.18)
    xs = np.linspace(xmin - padding, xmax + padding, 260)

    # 샷별 가우시안 커널을 합쳐 작은 표본에서도 자연스러운 KDE 곡선을 만듭니다.
    density = np.zeros_like(xs, dtype=float)
    for value in values.to_numpy(dtype=float):
        density += np.exp(-0.5 * ((xs - value) / bandwidth) ** 2)
    density /= max(len(values) * bandwidth, 1e-9)

    fig, ax = plt.subplots(figsize=(6.1, 3.2), dpi=160)
    fig.patch.set_facecolor('#101b27')
    ax.set_facecolor('#101b27')
    ax.plot(xs, density, color='#3d94ff', lw=2.5, label=f'당일 {label}')
    ax.fill_between(xs, density, 0, color='#3d94ff', alpha=.12)
    ax.axvline(avg, color='#67cf45', ls='--', lw=1.8, label=f'당일 평균 {avg:.0f}m')
    if not pd.isna(selected_value):
        ax.axvline(selected_value, color='#ff7a29', lw=2.8, label=f'선택 샷 {selected_value:.0f}m')

    ax.set_xlabel(f'{label} (m)', color='#aab7c7')
    ax.set_yticks([])
    ax.tick_params(colors='#aab7c7', labelsize=8)
    ax.grid(axis='x', color='#263548', alpha=.25)
    ax.legend(frameon=False, labelcolor='#d8e3ee', fontsize=8, loc='upper left')
    for spine in ax.spines.values():
        spine.set_color('#263548')
    st.pyplot(fig, clear_figure=True)


def _render_blinking_direction_distribution(
    shots: pd.DataFrame,
    selected_index: int,
    distance_metric: str = "Carry_m",
) -> None:
    """당일 전체 샷과 선택 샷을 좌우 편차/거리 탄착군으로 표시합니다."""
    if "TotalSide_m" not in shots.columns:
        st.info("방향성 데이터가 없습니다.")
        return

    work = shots.copy()
    work["TotalSide_m"] = pd.to_numeric(work["TotalSide_m"], errors="coerce")
    requested_metric = distance_metric if distance_metric in {"Carry_m", "Total_m"} else "Carry_m"
    y_metric = requested_metric if requested_metric in work.columns else ("Carry_m" if "Carry_m" in work.columns else None)

    if y_metric:
        work[y_metric] = pd.to_numeric(work[y_metric], errors="coerce")
        work = work.dropna(subset=["TotalSide_m", y_metric])
    else:
        work = work.dropna(subset=["TotalSide_m"])
        work["_y"] = np.arange(len(work), dtype=float)
        y_metric = "_y"

    if work.empty:
        st.info("방향성 데이터가 없습니다.")
        return

    selected_original_index = shots.index[selected_index]

    width, height = 700, 330
    pad_l, pad_r, pad_t, pad_b = 62, 24, 28, 52

    max_abs_side = max(10.0, float(work["TotalSide_m"].abs().max()) + 3.0)
    axis_limit = max(10, int(math.ceil(max_abs_side / 5.0) * 5))
    xmin, xmax = -float(axis_limit), float(axis_limit)

    ymin = float(work[y_metric].min())
    ymax = float(work[y_metric].max())
    y_padding = max(4.0, (ymax - ymin) * 0.12)
    if abs(ymax - ymin) < 1e-9:
        ymin -= 1.0
        ymax += 1.0
    else:
        ymin -= y_padding
        ymax += y_padding

    def sx(x: float) -> float:
        return pad_l + (float(x) - xmin) / (xmax - xmin) * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return pad_t + (ymax - float(y)) / (ymax - ymin) * (height - pad_t - pad_b)

    # Golfzon 분포도처럼 중앙을 기준으로 5m 간격 세로 기준선과 수치 라벨을 표시합니다.
    grid_lines: list[str] = []
    grid_labels: list[str] = []
    for side_value in range(-axis_limit, axis_limit + 1, 5):
        x = sx(side_value)
        if side_value == 0:
            stroke = "#dbe5ef"
            opacity = ".82"
            dash = ""
            line_width = "1.5"
        else:
            stroke = "#4a5a6d"
            opacity = ".58"
            dash = "stroke-dasharray='4 4'"
            line_width = "1"

        grid_lines.append(
            f"<line x1='{x:.1f}' y1='{pad_t}' x2='{x:.1f}' y2='{height-pad_b}' "
            f"stroke='{stroke}' stroke-width='{line_width}' opacity='{opacity}' {dash}/>"
        )

        if side_value < 0:
            label = f"{abs(side_value)}L"
            fill = "#ff6259"
        elif side_value > 0:
            label = f"{side_value}R"
            fill = "#4aa3ff"
        else:
            label = "0"
            fill = "#dbe5ef"

        grid_labels.append(
            f"<text x='{x:.1f}' y='{height-17}' fill='{fill}' font-size='11.5' "
            f"font-weight='700' text-anchor='middle'>{label}</text>"
        )

    # 거리축은 실제 m 값을 읽을 수 있도록 보기 좋은 간격의 가로선과 라벨을 표시합니다.
    horizontal_lines: list[str] = []
    distance_span = max(ymax - ymin, 1.0)
    raw_step = distance_span / 5.0
    if raw_step <= 5:
        y_step = 5
    elif raw_step <= 10:
        y_step = 10
    elif raw_step <= 20:
        y_step = 20
    elif raw_step <= 25:
        y_step = 25
    else:
        y_step = 50
    first_tick = int(math.floor(ymin / y_step) * y_step)
    last_tick = int(math.ceil(ymax / y_step) * y_step)
    for distance_value in range(first_tick, last_tick + y_step, y_step):
        if distance_value < ymin or distance_value > ymax:
            continue
        y = sy(distance_value)
        horizontal_lines.append(
            f"<line x1='{pad_l}' y1='{y:.1f}' x2='{width-pad_r}' y2='{y:.1f}' "
            "stroke='#314154' stroke-width='1' opacity='.50'/>"
            f"<text x='{pad_l-8}' y='{y+4:.1f}' fill='#aab7c7' font-size='11' "
            f"text-anchor='end'>{distance_value}</text>"
        )

    circles: list[str] = []
    selected_svg = ""
    for idx, row in work.iterrows():
        cx, cy = sx(row["TotalSide_m"]), sy(row[y_metric])
        if idx == selected_original_index:
            selected_svg = (
                f"<circle class='tm-blink-shot' cx='{cx:.1f}' cy='{cy:.1f}' r='9'/>"
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4.5' fill='#fff4df'/>"
            )
        else:
            circles.append(
                f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='4.6' "
                "fill='#4aa3ff' fill-opacity='.68'/>"
            )

    avg_side = float(work["TotalSide_m"].mean())
    avg_x = sx(avg_side)
    avg_label = f"{abs(avg_side):.1f}{'R' if avg_side > 0.05 else 'L' if avg_side < -0.05 else ''}"

    svg = f"""
    <style>
    @keyframes tmShotBlink {{
      0%,100% {{ r:8; opacity:1; stroke-width:3; }}
      50% {{ r:16; opacity:.25; stroke-width:7; }}
    }}
    .tm-blink-shot {{
      fill:#ff7a29;
      stroke:#ffd2ad;
      animation:tmShotBlink 1.05s ease-in-out infinite;
      transform-origin:center;
    }}
    </style>
    <div style='border:1px solid #263548;border-radius:12px;background:#101b27;padding:6px 8px 2px;overflow-x:auto'>
      <svg viewBox='0 0 {width} {height}' width='100%' style='min-width:360px;display:block'>
        <rect x='0' y='0' width='{width}' height='{height}' fill='#101b27'/>
        {''.join(horizontal_lines)}
        {''.join(grid_lines)}
        <line x1='{avg_x:.1f}' y1='{pad_t}' x2='{avg_x:.1f}' y2='{height-pad_b}'
              stroke='#67cf45' stroke-width='1.5' stroke-dasharray='5 4' opacity='.92'/>
        <line x1='{pad_l}' y1='{height-pad_b}' x2='{width-pad_r}' y2='{height-pad_b}' stroke='#39495c'/>
        {''.join(circles)}
        {selected_svg}
        {''.join(grid_labels)}
        <text x='{pad_l}' y='{pad_t+14}' fill='#cbd6e5' font-size='12'>{'캐리' if y_metric == 'Carry_m' else '토탈'} 거리 (m)</text>
        <text x='{avg_x+5:.1f}' y='{pad_t+29}' fill='#67cf45' font-size='12' font-weight='700'>평균 {avg_label}</text>
        <text x='{width/2:.1f}' y='{height-2}' fill='#9fb0c2' font-size='12' text-anchor='middle'>좌우 편차 (m)</text>
      </svg>
    </div>
    """
    st.markdown(svg, unsafe_allow_html=True)

def _app_secret(name: str, default: str = "") -> str:
    """Streamlit Secrets를 우선 사용하고, 없으면 환경변수를 사용합니다."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.getenv(name, default)
    return str(value or "").strip()


storage = TrackmanStorage(
    supabase_url=_app_secret("SUPABASE_URL"),
    supabase_key=_app_secret("SUPABASE_KEY"),
    bucket=_app_secret("SUPABASE_BUCKET", "trackman-reports"),
)



def _sync_journal_database_after_cloud_pull():
    """
    Storage에서 내려받은 JSON을 DODOS DB/AI까지 즉시 연결합니다.
    최신 데이터 버튼과 자동 cloud refresh가 같은 경로를 사용합니다.
    """
    return sync_journal_db_after_pull(
        project_dir=APP_DIR,
        user_email=AUTH_EMAIL,
        secrets={
            "SUPABASE_URL": _app_secret("SUPABASE_URL"),
            "SUPABASE_KEY": _app_secret("SUPABASE_KEY"),
            "SUPABASE_SERVICE_ROLE_KEY": _app_secret("SUPABASE_SERVICE_ROLE_KEY"),
            "SUPABASE_BUCKET": _app_secret("SUPABASE_BUCKET", "trackman-reports"),
            "DODOS_USER_EMAIL": _app_secret("DODOS_USER_EMAIL", AUTH_EMAIL),
        },
    )

# 앱이 다시 실행될 때마다 Supabase 상태를 확인합니다.
# Streamlit Cloud의 로컬 파일은 앱 프로세스가 살아 있는 동안 유지되므로,
# 브라우저 새로고침만으로는 Mac에서 새로 업로드한 파일이 자동 복원되지 않습니다.
storage_status = storage.status(check_cloud=True)

local_report_count = len(storage.report_files())
cloud_report_count = storage_status.cloud_report_count

if (
    storage.cloud_configured
    and storage_status.cloud_connected
    and cloud_report_count is not None
    and cloud_report_count > local_report_count
):
    with st.spinner("Supabase의 최신 TrackMan 데이터를 반영하는 중입니다..."):
        restore_result = storage.pull_cloud_reports()

    if restore_result.downloaded:
        storage.invalidate_cache()
        st.cache_data.clear()
        storage.write_last_sync(
            source="supabase_auto_refresh",
            details={"downloaded": restore_result.downloaded},
        )

        # 신규 TrackMan JSON을 받았다면 연습일지 DB/AI도 같은 시점에 동기화
        db_sync_result = _sync_journal_database_after_cloud_pull()
        if not db_sync_result.ok:
            st.warning(
                "최신 TrackMan 데이터는 반영됐지만 "
                f"연습일지 DB 동기화에 실패했습니다: {db_sync_result.message}"
            )

        storage_status = storage.status(check_cloud=True)

st.sidebar.markdown("## Trackman 분석")
st.sidebar.caption("by seongcheoll.kim")
st.sidebar.divider()

st.sidebar.markdown("### 👤 로그인 사용자")
st.sidebar.caption(AUTH_NAME)
st.sidebar.caption(AUTH_EMAIL)
if st.sidebar.button("로그아웃", width="stretch", type="primary"):
    st.logout()

st.sidebar.divider()
st.sidebar.metric("저장된 연습", f"{storage_status.report_count}회")
if storage_status.last_sync is not None:
    st.sidebar.caption(f"마지막 동기화: {storage_status.last_sync.astimezone().strftime('%Y-%m-%d %H:%M')}")
else:
    st.sidebar.caption("마지막 동기화: 없음")

if not storage_status.cloud_configured:
    st.sidebar.warning("☁️ Supabase 설정 필요")
elif storage_status.cloud_connected:
    cloud_sessions = storage_status.cloud_report_count or 0
    cloud_shots = storage_status.cloud_shot_count
    cloud_label = f"☁️ Supabase 연결됨 · {cloud_sessions}회"
    if cloud_shots is not None:
        cloud_label += f" · {cloud_shots:,}샷"
    st.sidebar.success(cloud_label)
    if storage_status.cloud_updated_at is not None:
        st.sidebar.caption(
            "클라우드 마지막 백업: "
            + storage_status.cloud_updated_at.astimezone().strftime("%Y-%m-%d %H:%M")
        )
else:
    st.sidebar.error("☁️ Supabase 연결 실패")
    if storage_status.cloud_error:
        with st.sidebar.expander("Supabase 오류"):
            st.code(storage_status.cloud_error[-2000:])

if st.sidebar.button("☁️ 최신 데이터 불러오기", width="stretch", type="primary"):
    with st.spinner("Supabase의 최신 TrackMan 데이터를 불러오는 중입니다..."):
        pull_result = storage.pull_cloud_reports()

    if pull_result.ok:
        storage.invalidate_cache()
        storage.write_last_sync(
            source="supabase_manual_refresh",
            details={"downloaded": pull_result.downloaded},
        )
        st.cache_data.clear()

        with st.spinner("연습일지 DB와 AI 분석을 동기화하는 중입니다..."):
            db_sync_result = _sync_journal_database_after_cloud_pull()

        if db_sync_result.ok:
            st.sidebar.success(
                f"최신 데이터 반영 완료 · 신규 {pull_result.downloaded}회 · 연습일지 동기화 완료"
            )
            st.rerun()
        else:
            st.sidebar.error(
                "TrackMan 데이터는 반영됐지만 연습일지 DB 동기화에 실패했습니다."
            )
            with st.sidebar.expander("연습일지 동기화 오류"):
                st.code(
                    (
                        db_sync_result.message
                        + "\n\n"
                        + (db_sync_result.stderr or db_sync_result.stdout)[-4000:]
                    ).strip()
                )
    else:
        st.sidebar.error("Supabase 데이터 불러오기에 실패했습니다.")
        with st.sidebar.expander("오류 내용"):
            st.code("\n".join(pull_result.errors)[-4000:])

uploaded = []

with st.sidebar.expander("⋯ 더보기", expanded=False):
    st.caption("필요할 때만 사용하는 데이터 관리 기능입니다.")

    direct_sync_required = [
        APP_DIR / "activity_list.curl",
        APP_DIR / "activity_report.curl",
        APP_DIR / "download_all_trackman_reports.py",
        APP_DIR / "trackman_auth_refresh.py",
    ]
    direct_sync_available = all(path.exists() for path in direct_sync_required)

    if direct_sync_available:
        if st.button("🖥️ 이 Mac에서 TrackMan 직접 동기화", width="stretch"):
            with st.spinner("TrackMan 데이터를 수집하고 Supabase에 백업하는 중입니다..."):
                sync_result = sync_trackman_reports(storage=storage)

            if sync_result.ok:
                cloud_uploaded = sync_result.cloud.uploaded if sync_result.cloud else 0
                storage.invalidate_cache()
                st.cache_data.clear()
                st.success(
                    f"직접 동기화 완료 · 신규 {sync_result.downloaded_count}회 "
                    f"· 클라우드 {cloud_uploaded}회"
                )
                st.rerun()
            else:
                st.error("TrackMan 직접 동기화에 실패했습니다.")
                with st.expander("오류 내용"):
                    st.code((sync_result.stderr or sync_result.stdout)[-4000:])
    else:
        st.caption(
            "TrackMan 원본 수집은 Mac의 LaunchAgent가 담당합니다. "
            "웹앱에서는 위의 ‘최신 데이터 불러오기’를 사용하세요."
        )

    if st.button("⬆️ 로컬 데이터 백업", width="stretch", disabled=not storage.cloud_configured):
        with st.spinner("로컬 보고서를 Supabase에 백업하는 중입니다..."):
            upload_result = storage.upload_local_reports()
        if upload_result.ok:
            storage.write_last_sync(source="supabase_backup", details={"uploaded": upload_result.uploaded})
            st.success(f"백업 완료 · 신규 {upload_result.uploaded}회 · 기존 {upload_result.skipped}회")
            st.rerun()
        else:
            st.error("Supabase 백업에 실패했습니다.")
            with st.expander("오류 내용"):
                st.code("\n".join(upload_result.errors)[-4000:])

    if st.button("↻ 로컬 캐시 새로고침", width="stretch"):
        storage.invalidate_cache()
        st.cache_data.clear()
        st.rerun()

    st.markdown("##### JSON/CSV 직접 추가")
    uploaded = st.file_uploader(
        "파일 추가",
        type=["json", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="hidden_manual_upload",
    )
    json_uploads = [f for f in (uploaded or []) if f.name.lower().endswith((".json", ".txt"))]
    if st.button("선택한 JSON 영구 저장", width="stretch", disabled=not json_uploads):
        saved, errors = 0, []
        for file in json_uploads:
            try:
                storage.save_uploaded_json(file.name, file.getvalue(), upload_cloud=True)
                saved += 1
            except Exception as exc:
                errors.append(f"{file.name}: {exc}")
        if saved:
            storage.write_last_sync(source="manual_upload", details={"saved": saved})
            st.success(f"{saved}개 파일을 영구 저장했습니다.")
            st.cache_data.clear()
        for error in errors:
            st.error(error)
        if saved and not errors:
            st.rerun()

rows, load_errors = storage.load_rows(parse_trackman_report)
uploaded_rows, uploaded_errors = load_uploaded_files(uploaded)
rows.extend(uploaded_rows)
load_errors.extend(uploaded_errors)
for err in load_errors:
    st.warning(err)
if not rows:
    st.info("저장된 TrackMan 데이터가 없습니다. 사이드바에서 동기화하거나 JSON 파일을 영구 저장해 주세요.")
    st.stop()

df=prepare_df(rows); clubs=sorted(df['Club'].dropna().unique().tolist(),key=club_sort_key); dates=sorted(df['Date'].dropna().unique().tolist())

club=st.sidebar.selectbox('1. 클럽 선택',clubs,index=0)
club_dates=sorted(df[df['Club']==club]['Date'].dropna().unique().tolist(), reverse=True)
selected_date=st.sidebar.selectbox('분석 날짜',club_dates,index=0)
st.sidebar.markdown('### 2. 비교 기준')
scope=st.sidebar.radio('비교 범위',['월간 평균 + 연간 평균','월간 평균만','연간 평균만'],label_visibility='collapsed')
st.sidebar.markdown('### 3. 평균 방식')
mode=st.sidebar.radio('평균 방식',['연습일 평균','전체 샷 평균'],index=0,label_visibility='collapsed')
st.sidebar.caption('연습일 평균: 날짜별 평균을 동일 가중치로 평균')
st.sidebar.markdown('### 4. 옵션')
exclude=st.sidebar.checkbox('선택일을 비교 평균에서 제외',value=True)

# 사이드바 상세 범위 필터: 범위와 기본값은 현재 선택 클럽 데이터에서 계산합니다.
filtered_df = df.copy()
range_specs = [
    ('Carry_m', '캐리', 1.0),
    ('Total_m', '토탈', 1.0),
    ('BallSpeed_mps', '볼 스피드', 0.5),
    ('ClubSpeed_mps', '클럽 스피드', 0.5),
    ('SpinRate_rpm', '스핀량', 50.0),
    ('LaunchAngle_deg', '발사각', 0.5),
    ('TotalSide_m', '사이드', 1.0),
]
range_version_key = f'range_filter_version::{club}'
if range_version_key not in st.session_state:
    st.session_state[range_version_key] = 0

with st.sidebar.expander('상세 필터 (범위 필터)', expanded=False):
    st.caption(f'{_club_korean_name(club)} 데이터 범위 안의 샷만 분석에 반영됩니다.')

    # 버튼을 슬라이더보다 먼저 처리하고 key 버전을 바꿔 위젯 상태까지 확실히 초기화합니다.
    if st.button('범위 필터 초기화', width='stretch', key=f'reset_ranges::{club}'):
        st.session_state[range_version_key] += 1
        st.rerun()

    # 슬라이더 최소/최대 범위는 현재 선택한 클럽의 전체 기간 데이터만 기준으로 계산합니다.
    # 분석 날짜는 범위 산정에 사용하지 않습니다.
    range_base = df[df['Club'] == club].copy()
    range_basis_label = f"{_club_korean_name(club)} 전체 기간"
    active_ranges: dict[str, tuple[float, float]] = {}
    version = int(st.session_state[range_version_key])

    for column, label, step in range_specs:
        if column not in range_base.columns:
            continue
        bounds = _numeric_range(range_base[column])
        if bounds is None:
            continue
        low, high = bounds
        # Streamlit slider의 부동소수점 경계 오류를 피하도록 step 배수로 여유 있게 정리합니다.
        low = float(np.floor(low / step) * step)
        high = float(np.ceil(high / step) * step)
        if high <= low:
            high = low + step
        st.markdown(
            f"<div class='tm-range-caption'>{label}: "
            f"{low:,.1f} ~ {high:,.1f} · {range_basis_label} 기준</div>",
            unsafe_allow_html=True,
        )
        chosen = st.slider(
            label,
            min_value=low,
            max_value=high,
            value=(low, high),
            step=float(step),
            key=f'range_filter::{club}::{column}::v{version}',
            label_visibility='collapsed',
        )
        active_ranges[column] = chosen

selected_club_mask = filtered_df['Club'] == club
selected_club_filtered = filtered_df[selected_club_mask].copy()
for column, chosen_range in active_ranges.items():
    selected_club_filtered = _apply_numeric_range(selected_club_filtered, column, chosen_range)
filtered_df = pd.concat(
    [filtered_df[~selected_club_mask], selected_club_filtered],
    ignore_index=True,
)

filtered_club_day = filtered_df[(filtered_df['Club'] == club) & (filtered_df['Date'] == selected_date)]
if filtered_club_day.empty:
    st.warning('현재 범위 필터 조건에 해당하는 선택일 샷이 없습니다. 범위를 넓혀 주세요.')

# 이후 모든 분석은 범위 필터가 반영된 데이터프레임을 사용합니다.
day,month,year,mlabel,ylabel=_periods(filtered_df,selected_date,exclude)
ds=_summary(day,club,'전체 샷 평균')
ms=_summary(month,club,mode)
ys=_summary(year,club,mode)
if scope=='월간 평균만':
    ys=pd.Series(dtype='object')
    year=year.iloc[0:0]
if scope=='연간 평균만':
    ms=pd.Series(dtype='object')
    month=month.iloc[0:0]


analysis_tabs = st.tabs(['📊 평균 분석', '🎯 샷별 분석', '🤖 AI 스윙 진단', '📓 연습 일지', '⛳ 라운드 기록'])

with analysis_tabs[0]:
    render_average_analysis(
        club=club,
        selected_date=selected_date,
        month_label=mlabel,
        year_label=ylabel,
        mode=mode,
        day_df=day,
        month_df=month,
        year_df=year,
        day_summary=ds,
        month_summary=ms,
        year_summary=ys,
        filtered_df=filtered_df,
        clubs=clubs,
        dates=dates,
        render_compare_cards=_render_compare_cards,
        club_korean_name=_club_korean_name,
        distance_chart=_distance_chart,
        side_chart=_side_chart,
        trend_chart=_trend,
        impact_face_fig=impact_face_fig,
        club_path_fig=club_path_fig,
        loft_spin_fig=loft_spin_fig,
        period_row=_period_row,
        auto_text=_auto_text,
        make_summary=make_summary,
        summary_columns=SUMMARY_COLUMNS,
        club_sort_key=club_sort_key,
        render_club_cards=render_club_cards,
        safe_dataframe_for_streamlit=safe_dataframe_for_streamlit,
        render_dark_dataframe=render_dark_dataframe,
    )

with analysis_tabs[1]:
    render_single_shot_analysis(
        day_df=day,
        month_df=month,
        year_df=year,
        club=club,
        selected_date=selected_date,
        day_summary=ds,
        month_summary=ms,
        year_summary=ys,
        shot_sort_columns=_shot_sort_columns,
        shot_display_number=_shot_display_number,
        club_korean_name=_club_korean_name,
        render_clickable_shot_distribution=_render_clickable_shot_distribution,
        impact_face_fig=impact_face_fig,
        club_path_fig=club_path_fig,
        loft_spin_fig=loft_spin_fig,
        shot_metric_items=_shot_metric_items,
        render_shot_detail_panel=render_shot_detail_panel,
        shot_table=_shot_table,
        render_shot_compare_cards=_render_shot_compare_cards,
    )


# -----------------------------------------------------------------------------
# DODOS Golf Solution Phase 2 / Step 1 - Rule-based AI Swing Diagnosis
# -----------------------------------------------------------------------------
with analysis_tabs[2]:
    ai_report = render_ai_summary(
        df=df,
        selected_date=selected_date,
    )

    render_ai_club_detail(
        ai_report=ai_report,
        selected_club=club,
        selected_date=selected_date,
    )


# -----------------------------------------------------------------------------
# Phase 3-1 · Practice Journal
# -----------------------------------------------------------------------------
with analysis_tabs[3]:
    render_practice_journal_tab(
        user_email=AUTH_EMAIL,
        initial_date=selected_date,
    )

with analysis_tabs[4]:
    render_round_record(user_email=st.session_state.get("user_email", ""))
