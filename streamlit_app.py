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

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

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
CLUB_ASSET_DIR = APP_DIR / "assets" / "clubs"

CLUB_ASSET_MAP = {
    "Driver": {"face": "qi35_face.png", "side": "qi35_side.svg"},
    "5Wood": {"face": "stealth2_face.png", "side": "stealth2_side.svg"},
    "4Hybrid": {"face": "g430_face.png", "side": "g430_side.svg"},

    "5Iron": {"face": "p790_face.png", "side": "p790_side.svg"},
    "6Iron": {"face": "p790_face.png", "side": "p790_side.svg"},
    "7Iron": {"face": "p790_face.png", "side": "p790_side.svg"},
    "8Iron": {"face": "p790_face.png", "side": "p790_side.svg"},
    "9Iron": {"face": "p790_face.png", "side": "p790_side.svg"},
    "PitchingWedge": {"face": "p790_face.png", "side": "p790_side.svg"},

    "50Wedge": {"face": "zipcore_face.png", "side": "zipcore_side.svg"},
    "56Wedge": {"face": "zipcore_face.png", "side": "zipcore_side.svg"},
    "SandWedge": {"face": "zipcore_face.png", "side": "zipcore_side.svg"},
}


# 사용 중인 실제 클럽군에 맞춘 임팩트 맵 시각화 보정값.
# physical_width_mm / physical_height_mm는 타점 데이터(mm)를 화면 좌표로 옮기기 위한
# 시각화 기준치이며 제조사의 공식 헤드 치수 표기는 아닙니다.
IMPACT_CALIBRATION = {
    "Driver": {
        "model": "TaylorMade Qi35",
        "physical_width_mm": 102.0,
        "physical_height_mm": 56.0,
        "face_center": (0.0, -0.5),
        "face_plot_size": (37.0, 18.0),
        "side_center": (0.0, -0.7),
        "side_plot_size": (35.0, 17.0),
    },
    "5Wood": {
        "model": "TaylorMade Stealth 2 5W",
        "physical_width_mm": 90.0,
        "physical_height_mm": 46.0,
        "face_center": (-0.5, -0.5),
        "face_plot_size": (34.0, 16.0),
        "side_center": (-0.5, -0.8),
        "side_plot_size": (32.0, 15.0),
    },
    "4Hybrid": {
        "model": "PING G430 4H",
        "physical_width_mm": 84.0,
        "physical_height_mm": 43.0,
        "face_center": (-0.5, -0.7),
        "face_plot_size": (31.0, 15.0),
        "side_center": (-0.5, -0.8),
        "side_plot_size": (30.0, 14.0),
    },
    "P790": {
        "model": "TaylorMade P790",
        "physical_width_mm": 80.0,
        "physical_height_mm": 45.0,
        "face_center": (-1.0, -0.4),
        "face_plot_size": (30.0, 17.0),
        "side_center": (-1.0, -0.7),
        "side_plot_size": (29.0, 16.0),
    },
    "ZipCore": {
        "model": "Cleveland ZipCore",
        "physical_width_mm": 79.0,
        "physical_height_mm": 47.0,
        "face_center": (-1.0, -0.2),
        "face_plot_size": (29.5, 17.5),
        "side_center": (-1.0, -0.5),
        "side_plot_size": (28.5, 16.5),
    },
}


def get_impact_calibration(club: str) -> dict:
    if club == "Driver":
        return IMPACT_CALIBRATION["Driver"]
    if club == "5Wood" or "Wood" in str(club):
        return IMPACT_CALIBRATION["5Wood"]
    if club == "4Hybrid" or "Hybrid" in str(club):
        return IMPACT_CALIBRATION["4Hybrid"]
    if club in {"50Wedge", "52Wedge", "56Wedge", "SandWedge"} or "Wedge" in str(club):
        return IMPACT_CALIBRATION["ZipCore"]
    return IMPACT_CALIBRATION["P790"]


def impact_mm_to_plot(
    offset_mm,
    height_mm,
    club: str,
    mode: str = "face",
) -> tuple[float, float]:
    """TrackMan 타점 mm를 선택 클럽 SVG의 실제 페이스 영역에 맞춰 변환."""
    cal = get_impact_calibration(club)

    if mode == "face":
        center_x, center_y = cal["face_center"]
        plot_width, plot_height = cal["face_plot_size"]
    else:
        center_x, center_y = cal["side_center"]
        plot_width, plot_height = cal["side_plot_size"]

    half_width_mm = cal["physical_width_mm"] / 2.0
    half_height_mm = cal["physical_height_mm"] / 2.0

    try:
        ox = 0.0 if pd.isna(offset_mm) else float(offset_mm)
    except Exception:
        ox = 0.0
    try:
        hy = 0.0 if pd.isna(height_mm) else float(height_mm)
    except Exception:
        hy = 0.0

    # 비정상적으로 큰 값은 클럽 페이스 가장자리에서 잘라 표시
    ox = float(np.clip(ox, -half_width_mm, half_width_mm))
    hy = float(np.clip(hy, -half_height_mm, half_height_mm))

    x = center_x + ox * (plot_width / cal["physical_width_mm"])
    y = center_y + hy * (plot_height / cal["physical_height_mm"])
    return x, y


# 승인된 렌더 이미지에서 "실제 페이스 중심"이 이미지 캔버스 중앙과 다르기 때문에
# 클럽별로 페이스 중심 위치를 보정한다.
#
# face_center_x / face_center_y:
#   이미지의 왼쪽/아래쪽을 0, 오른쪽/위쪽을 1로 본 정규화 좌표.
# display_width:
#   대시보드 좌표계에서 렌더 이미지가 차지할 가로 폭.
CLUB_RENDER_LAYOUT = {
    "Driver": {
        "face_center_x": 0.43,
        "face_center_y": 0.43,
        "display_width": 47.0,
    },
    "5Wood": {
        "face_center_x": 0.45,
        "face_center_y": 0.43,
        "display_width": 46.0,
    },
    "4Hybrid": {
        "face_center_x": 0.44,
        "face_center_y": 0.43,
        "display_width": 43.0,
    },
    "P790": {
        "face_center_x": 0.37,
        "face_center_y": 0.41,
        "display_width": 45.0,
    },
    "ZipCore": {
        "face_center_x": 0.38,
        "face_center_y": 0.41,
        "display_width": 44.0,
    },
}


def get_render_layout(club: str) -> dict:
    if club == "Driver":
        return CLUB_RENDER_LAYOUT["Driver"]
    if club == "5Wood" or "Wood" in str(club):
        return CLUB_RENDER_LAYOUT["5Wood"]
    if club == "4Hybrid" or "Hybrid" in str(club):
        return CLUB_RENDER_LAYOUT["4Hybrid"]
    if club in {"50Wedge", "52Wedge", "56Wedge", "SandWedge"} or "Wedge" in str(club):
        return CLUB_RENDER_LAYOUT["ZipCore"]
    return CLUB_RENDER_LAYOUT["P790"]


def get_club_asset(club: str, view: str = "face") -> Path | None:
    """선택한 클럽과 뷰에 해당하는 SVG 경로를 반환."""
    mapping = CLUB_ASSET_MAP.get(club)

    if mapping is None:
        if "Wood" in str(club):
            mapping = CLUB_ASSET_MAP["5Wood"]
        elif "Hybrid" in str(club):
            mapping = CLUB_ASSET_MAP["4Hybrid"]
        elif "Iron" in str(club):
            mapping = CLUB_ASSET_MAP["7Iron"]
        elif "Wedge" in str(club):
            mapping = CLUB_ASSET_MAP["50Wedge"]
        else:
            mapping = CLUB_ASSET_MAP["Driver"]

    path = CLUB_ASSET_DIR / mapping.get(view, mapping["face"])
    return path if path.exists() else None


@st.cache_data(show_spinner=False)
def load_club_image(image_path: str) -> np.ndarray:
    """PNG 또는 SVG 클럽 이미지를 RGBA 배열로 로드."""
    path = Path(image_path)

    if path.suffix.lower() == ".svg":
        import io
        import cairosvg
        from PIL import Image

        png_bytes = cairosvg.svg2png(url=str(path), output_width=1400)
        return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))

    return mpimg.imread(path)


def draw_club_asset(
    ax,
    club: str,
    view: str = "face",
    extent=(-24, 24, -17, 18),
) -> bool:
    """
    승인된 렌더 이미지를 원본 종횡비로 표시한다.

    face 뷰에서는 이미지 캔버스 중앙이 아니라 실제 클럽 페이스 중심이
    그래프의 (0, 0)에 오도록 클럽별 보정값을 적용한다.
    """
    path = get_club_asset(club, view)
    if path is None:
        return False

    image = load_club_image(str(path))
    img_h, img_w = image.shape[:2]
    img_ratio = img_w / img_h

    if view == "face":
        layout = get_render_layout(club)
        draw_w = float(layout["display_width"])
        draw_h = draw_w / img_ratio

        face_x = float(layout["face_center_x"])
        face_y = float(layout["face_center_y"])

        # 실제 페이스 중심이 좌표 (0, 0)에 오도록 이미지 전체를 이동
        actual_extent = (
            -face_x * draw_w,
            (1.0 - face_x) * draw_w,
            -face_y * draw_h,
            (1.0 - face_y) * draw_h,
        )
    else:
        x0, x1, y0, y1 = extent
        box_w = x1 - x0
        box_h = y1 - y0
        box_ratio = box_w / box_h

        if img_ratio >= box_ratio:
            draw_w = box_w
            draw_h = draw_w / img_ratio
        else:
            draw_h = box_h
            draw_w = draw_h * img_ratio

        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        actual_extent = (
            cx - draw_w / 2,
            cx + draw_w / 2,
            cy - draw_h / 2,
            cy + draw_h / 2,
        )

    ax.imshow(
        image,
        extent=actual_extent,
        interpolation="lanczos",
        origin="upper",
        zorder=0,
    )
    return True


def mean_existing_column(df: pd.DataFrame, candidates: list[str]) -> float:
    """후보 컬럼 중 존재하면서 유효한 첫 번째 컬럼의 평균을 반환."""
    for column in candidates:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if not values.empty:
                return float(values.mean())
    return float("nan")


def fmt(v: Any, nd: int = 1, suffix: str = "") -> str:
    try:
        if pd.isna(v):
            return "-"
        return f"{float(v):.{nd}f}{suffix}"
    except Exception:
        return "-"
    
def fmt_int(v: Any, comma: bool = False) -> str:
    """화면 표시용 정수 반올림."""
    try:
        if pd.isna(v):
            return "-"
        value = int(round(float(v)))
        return f"{value:,}" if comma else str(value)
    except Exception:
        return "-"


def side_text(v: Any) -> str:
    try:
        x = float(v)
        if abs(x) < 0.5:
            return "0"
        return f"{abs(int(round(x)))}{'R' if x > 0 else 'L'}"
    except Exception:
        return "-"

def render_top_metrics(items: list[tuple[str, str, str]]) -> None:
    """상단 KPI 카드를 Markdown 코드 블록으로 오인하지 않도록 한 줄 HTML로 렌더링."""
    cards: list[str] = []

    for label, value, unit in items:
        unit_html = f"<span class='tm-kpi-unit'>{unit}</span>" if unit else ""
        cards.append(
            "<div class='tm-kpi-card'>"
            f"<div class='tm-kpi-label'>{label}</div>"
            f"<div class='tm-kpi-value'>{value}{unit_html}</div>"
            "</div>"
        )

    cards_html = "".join(cards)
    st.markdown(
        f"<div class='tm-kpi-grid'>{cards_html}</div>",
        unsafe_allow_html=True,
    )


def classify_face_to_path(v: float | None) -> str:
    if v is None or pd.isna(v): return "-"
    if v > 0.8: return "오픈"
    if v < -0.8: return "클로즈"
    return "중립"


def classify_path(v: float | None) -> str:
    if v is None or pd.isna(v): return "-"
    if v > 1.0: return "인-아웃"
    if v < -1.0: return "아웃-인"
    return "중립"


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


def add_driver_face(ax, mode: str = "front"):
    # front: face view, side: side/top-like view, top: club path view
    if mode == "front":
        body = patches.Ellipse((0, 0), 40, 22, facecolor="#222c36", edgecolor="#8291a5", lw=1.3, alpha=.96)
        ax.add_patch(body)
        crown = patches.Ellipse((1, 1.5), 35, 17, facecolor="#2f3a46", edgecolor="none", alpha=.55)
        ax.add_patch(crown)
        for y in [-6, -3, 0, 3, 6]: ax.plot([-15, 15], [y, y], color="#7e8da0", lw=.65, alpha=.8)
        for x in [-15,-10,-5,0,5,10,15]: ax.plot([x,x],[-8,8], color="#334253", lw=.4, alpha=.55)
        # hosel
        ax.plot([16, 23], [7, 22], color="#758599", lw=4, alpha=.75)
        ax.plot([17, 24], [8, 23], color="#d4d9df", lw=1, alpha=.65)
    elif mode == "side":
        body = patches.Ellipse((0, 0), 42, 18, facecolor="#222c36", edgecolor="#8291a5", lw=1.3, alpha=.96)
        ax.add_patch(body)
        ax.add_patch(patches.Ellipse((-3,-1), 32, 12, facecolor="#303c49", edgecolor="none", alpha=.55))
        ax.plot([15, 20], [5, 22], color="#758599", lw=4, alpha=.75)
        ax.plot([-20, 18], [0, 0], color="#6f7e91", lw=.6, alpha=.8)
    else:
        body = patches.Ellipse((0,0), .9, .55, facecolor="#222c36", edgecolor="#8291a5", lw=1.3, alpha=.96)
        ax.add_patch(body)
        ax.plot([0.36,0.48],[0.18,0.75], color="#758599", lw=4, alpha=.75)
        ax.plot([0.37,0.49],[0.18,0.75], color="#d4d9df", lw=1, alpha=.7)


def impact_face_fig(offset, height, club, points=None, mode="face", figsize=(7.6, 4.3)):
    """승인된 고해상도 렌더를 사용하는 TrackMan 스타일 페이스 임팩트 카드."""
    fig, ax = plt.subplots(figsize=figsize, dpi=190)
    fig.patch.set_facecolor("#0b1016")
    ax.set_facecolor("#0b1016")
    ax.set_xlim(-30, 30)
    ax.set_ylim(-20, 22)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # 승인된 시안의 클럽 렌더를 크게 배치
    loaded = draw_club_asset(ax, club, "face")
    if not loaded:
        add_driver_face(ax, "front")

    cal = get_impact_calibration(club)
    center_x, center_y = cal["face_center"]
    plot_width, plot_height = cal["face_plot_size"]

    # 화면의 실제 페이스 영역에 맞춰 TrackMan mm 좌표를 재매핑
    layout = get_render_layout(club)

    # 렌더 이미지의 실제 페이스 폭/높이에 맞춘 표시 스케일.
    # 아이언은 샤프트가 포함된 캔버스 때문에 이미지 전체 폭이 넓으므로
    # 타점 이동 폭은 페이스 영역 기준으로 별도 계산한다.
    if club == "Driver":
        visual_face_width, visual_face_height = 34.0, 14.0
    elif club == "5Wood" or "Wood" in str(club):
        visual_face_width, visual_face_height = 32.0, 13.0
    elif club == "4Hybrid" or "Hybrid" in str(club):
        visual_face_width, visual_face_height = 29.0, 12.0
    elif club in {"50Wedge", "52Wedge", "56Wedge", "SandWedge"} or "Wedge" in str(club):
        visual_face_width, visual_face_height = 28.0, 14.0
    else:
        visual_face_width, visual_face_height = 28.5, 13.5

    scale_x = visual_face_width / plot_width
    scale_y = visual_face_height / plot_height

    def mapped(offset_value, height_value):
        x0, y0 = impact_mm_to_plot(offset_value, height_value, club, "face")
        return ((x0 - center_x) * scale_x, (y0 - center_y) * scale_y)

    x, y = mapped(offset, height)

    # 중심 십자선
    ax.plot([0, 0], [-13.0, 13.0], color="#e2e7eb", lw=1.0, ls=(0, (3, 3)), alpha=.82, zorder=5)
    ax.plot([-24.0, 24.0], [0, 0], color="#e2e7eb", lw=1.0, ls=(0, (3, 3)), alpha=.82, zorder=5)

    # 개별 타점
    if points is not None and not points.empty:
        pts = points.dropna(subset=["ImpactOffset_mm", "ImpactHeight_mm"])
        if not pts.empty:
            mapped_points = [mapped(r["ImpactOffset_mm"], r["ImpactHeight_mm"]) for _, r in pts.iterrows()]
            ax.scatter(
                [p[0] for p in mapped_points],
                [p[1] for p in mapped_points],
                s=10,
                color="#66a8ff",
                alpha=.12,
                edgecolors="none",
                zorder=6,
            )

    # 오렌지 글로우
    for size, alpha, color in [
        (900, .03, "#ff9a3c"),
        (580, .06, "#ff8125"),
        (320, .12, "#ff6c16"),
        (165, .22, "#ff8b32"),
    ]:
        ax.scatter([x], [y], s=size, color=color, alpha=alpha, edgecolors="none", zorder=7)

    ax.scatter([x], [y], s=88, color="#ff8a32", edgecolors="#ffd0a3", lw=1.2, zorder=8)
    ax.scatter([x], [y], s=18, color="#fff1df", edgecolors="none", zorder=9)

    try:
        off_val = float(offset)
        off_dir = "toe" if off_val > .5 else ("heel" if off_val < -.5 else "center")
        off_num = abs(off_val)
    except Exception:
        off_dir, off_num = "-", 0.0

    try:
        h_val = float(height)
        h_dir = "above" if h_val > .5 else ("below" if h_val < -.5 else "center")
        h_num = abs(h_val)
    except Exception:
        h_dir, h_num = "-", 0.0

    # 하단 정보 영역
    ax.text(-21.5, -12.3, "IMPACT OFFSET", color="#d6dde4", fontsize=8.5, fontweight="bold", ha="center")
    ax.text(-21.5, -16.1, f"{off_num:.0f}", color="#f5f8fb", fontsize=21, fontweight="bold", ha="center")
    ax.text(-17.4, -15.9, "mm", color="#d6dde4", fontsize=8.5, fontweight="bold", ha="left")
    ax.text(-21.5, -18.6, off_dir, color="#ff6b00", fontsize=9.5, fontweight="bold", ha="center")

    ax.text(0, -12.3, "IMPACT HEIGHT", color="#d6dde4", fontsize=8.5, fontweight="bold", ha="center")
    ax.text(0, -16.1, f"{h_num:.0f}", color="#f5f8fb", fontsize=21, fontweight="bold", ha="center")
    ax.text(4.1, -15.9, "mm", color="#d6dde4", fontsize=8.5, fontweight="bold", ha="left")
    ax.text(0, -18.6, h_dir, color="#ff6b00", fontsize=9.5, fontweight="bold", ha="center")

    ax.text(21.5, -12.3, "IMPACT POSITION", color="#d6dde4", fontsize=8.5, fontweight="bold", ha="center")
    ax.text(21.5, -16.1, f"{fmt(offset, 1)} / {fmt(height, 1)}", color="#f5f8fb", fontsize=15, fontweight="bold", ha="center")
    ax.text(21.5, -18.6, "offset / height", color="#9faab5", fontsize=8.3, fontweight="bold", ha="center")

    ax.text(-27.5, 20.0, club, color="#f4f7fa", fontsize=12, fontweight="bold", va="top")
    return fig



def _draw_top_view_club(ax, club: str) -> None:
    """클럽 패스용 깨끗한 탑뷰 벡터 렌더."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import PathPatch

    # 클럽군별 실루엣
    if club == "Driver":
        verts = [(-.26,.30),(-.06,.43),(.16,.37),(.29,.20),(.31,-.14),(.18,-.34),(-.05,-.40),(-.25,-.26),(-.31,0),(-.26,.30)]
    elif club == "5Wood" or "Wood" in str(club):
        verts = [(-.24,.28),(-.05,.39),(.14,.34),(.27,.18),(.29,-.12),(.17,-.31),(-.04,-.36),(-.23,-.24),(-.28,0),(-.24,.28)]
    elif club == "4Hybrid" or "Hybrid" in str(club):
        verts = [(-.22,.25),(-.05,.35),(.12,.31),(.24,.16),(.25,-.11),(.14,-.28),(-.03,-.32),(-.20,-.21),(-.25,0),(-.22,.25)]
    else:
        verts = [(-.10,.32),(.02,.38),(.10,.26),(.11,-.24),(.02,-.34),(-.10,-.28),(-.12,0),(-.10,.32)]

    codes = [MplPath.MOVETO] + [MplPath.CURVE3]*(len(verts)-2) + [MplPath.CLOSEPOLY]
    patch = PathPatch(
        MplPath(verts, codes),
        facecolor="#121820",
        edgecolor="#8d99a7",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(patch)

    # 내부 광택
    for radius, alpha in [(0.25,.13),(0.20,.10),(0.15,.07)]:
        ax.add_patch(
            patches.Ellipse(
                (-.02,.06), radius*1.6, radius,
                facecolor="#9ca7b2", edgecolor="none",
                alpha=alpha, zorder=3
            )
        )

    # 호젤/샤프트
    if "Iron" in str(club) or "Wedge" in str(club):
        ax.plot([.07,.10],[.26,.54], color="#737f8b", lw=5.2, solid_capstyle="round", zorder=3)
        ax.plot([.075,.105],[.27,.55], color="#d2d8dd", lw=1.1, alpha=.7, zorder=4)
    else:
        ax.plot([.19,.29],[.25,.56], color="#737f8b", lw=5.8, solid_capstyle="round", zorder=3)
        ax.plot([.195,.295],[.26,.57], color="#d2d8dd", lw=1.1, alpha=.7, zorder=4)

    # 모델명
    label = "Qi35" if club == "Driver" else "G430" if "Hybrid" in str(club) else "P790" if "Iron" in str(club) else "ZIPCORE" if "Wedge" in str(club) else "STEALTH 2"
    ax.text(0, -.31 if "Iron" not in str(club) and "Wedge" not in str(club) else -.27,
            label, color="#d7dde3", fontsize=6.5, ha="center", va="center", zorder=5)


def _draw_side_view_club(ax, club: str) -> None:
    """다이나믹 로프트용 깨끗한 측면 벡터 렌더."""
    if club == "Driver":
        body = patches.PathPatch(
            patches.Path(
                [(-.38,-.10),(-.24,.16),(.02,.20),(.20,.06),(.18,-.16),(-.10,-.26),(-.34,-.18),(-.38,-.10)],
                [1,3,3,3,3,3,3,79]
            ),
            facecolor="#171d24", edgecolor="#7d8792", lw=1.4, zorder=3
        )
    elif club == "5Wood" or "Wood" in str(club) or "Hybrid" in str(club):
        body = patches.PathPatch(
            patches.Path(
                [(-.34,-.08),(-.18,.13),(.02,.16),(.17,.05),(.15,-.14),(-.08,-.22),(-.30,-.16),(-.34,-.08)],
                [1,3,3,3,3,3,3,79]
            ),
            facecolor="#171d24", edgecolor="#7d8792", lw=1.4, zorder=3
        )
    else:
        body = patches.FancyBboxPatch(
            (-.12,-.24), .20, .50,
            boxstyle="round,pad=0.02,rounding_size=.03",
            facecolor="#aeb5bc", edgecolor="#e1e5e8", lw=1.2, zorder=3
        )
    ax.add_patch(body)

    # 샤프트
    shaft_x = .12 if "Iron" in str(club) or "Wedge" in str(club) else .13
    ax.plot([shaft_x,.18],[.22,.60], color="#697581", lw=5.0, solid_capstyle="round", zorder=3)
    ax.plot([shaft_x+.005,.185],[.23,.61], color="#d0d6dc", lw=1.0, alpha=.7, zorder=4)


def club_path_fig(row, figsize=(5.0, 3.9)):
    """
    고품질 TrackMan 스타일 클럽 패스 패널.

    오른손잡이 화면 규칙:
    + Club Path / + Face Angle -> 오른쪽 아래
    - Club Path / - Face Angle -> 오른쪽 위
    Face To Path = Face Angle - Club Path
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=190)
    fig.patch.set_facecolor("#0b141e")
    ax.set_facecolor("#0b141e")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-.72, .72)
    ax.axis("off")

    club = row.get("Club", "Driver")
    path = pd.to_numeric(pd.Series([row.get("ClubPath_deg")]), errors="coerce").iloc[0]
    face = pd.to_numeric(pd.Series([row.get("FaceAngle_deg")]), errors="coerce").iloc[0]
    ftp_source = pd.to_numeric(pd.Series([row.get("FaceToPath_deg")]), errors="coerce").iloc[0]

    ftp_calc = face - path if not pd.isna(face) and not pd.isna(path) else np.nan
    if pd.isna(ftp_source):
        ftp = ftp_calc
    elif pd.isna(ftp_calc):
        ftp = ftp_source
    elif abs(float(ftp_source) - float(ftp_calc)) <= 0.3:
        ftp = ftp_source
    else:
        ftp = ftp_calc

    # 기준선
    ax.plot([-1.0, .92], [0, 0], color="#d4dde5", lw=.9, ls=(0, (4, 4)), alpha=.48, zorder=1)

    _draw_top_view_club(ax, club)

    # 공
    ball = patches.Circle((.80, 0), .105, facecolor="#edf2f6", edgecolor="#c8d1da", lw=1.0, zorder=8)
    ax.add_patch(ball)
    for dx, dy in [(-.03,.035),(.025,.04),(.045,-.015),(-.035,-.03)]:
        ax.add_patch(patches.Circle((.80+dx,dy),.012,facecolor="#c8d0d8",edgecolor="none",alpha=.55,zorder=9))

    def endpoints(angle_deg, yoff):
        if pd.isna(angle_deg):
            return (-.82,yoff),(.68,yoff)
        visual = -float(np.clip(angle_deg*1.7,-16,16))
        rad = math.radians(visual)
        x0,x1 = -.82,.68
        return (x0, math.tan(rad)*x0*.48+yoff), (x1, math.tan(rad)*x1*.48+yoff)

    p0,p1 = endpoints(path,.028)
    f0,f1 = endpoints(face,-.035)

    # 선 그림자 + 본선
    for a,b,color in [(p0,p1,"#2f8cff"),(f0,f1,"#ff4238")]:
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=4.4,color="#02070c",alpha=.95),zorder=5)
        ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",lw=2.7,color=color,alpha=1.0),zorder=7)

    def signed_value(v):
        if pd.isna(v): return "-"
        suffix = "R" if v > .05 else "L" if v < -.05 else ""
        return f"{abs(float(v)):.1f}°{suffix}"

    def dir_text(v, pos, neg):
        if pd.isna(v): return "-"
        return pos if v > .05 else neg if v < -.05 else "neutral"

    ax.text(-1.00,.60,"CLUB PATH",color="#dce4ec",fontsize=8.8,fontweight="bold")
    ax.text(-1.00,.40,signed_value(path),color="#f5f8fb",fontsize=17.5,fontweight="bold")
    ax.text(-1.00,.27,dir_text(path,"in to out","out to in"),color="#aebbc8",fontsize=8.2)

    ax.text(.44,.60,"FACE TO PATH",color="#dce4ec",fontsize=8.8,fontweight="bold")
    ax.text(.44,.40,signed_value(ftp),color="#f5f8fb",fontsize=17.5,fontweight="bold")
    ax.text(.44,.27,dir_text(ftp,"open","closed"),color="#aebbc8",fontsize=8.2)

    ax.text(-1.00,-.58,f"Face Angle  {signed_value(face)}",color="#ff5a50",fontsize=9.0)
    return fig


def loft_spin_fig(row, figsize=(5.0, 3.9)):
    """고품질 TrackMan 스타일 런치 앵글 / 스핀 로프트 패널."""
    fig, ax = plt.subplots(figsize=figsize, dpi=190)
    fig.patch.set_facecolor("#0b141e")
    ax.set_facecolor("#0b141e")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-.72, .72)
    ax.axis("off")

    club = row.get("Club", "Driver")
    attack = pd.to_numeric(pd.Series([row.get("AttackAngle_deg")]), errors="coerce").iloc[0]
    launch_angle = pd.to_numeric(pd.Series([row.get("LaunchAngle_deg")]), errors="coerce").iloc[0]
    dynamic_loft = pd.to_numeric(pd.Series([row.get("DynamicLoft_deg")]), errors="coerce").iloc[0]
    spin_loft_source = pd.to_numeric(pd.Series([row.get("SpinLoft_deg")]), errors="coerce").iloc[0]
    spin_rate = pd.to_numeric(pd.Series([row.get("SpinRate_rpm")]), errors="coerce").iloc[0]

    spin_loft = spin_loft_source
    if pd.isna(spin_loft) and not pd.isna(dynamic_loft) and not pd.isna(attack):
        spin_loft = float(dynamic_loft)-float(attack)

    ax.plot([-1.0,.92],[0,0],color="#d4dde5",lw=.9,alpha=.45,zorder=1)
    _draw_side_view_club(ax,club)

    ball = patches.Circle((.22,0),.11,facecolor="#edf2f6",edgecolor="#c8d1da",lw=1.0,zorder=8)
    ax.add_patch(ball)

    # Attack line
    if not pd.isna(attack):
        a_vis = float(np.clip(attack*1.8,-18,18))
        ar = math.radians(a_vis)
        x0,x1 = -.72,.66
        y0,y1 = math.tan(ar)*x0*.40, math.tan(ar)*x1*.40
        ax.plot([x0,x1],[y0,y1],color="#2f8cff",lw=2.5,zorder=6)

    # Launch angle line and spin loft fan
    if not pd.isna(launch_angle):
        l_vis = float(np.clip(launch_angle*.85,-8,32))
        lr = math.radians(l_vis)
        x0,x1 = .22,.78
        ly0,ly1 = 0, math.tan(lr)*(x1-x0)*.85
        ax.plot([x0,x1],[ly0,ly1],color="#ff4238",lw=2.6,zorder=7)

        if not pd.isna(attack):
            attack_y = math.tan(math.radians(float(np.clip(attack*1.8,-18,18))))*(x1-x0)*.40
            ax.fill([x0,x1,x1],[0,ly1,attack_y],color="#b7bdc3",alpha=.30,zorder=2)

    def val(v, unit="°", nd=1):
        return "-" if pd.isna(v) else f"{float(v):.{nd}f}{unit}"

    ax.text(-1.00,.60,"LAUNCH ANGLE",color="#dce4ec",fontsize=8.8,fontweight="bold")
    ax.text(-1.00,.40,val(launch_angle),color="#f5f8fb",fontsize=17.5,fontweight="bold")

    ax.text(.52,.60,"SPIN RATE",color="#dce4ec",fontsize=8.8,fontweight="bold")
    spin_text = "-" if pd.isna(spin_rate) else f"{int(round(float(spin_rate))):,} rpm"
    ax.text(.52,.40,spin_text,color="#f5f8fb",fontsize=15.5,fontweight="bold")

    ax.text(-1.00,-.33,"ATTACK ANGLE",color="#dce4ec",fontsize=8.8,fontweight="bold")
    ax.text(-1.00,-.54,val(attack),color="#f5f8fb",fontsize=17.5,fontweight="bold")

    ax.text(.52,-.33,"SPIN LOFT",color="#dce4ec",fontsize=8.8,fontweight="bold")
    ax.text(.52,-.54,val(spin_loft),color="#f5f8fb",fontsize=17.5,fontweight="bold")

    return fig


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


def _shot_table(df_shots: pd.DataFrame, selected_index: int, key: str) -> int:
    """샷 목록을 앱의 다크 테마로 표시합니다. 샷 선택은 위 슬라이더/버튼과 연동됩니다."""
    table_columns = [
        "StrokeNo", "ShotTimeLocal", "Carry_m", "Total_m", "BallSpeed_mps",
        "ClubSpeed_mps", "SmashFactor", "SpinRate_rpm", "LaunchAngle_deg",
        "AttackAngle_deg", "ClubPath_deg", "FaceAngle_deg", "FaceToPath_deg",
        "TotalSide_m", "ImpactOffset_mm", "ImpactHeight_mm",
    ]
    available = [column for column in table_columns if column in df_shots.columns]
    table_df = df_shots.loc[:, available].copy().reset_index(drop=True)
    table_df.insert(0, "선택", ["▶" if idx == selected_index else "" for idx in range(len(table_df))])
    render_dark_dataframe(table_df, selected_row=selected_index)
    st.caption("샷 변경은 위의 이전/다음 버튼 또는 샷 선택 슬라이더를 사용하세요.")
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

def render_single_shot_analysis(
    day_df: pd.DataFrame,
    month_df: pd.DataFrame,
    year_df: pd.DataFrame,
    club: str,
    selected_date: str,
    day_summary: pd.Series,
    month_summary: pd.Series,
    year_summary: pd.Series,
) -> None:
    """Step 1~3: 개별 샷 탐색, 목록 선택, 기간 평균 비교를 한 화면에 렌더링합니다."""
    shots = day_df[day_df["Club"] == club].copy()
    sort_columns = _shot_sort_columns(shots)
    if sort_columns:
        shots = shots.sort_values(sort_columns, kind="stable")
    shots = shots.reset_index(drop=True)

    if shots.empty:
        st.info("선택한 날짜에 해당 클럽의 샷 데이터가 없습니다.")
        return

    state_key = f"shot_index::{club}::{selected_date}"
    if state_key not in st.session_state:
        st.session_state[state_key] = 0
    st.session_state[state_key] = min(max(int(st.session_state[state_key]), 0), len(shots) - 1)

    st.markdown(
        f"<div class='tm-shot-heading'><div class='tm-shot-heading-title'>{club} 개별 샷 분석</div>"
        f"<div class='tm-shot-heading-sub'>{selected_date} · 총 {len(shots)}샷</div></div>",
        unsafe_allow_html=True,
    )

    previous_col, slider_col, next_col = st.columns([0.8, 3.4, 0.8])
    with previous_col:
        if st.button("◀ 이전 샷", width="stretch", disabled=st.session_state[state_key] <= 0, key=f"prev::{state_key}"):
            st.session_state[state_key] -= 1
            st.rerun()
    with slider_col:
        selected_number = st.slider(
            "샷 선택",
            min_value=1,
            max_value=len(shots),
            value=st.session_state[state_key] + 1,
            step=1,
            key=f"slider::{state_key}",
        )
        slider_index = selected_number - 1
        if slider_index != st.session_state[state_key]:
            st.session_state[state_key] = slider_index
            st.rerun()
    with next_col:
        if st.button("다음 샷 ▶", width="stretch", disabled=st.session_state[state_key] >= len(shots) - 1, key=f"next::{state_key}"):
            st.session_state[state_key] += 1
            st.rerun()

    selected_index = st.session_state[state_key]
    row = shots.iloc[selected_index]
    shot_no = _shot_display_number(row, selected_index)
    shot_time = str(row.get("ShotTimeLocal", "") or "")
    st.caption(f"현재 선택: Shot {shot_no} · {shot_time} · {selected_index + 1}/{len(shots)}")

    # Step 1: 선택 샷의 모든 핵심 수치 및 3개 시각화
    render_top_metrics(_shot_metric_items(row))
    render_shot_detail_panel(row, state_suffix=f"{club}::{selected_date}")

    # 탄착군과 선택 샷의 임팩트/패스/로프트를 한 행에 배치해 화면 균형을 맞춥니다.
    club_title = _club_korean_name(club)
    visual_cols = st.columns([1.18, 0.92, 0.92, 0.92])
    with visual_cols[0]:
        st.markdown(f"<div class='tm-panel-title'>{club_title} 탄착군</div>", unsafe_allow_html=True)
        _render_blinking_direction_distribution(
            shots,
            selected_index,
            distance_metric="Carry_m",
        )
    with visual_cols[1]:
        st.markdown("<div class='tm-panel-title'>선택 샷 임팩트 위치</div>", unsafe_allow_html=True)
        st.pyplot(
            impact_face_fig(
                row.get("ImpactOffset_mm"),
                row.get("ImpactHeight_mm"),
                club,
                points=None,
                figsize=(5.0, 3.45),
            ),
            clear_figure=True,
        )
    with visual_cols[2]:
        st.markdown("<div class='tm-panel-title'>선택 샷 클럽 패스</div>", unsafe_allow_html=True)
        st.pyplot(club_path_fig(row, figsize=(4.8, 3.45)), clear_figure=True)
    with visual_cols[3]:
        st.markdown("<div class='tm-panel-title'>선택 샷 로프트 / 스핀 로프트</div>", unsafe_allow_html=True)
        st.pyplot(loft_spin_fig(row, figsize=(4.8, 3.45)), clear_figure=True)

    # Step 2: 행을 클릭하면 선택 샷 이동
    st.markdown("### 샷 목록")
    st.caption("표에서 한 행을 선택하면 위의 샷 상세 화면이 해당 샷으로 이동합니다.")
    clicked_index = _shot_table(shots, selected_index, key=f"shot_table::{club}::{selected_date}")
    if clicked_index != selected_index:
        st.session_state[state_key] = clicked_index
        st.rerun()

    # Step 3: 선택 샷과 평균 비교
    st.markdown("### 선택 샷 vs 당일·월간·연간 평균")
    _render_shot_compare_cards(row, day_summary, month_summary, year_summary)

# Header
st.markdown(
    "<div class='tm-logo' style='padding-top:8px'><span class='tm-orange'>▰</span> TRACKMAN DASHBOARD</div>",
    unsafe_allow_html=True,
)

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
        st.sidebar.success(f"최신 데이터 반영 완료 · 신규 {pull_result.downloaded}회")
        st.rerun()
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

analysis_tabs = st.tabs(['📊 평균 분석', '🎯 샷별 분석'])

with analysis_tabs[0]:
    st.markdown(f"<div class='tm-title'>기간 비교 ({club})</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='tm-legend'><span><i class='tm-dot tm-dot-day'></i>선택일 ({selected_date})</span><span><i class='tm-dot tm-dot-month'></i>월간 평균 ({mlabel})</span><span><i class='tm-dot tm-dot-year'></i>연간 평균 ({ylabel})</span></div>",unsafe_allow_html=True)
    _render_compare_cards(ds,ms,ys)
    # 세 패널의 제목/컨트롤 행을 먼저 만들고 그래프 행을 분리해 상단 정렬을 맞춥니다.
    compare_headers = st.columns(3)
    with compare_headers[0]:
        title_col, control_col = st.columns([1.0, 0.72], vertical_alignment="center")
        with title_col:
            st.markdown("<div class='tm-panel-title'>거리 분포 비교</div>", unsafe_allow_html=True)
        with control_col:
            avg_distance_choice = st.radio(
                '거리 기준', ['캐리', '토탈'], horizontal=True,
                key=f'avg_distance_metric::{club}::{selected_date}',
                label_visibility='collapsed',
            )
    with compare_headers[1]:
        st.markdown(f"<div class='tm-panel-title'>{_club_korean_name(club)} 탄착군 비교</div>",unsafe_allow_html=True)
    with compare_headers[2]:
        st.markdown(f"<div class='tm-panel-title'>월별 {'캐리' if avg_distance_choice == '캐리' else '토탈'} 추세</div>",unsafe_allow_html=True)

    avg_distance_metric = 'Carry_m' if avg_distance_choice == '캐리' else 'Total_m'
    cc=st.columns(3)
    with cc[0]:
        _distance_chart(day,month,year,club,avg_distance_metric)
    with cc[1]:
        _side_chart(day,month,year,club,avg_distance_metric)
    with cc[2]:
        _trend(filtered_df,club,pd.to_datetime(selected_date).year,mode,avg_distance_metric)
    periods=[('선택일',selected_date,day,ds,'전체 샷 평균'),('월간 평균',mlabel,month,ms,mode),('연간 평균',ylabel,year,ys,mode)]
    st.markdown('### 임팩트 위치 비교')
    cols=st.columns(3)
    for c,(title,label,raw,s,m) in zip(cols,periods):
        with c:
            st.markdown(f'**{title} ({label})**')
            if s.empty:
                st.info('비교 데이터 없음')
            else:
                st.pyplot(impact_face_fig(s.get('Avg_ImpactOffset_mm'),s.get('Avg_ImpactHeight_mm'),club,raw[raw['Club']==club],figsize=(5,3.2)),clear_figure=True)
    st.markdown('### 클럽 패스 비교')
    cols=st.columns(3)
    for c,(title,label,raw,s,m) in zip(cols,periods):
        with c:
            st.markdown(f'**{title} ({label})**')
            if s.empty:
                st.info('비교 데이터 없음')
            else:
                st.pyplot(club_path_fig(_period_row(raw,s,club,m),figsize=(4.8,3.4)),clear_figure=True)
    st.markdown('### 런치 앵글 / 스핀 로프트 비교')
    cols=st.columns(3)
    for c,(title,label,raw,s,m) in zip(cols,periods):
        with c:
            st.markdown(f'**{title} ({label})**')
            if s.empty:
                st.info('비교 데이터 없음')
            else:
                st.pyplot(loft_spin_fig(_period_row(raw,s,club,m),figsize=(4.8,3.4)),clear_figure=True)
    st.markdown('### 자동 분석 요약')
    st.markdown(f"<div class='tm-auto-summary'>{_auto_text(ds,ms,ys,club)}</div>",unsafe_allow_html=True)

    # 상세 분석 화면에서 유용했던 기능은 평균 분석 하단의 접이식 메뉴로 통합합니다.
    with st.expander('전체 클럽 요약', expanded=False):
        selected_day_all_clubs = filtered_df[filtered_df['Date'] == selected_date].copy()
        all_club_summary = pd.DataFrame(
            make_summary(selected_day_all_clubs.to_dict('records'))
        )
        if all_club_summary.empty:
            st.info('선택한 날짜의 클럽 요약 데이터가 없습니다.')
        else:
            all_club_summary = all_club_summary[
                [c for c in SUMMARY_COLUMNS if c in all_club_summary.columns]
            ]
            if 'Club' in all_club_summary.columns:
                all_club_summary = all_club_summary.assign(
                    _club_order=all_club_summary['Club'].map(club_sort_key)
                ).sort_values('_club_order').drop(columns='_club_order')
            render_club_cards(all_club_summary, max_cards=6)
            st.dataframe(
                safe_dataframe_for_streamlit(all_club_summary),
                width='stretch',
                hide_index=True,
            )

    with st.expander('원본 샷 데이터', expanded=False):
        raw_filter_col1, raw_filter_col2 = st.columns(2)
        with raw_filter_col1:
            raw_clubs = st.multiselect(
                '클럽',
                clubs,
                default=[club],
                key=f'raw_shot_clubs::{club}::{selected_date}',
            )
        with raw_filter_col2:
            raw_dates = st.multiselect(
                '날짜',
                sorted(dates, reverse=True),
                default=[selected_date],
                key=f'raw_shot_dates::{club}::{selected_date}',
            )

        raw_df = filtered_df.copy()
        if raw_clubs:
            raw_df = raw_df[raw_df['Club'].isin(raw_clubs)]
        if raw_dates:
            raw_df = raw_df[raw_df['Date'].isin(raw_dates)]

        raw_columns = [
            c for c in [
                'Club', 'StrokeNo', 'Date', 'ShotTimeLocal',
                'Carry_m', 'Total_m', 'Run_m',
                'BallSpeed_mps', 'ClubSpeed_mps', 'SmashFactor',
                'SpinRate_rpm', 'LaunchAngle_deg', 'AttackAngle_deg',
                'ClubPath_deg', 'FaceAngle_deg', 'FaceToPath_deg',
                'TotalSide_m', 'ImpactOffset_mm', 'ImpactHeight_mm',
            ] if c in raw_df.columns
        ]
        raw_df = raw_df.sort_values(
            [c for c in ['Date', 'Club', 'StrokeNo'] if c in raw_df.columns],
            ascending=[False, True, True][:len([c for c in ['Date', 'Club', 'StrokeNo'] if c in raw_df.columns])],
        ) if not raw_df.empty else raw_df
        st.caption(f'표시 샷 수: {len(raw_df):,}개')
        raw_display_df = raw_df.loc[:, raw_columns].reset_index(drop=True)
        csv_bytes = raw_display_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            'CSV 다운로드',
            data=csv_bytes,
            file_name=f'trackman_raw_{selected_date}.csv',
            mime='text/csv',
            key=f'raw_csv::{club}::{selected_date}',
        )
        render_dark_dataframe(raw_display_df)

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
    )
