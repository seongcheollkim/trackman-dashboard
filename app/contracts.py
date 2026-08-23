from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DashboardContext:
    """화면 렌더링 계층에 전달할 공통 실행 컨텍스트."""

    all_df: pd.DataFrame
    filtered_df: pd.DataFrame
    selected_date: Any | None = None
    selected_clubs: tuple[str, ...] = ()
    session_label: str = ""


@dataclass(slots=True)
class DashboardPeriods:
    """선택일/월간/연간 비교 데이터 묶음."""

    day: pd.DataFrame
    month: pd.DataFrame
    year: pd.DataFrame
    month_label: str = ""
    year_label: str = ""
