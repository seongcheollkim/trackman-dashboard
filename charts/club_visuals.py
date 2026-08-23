
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ui.common import fmt


PROJECT_DIR = Path(__file__).resolve().parents[1]
CLUB_ASSET_DIR = PROJECT_DIR / "assets" / "clubs"

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
