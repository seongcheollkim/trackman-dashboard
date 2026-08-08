from __future__ import annotations

import html

from .ai_engine import PracticeDiagnosis
from .ai_rules import korean_club


def _escape(value: object) -> str:
    return html.escape(str(value))


def _stars_html(value: float) -> str:
    full = int(value)
    half = 1 if value - full >= 0.5 else 0
    empty = max(0, 5 - full - half)
    return (
        "<span class='dodos-star-on'>" + ("★" * full) + "</span>"
        + ("<span class='dodos-star-half'>★</span>" if half else "")
        + "<span class='dodos-star-off'>" + ("★" * empty) + "</span>"
    )


def diagnosis_html(report: PracticeDiagnosis) -> str:
    score = f"{report.score:.0f}" if report.grade != "-" else "-"
    confidence = f"{report.confidence}%"

    def items(values: tuple[str, ...], empty: str) -> str:
        if not values:
            return f"<div class='dodos-ai-empty'>{_escape(empty)}</div>"
        return "".join(
            f"<div class='dodos-ai-list-item'>• {_escape(value)}</div>"
            for value in values
        )

    category_cards = ""
    preferred_order = ["드라이버", "우드", "유틸리티", "아이언", "웨지", "기타"]
    for label in preferred_order:
        if label not in report.category_scores:
            continue
        category_cards += (
            "<div class='dodos-category-card'>"
            f"<div class='dodos-category-name'>{_escape(label)}</div>"
            f"<div class='dodos-category-stars'>{_stars_html(report.category_stars.get(label, 0))}</div>"
            f"<div class='dodos-category-score'>{report.category_scores[label]:.0f}점</div>"
            "</div>"
        )

    club_cards = ""
    for club in report.clubs:
        club_cards += (
            "<div class='dodos-ai-club-card'>"
            f"<div class='dodos-ai-club-name'>{_escape(korean_club(club.club))}</div>"
            f"<div class='dodos-ai-club-score'>{club.score:.0f}<span>/100</span></div>"
            f"<div class='dodos-ai-club-meta'>{_escape(club.grade)} · 신뢰도 {club.confidence}% · {club.shots}샷</div>"
            f"<div class='dodos-ai-club-meta'>P {club.performance_score:.0f} · C {club.consistency_score:.0f} · T {club.trend_score:.0f}</div>"
            "</div>"
        )

    best_name = korean_club(report.best_club) if report.best_club else "-"
    focus_name = korean_club(report.focus_club) if report.focus_club else "-"

    return f"""
    <div class="dodos-report-title-row">
      <div>
        <div class="dodos-ai-eyebrow">DODOS DAILY PRACTICE REPORT · {_escape(report.date)}</div>
        <div class="dodos-report-title">오늘의 연습 리포트</div>
        <div class="dodos-ai-meta">{_escape(report.goal_label)} 목표 · 총 {report.total_shots}샷 · 분석 클럽 {len(report.clubs)}개</div>
      </div>
      <div class="dodos-ai-score-wrap dodos-report-score">
        <div class="dodos-ai-score">{score}</div>
        <div class="dodos-ai-grade">{_escape(report.grade)}</div>
        <div class="dodos-ai-confidence">Confidence {confidence}</div>
      </div>
    </div>

    <div class="dodos-category-grid">{category_cards}</div>

    <div class="dodos-highlight-grid">
      <div class="dodos-highlight-card best">
        <div class="dodos-highlight-label">🏆 오늘 가장 좋았던 클럽</div>
        <div class="dodos-highlight-name">{_escape(best_name)}</div>
        <div class="dodos-highlight-score">{report.best_club_score:.0f}점</div>
      </div>
      <div class="dodos-highlight-card focus">
        <div class="dodos-highlight-label">🎯 오늘 우선 개선할 클럽</div>
        <div class="dodos-highlight-name">{_escape(focus_name)}</div>
        <div class="dodos-highlight-score">{report.focus_club_score:.0f}점</div>
      </div>
      <div class="dodos-coaching-card">
        <div class="dodos-highlight-label">💬 오늘의 한 줄 코칭</div>
        <div class="dodos-coaching-text">{_escape(report.coaching_summary)}</div>
      </div>
    </div>

    <details class="dodos-score-details">
      <summary>AI 점수 구성 보기</summary>
      <div class="dodos-ai-score-breakdown">
        <div><b>{report.performance_score:.0f}</b><span>Performance · 50%</span></div>
        <div><b>{report.consistency_score:.0f}</b><span>Consistency · 30%</span></div>
        <div><b>{report.trend_score:.0f}</b><span>Trend · 20%</span></div>
      </div>
    </details>

    <div class="dodos-ai-grid">
      <div class="dodos-ai-panel">
        <div class="dodos-ai-panel-title">✅ 오늘 잘된 점</div>
        {items(report.strengths, "오늘 유지할 만한 뚜렷한 강점이 아직 충분하지 않습니다.")}
      </div>
      <div class="dodos-ai-panel">
        <div class="dodos-ai-panel-title">🎯 오늘 아쉬웠던 점</div>
        {items(report.improvements, "오늘 크게 수정할 필요가 있는 클럽은 없습니다.")}
      </div>
      <div class="dodos-ai-panel">
        <div class="dodos-ai-panel-title">🏌️ 다음 연습 우선순위</div>
        {items(report.tasks, "현재 좋은 리듬을 유지하며 품질 샷을 이어가세요.")}
      </div>
    </div>

    <div class="dodos-ai-club-grid">{club_cards}</div>
    """

AI_CSS = r"""
<style>
.dodos-ai-hero{
  display:grid;
  grid-template-columns:minmax(0,1fr) 220px;
  gap:20px;
  align-items:center;
  border:1px solid #2b3b4e;
  border-radius:16px;
  padding:22px 24px;
  margin:8px 0 16px;
  background:linear-gradient(135deg,#111f2d,#0a131e);
}
.dodos-ai-eyebrow{font-size:.76rem;font-weight:850;letter-spacing:.08em;color:#ff8a32;margin-bottom:10px}
.dodos-ai-headline{font-size:1.28rem;font-weight:850;color:#f6f9fc;line-height:1.55}
.dodos-ai-meta{color:#96a8bb;font-size:.82rem;margin-top:9px}
.dodos-ai-score-wrap{text-align:center;border-left:1px solid #2b3b4e}
.dodos-ai-score{font-size:3.35rem;font-weight:950;color:#ffffff;line-height:1}
.dodos-ai-grade{font-size:1.15rem;font-weight:900;color:#ff8a32;margin-top:5px}
.dodos-ai-confidence{font-size:.76rem;color:#9db0c4;margin-top:7px}
.dodos-ai-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:12px 0 18px}
.dodos-ai-panel{border:1px solid #263548;border-radius:12px;padding:15px 16px;background:linear-gradient(180deg,#111e2b,#0d1721);min-height:165px}
.dodos-ai-panel-title{font-size:.98rem;font-weight:900;color:#f5f8fc;margin-bottom:11px}
.dodos-ai-list-item{color:#d7e2ed;font-size:.86rem;line-height:1.62;margin:5px 0}
.dodos-ai-empty{color:#8fa3b7;font-size:.84rem;line-height:1.55}
.dodos-ai-club-grid{display:grid;grid-template-columns:repeat(4,minmax(145px,1fr));gap:10px;margin:8px 0 18px}
.dodos-ai-club-card{border:1px solid #263548;border-radius:11px;padding:13px 14px;background:#0e1924}
.dodos-ai-club-name{font-size:.85rem;color:#b9c8d8;font-weight:750}
.dodos-ai-club-score{font-size:1.55rem;color:#fff;font-weight:900;margin-top:7px}
.dodos-ai-club-score span{font-size:.72rem;color:#879aaf;font-weight:650;margin-left:3px}
.dodos-ai-club-meta{font-size:.72rem;color:#8fa3b7;margin-top:5px}
@media(max-width:900px){
  .dodos-ai-hero{grid-template-columns:1fr}
  .dodos-ai-score-wrap{border-left:0;border-top:1px solid #2b3b4e;padding-top:16px}
  .dodos-ai-grid{grid-template-columns:1fr}
  .dodos-ai-club-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}

.dodos-ai-score-breakdown{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:0 0 16px}
.dodos-ai-score-breakdown>div{border:1px solid #263548;border-radius:11px;background:#0e1924;padding:12px 14px;text-align:center}
.dodos-ai-score-breakdown b{display:block;color:#fff;font-size:1.42rem}
.dodos-ai-score-breakdown span{display:block;color:#8fa3b7;font-size:.75rem;margin-top:3px}
@media(max-width:900px){.dodos-ai-score-breakdown{grid-template-columns:1fr}}

.dodos-report-title-row{
  display:grid;grid-template-columns:minmax(0,1fr) 210px;gap:20px;align-items:center;
  border:1px solid #2b3b4e;border-radius:16px;padding:22px 24px;margin:8px 0 14px;
  background:linear-gradient(135deg,#111f2d,#0a131e)
}
.dodos-report-title{font-size:1.48rem;font-weight:920;color:#f7f9fc;line-height:1.35}
.dodos-report-score{border-left:1px solid #2b3b4e}
.dodos-category-grid{display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:10px;margin:0 0 14px}
.dodos-category-card{border:1px solid #263548;border-radius:12px;background:#0e1924;padding:13px 14px}
.dodos-category-name{color:#dce6ef;font-weight:850;font-size:.88rem}
.dodos-category-stars{font-size:1.0rem;letter-spacing:.04em;margin:8px 0 4px}
.dodos-star-on{color:#ff8a32}.dodos-star-half{color:#ffb16f}.dodos-star-off{color:#35485d}
.dodos-category-score{font-size:.76rem;color:#8fa3b7}
.dodos-highlight-grid{display:grid;grid-template-columns:1fr 1fr 2fr;gap:10px;margin:0 0 14px}
.dodos-highlight-card,.dodos-coaching-card{border:1px solid #263548;border-radius:12px;background:#0e1924;padding:15px 16px}
.dodos-highlight-card.best{border-color:#2f5c4a}.dodos-highlight-card.focus{border-color:#6a4933}
.dodos-highlight-label{font-size:.78rem;font-weight:800;color:#9fb1c4;margin-bottom:8px}
.dodos-highlight-name{font-size:1.08rem;font-weight:900;color:#fff}
.dodos-highlight-score{font-size:.8rem;color:#ff8a32;margin-top:4px}
.dodos-coaching-text{font-size:.94rem;color:#e1e9f1;line-height:1.65;font-weight:650}
.dodos-score-details{border:1px solid #263548;border-radius:11px;background:#0c1620;padding:0 13px;margin:0 0 14px}
.dodos-score-details summary{cursor:pointer;color:#a9b9c9;font-size:.82rem;font-weight:760;padding:11px 0}
.dodos-score-details .dodos-ai-score-breakdown{margin:0 0 12px}
@media(max-width:1050px){
  .dodos-category-grid{grid-template-columns:repeat(3,minmax(0,1fr))}
  .dodos-highlight-grid{grid-template-columns:1fr 1fr}
  .dodos-coaching-card{grid-column:1/-1}
}
@media(max-width:700px){
  .dodos-report-title-row{grid-template-columns:1fr}
  .dodos-report-score{border-left:0;border-top:1px solid #2b3b4e;padding-top:14px}
  .dodos-category-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
  .dodos-highlight-grid{grid-template-columns:1fr}
  .dodos-coaching-card{grid-column:auto}
}

</style>
"""


# v2 score breakdown CSS is injected as part of the same style block by Streamlit.
