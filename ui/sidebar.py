from __future__ import annotations

from typing import Any, Callable

import streamlit as st


def render_sidebar_header(
    *,
    auth_name: str,
    auth_email: str,
    on_logout: Callable[[], Any] | None = None,
) -> None:
    st.sidebar.markdown("## Trackman 분석")
    st.sidebar.caption("by seongcheoll.kim")
    st.sidebar.divider()

    st.sidebar.markdown("### 👤 로그인 사용자")
    st.sidebar.caption(auth_name)
    st.sidebar.caption(auth_email)
    if st.sidebar.button("로그아웃", width="stretch", type="primary"):
        if on_logout is not None:
            on_logout()
        else:
            st.logout()


def render_storage_status(storage_status: Any) -> None:
    """현재 저장/클라우드 상태 표시만 담당합니다."""
    st.sidebar.divider()
    st.sidebar.metric("저장된 연습", f"{storage_status.report_count}회")

    if storage_status.last_sync is not None:
        st.sidebar.caption(
            "마지막 동기화: "
            + storage_status.last_sync.astimezone().strftime("%Y-%m-%d %H:%M")
        )
    else:
        st.sidebar.caption("마지막 동기화: 없음")

    if not storage_status.cloud_configured:
        st.sidebar.warning("☁️ Supabase 설정 필요")
        return

    if storage_status.cloud_connected:
        cloud_sessions = storage_status.cloud_report_count or 0
        cloud_shots = storage_status.cloud_shot_count
        label = f"☁️ Supabase 연결됨 · {cloud_sessions}회"
        if cloud_shots is not None:
            label += f" · {cloud_shots:,}샷"
        st.sidebar.success(label)
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


def render_data_actions(
    *,
    storage: Any,
    storage_status: Any,
    sync_journal_callback: Callable[[], Any] | None = None,
    sync_trackman_callback: Callable[[], Any] | None = None,
    app_dir: Any | None = None,
    numeric_range_reset_callback: Callable[[], Any] | None = None,
) -> Any:
    """데이터 동기화/백업/업로드 UI를 렌더링합니다.

    실제 Storage/TrackMan 작업은 callback으로 주입하여 sidebar가
    비즈니스 로직을 직접 소유하지 않도록 합니다.
    """
    st.sidebar.divider()

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

            if sync_journal_callback is not None:
                with st.spinner("연습일지 DB와 AI 분석을 동기화하는 중입니다..."):
                    db_sync_result = sync_journal_callback()
            else:
                db_sync_result = None

            if db_sync_result is None or db_sync_result.ok:
                st.sidebar.success(
                    f"최신 데이터 반영 완료 · 신규 {pull_result.downloaded}회"
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

    st.sidebar.markdown("### 데이터 관리")

    direct_sync_required = []
    if app_dir is not None:
        direct_sync_required = [
            app_dir / "activity_list.curl",
            app_dir / "activity_report.curl",
            app_dir / "download_all_trackman_reports.py",
            app_dir / "trackman_auth_refresh.py",
        ]
    direct_sync_available = bool(direct_sync_required) and all(
        path.exists() for path in direct_sync_required
    )

    if direct_sync_available and sync_trackman_callback is not None:
        if st.button("🖥️ 이 Mac에서 TrackMan 직접 동기화", width="stretch"):
            with st.spinner("TrackMan 데이터를 수집하고 Supabase에 백업하는 중입니다..."):
                sync_result = sync_trackman_callback()
            if sync_result.ok:
                cloud_uploaded = sync_result.cloud.uploaded if sync_result.cloud else 0
                storage.invalidate_cache()
                st.cache_data.clear()
                st.success(
                    f"직접 동기화 완료 · 신규 {sync_result.downloaded_count}회 · 클라우드 {cloud_uploaded}회"
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

    if st.button(
        "⬆️ 로컬 데이터 백업",
        width="stretch",
        disabled=not storage.cloud_configured,
    ):
        with st.spinner("로컬 보고서를 Supabase에 백업하는 중입니다..."):
            upload_result = storage.upload_local_reports()
        if upload_result.ok:
            storage.write_last_sync(
                source="supabase_backup",
                details={"uploaded": upload_result.uploaded},
            )
            st.success(
                f"백업 완료 · 신규 {upload_result.uploaded}회 · 기존 {upload_result.skipped}회"
            )
            st.rerun()
        else:
            st.error("Supabase 백업에 실패했습니다.")
            with st.expander("오류 내용"):
                st.code("\n".join(upload_result.errors)[-4000:])

    if st.button("↻ 로컬 캐시 새로고침", width="stretch"):
        storage.invalidate_cache()
        st.cache_data.clear()
        st.rerun()


def render_manual_upload(*, storage: Any) -> list[Any]:
    """JSON/CSV 직접 추가 UI를 렌더링하고 업로드 객체를 반환합니다."""
    st.sidebar.markdown("##### JSON/CSV 직접 추가")
    uploaded = st.file_uploader(
        "파일 추가",
        type=["json", "csv", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="hidden_manual_upload",
    )

    json_uploads = [
        f for f in (uploaded or [])
        if f.name.lower().endswith((".json", ".txt"))
    ]

    if st.button(
        "선택한 JSON 영구 저장",
        width="stretch",
        disabled=not json_uploads,
    ):
        saved, errors = 0, []
        for file in json_uploads:
            try:
                storage.save_uploaded_json(
                    file.name,
                    file.getvalue(),
                    upload_cloud=True,
                )
                saved += 1
            except Exception as exc:
                errors.append(f"{file.name}: {exc}")

        if saved:
            storage.write_last_sync(
                source="manual_upload",
                details={"saved": saved},
            )
            st.success(f"{saved}개 파일을 영구 저장했습니다.")
            st.cache_data.clear()

        for error in errors:
            st.error(error)

        if saved and not errors:
            st.rerun()

    return uploaded or []


def render_sidebar(
    *,
    auth_name: str,
    auth_email: str,
    storage: Any,
    storage_status: Any,
    sync_journal_callback: Callable[[], Any] | None = None,
    sync_trackman_callback: Callable[[], Any] | None = None,
    app_dir: Any | None = None,
) -> list[Any]:
    """Sidebar 전체 UI 진입점."""
    render_sidebar_header(
        auth_name=auth_name,
        auth_email=auth_email,
    )
    render_storage_status(storage_status)
    render_data_actions(
        storage=storage,
        storage_status=storage_status,
        sync_journal_callback=sync_journal_callback,
        sync_trackman_callback=sync_trackman_callback,
        app_dir=app_dir,
    )
    return render_manual_upload(storage=storage)
