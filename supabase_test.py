from __future__ import annotations

import sys

import streamlit as st
from supabase import Client, create_client


def main() -> int:
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]
        bucket_name = st.secrets["SUPABASE_BUCKET"]
    except KeyError as exc:
        print(f"secrets.toml 항목이 없습니다: {exc}")
        return 1

    try:
        client: Client = create_client(supabase_url, supabase_key)

        # 버킷 내부 파일 목록 조회
        result = client.storage.from_(bucket_name).list()

        print("Supabase 연결 성공")
        print(f"버킷: {bucket_name}")
        print(f"현재 파일 수: {len(result)}")

        for item in result:
            print(item)

        return 0

    except Exception as exc:
        print("Supabase 연결 실패")
        print(type(exc).__name__)
        print(exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())