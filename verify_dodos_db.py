from __future__ import annotations

import argparse

from dodos_supabase import (
    DodosSupabaseConfig,
    create_dodos_client,
    get_dodos_user,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-email", default="")
    args = parser.parse_args()

    config = DodosSupabaseConfig.load()
    email = (args.user_email or config.user_email).lower().strip()
    client = create_dodos_client(config)
    user = get_dodos_user(client, email=email)
    user_id = str(user["id"])

    sessions = (
        client.table("dodos_practice_sessions")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    shots = (
        client.table("dodos_shots")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    raw = (
        client.table("dodos_raw_files")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    verified = (
        client.table("dodos_raw_files")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("shot_db_verified", True)
        .limit(1)
        .execute()
    )
    failed = (
        client.table("dodos_raw_files")
        .select("source_file_name,archive_error,expected_shot_count,archived_shot_count")
        .eq("user_id", user_id)
        .eq("shot_db_verified", False)
        .execute()
    )

    print("=" * 58)
    print("DODOS DATABASE VERIFY")
    print("=" * 58)
    print(f"User              : {email}")
    print(f"Practice sessions : {int(sessions.count or 0):,}")
    print(f"Shots             : {int(shots.count or 0):,}")
    print(f"Raw registry      : {int(raw.count or 0):,}")
    print(f"Verified raw      : {int(verified.count or 0):,}")
    print(f"Unverified raw    : {len(failed.data or []):,}")
    print("=" * 58)

    if failed.data:
        print("\n검증 필요 파일:")
        for item in failed.data:
            print(
                " - "
                f"{item.get('source_file_name')}: "
                f"{item.get('archived_shot_count')}/"
                f"{item.get('expected_shot_count')} "
                f"{item.get('archive_error') or ''}"
            )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
