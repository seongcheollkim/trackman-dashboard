from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright
except ImportError as exc:
    raise SystemExit(
        "playwright가 설치되어 있지 않습니다.\n"
        "설치 명령:\n"
        "  pip install playwright\n"
        "  playwright install chromium"
    ) from exc


AUTH_PATTERN = re.compile(
    r"(?im)^(\s*-H\s+['\"]authorization:\s*Bearer\s+)([^'\"]+)(['\"]\s*\\?\s*)$"
)
HEADER_PATTERN = re.compile(
    r"(?im)^\s*-H\s+['\"](?P<name>[^:]+):\s*(?P<value>.*?)['\"]\s*\\?\s*$"
)


def read_curl(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"cURL 파일을 찾을 수 없습니다: {path}")
    return path.read_text(encoding="utf-8")


def infer_portal_url(curl_text: str) -> str:
    headers: dict[str, str] = {}
    for match in HEADER_PATTERN.finditer(curl_text):
        headers[match.group("name").strip().lower()] = match.group("value").strip()

    for key in ("referer", "origin"):
        value = headers.get(key)
        if not value:
            continue
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"

    return "https://portal.trackmangolf.com"


def replace_or_add_bearer_token(path: Path, token: str) -> None:
    """기존 Authorization 헤더를 교체하고, 없으면 cURL 명령에 새로 추가합니다."""
    text = read_curl(path)

    updated, count = AUTH_PATTERN.subn(
        rf"\g<1>{token}\g<3>",
        text,
        count=1,
    )

    if count == 0:
        lines = text.splitlines(keepends=True)
        if not lines or not lines[0].lstrip().startswith("curl "):
            raise RuntimeError(
                f"{path.name}이 올바른 cURL 형식이 아닙니다."
            )

        newline = "\r\n" if lines[0].endswith("\r\n") else "\n"
        first_line = lines[0].rstrip("\r\n")

        # 첫 줄이 백슬래시로 이어지는 일반적인 Copy as cURL 형식이면
        # 그 바로 다음 줄에 Authorization 헤더를 삽입합니다.
        if first_line.rstrip().endswith("\\"):
            header_line = f"  -H 'authorization: Bearer {token}' \\\\{newline}"
            lines.insert(1, header_line)
        else:
            # 단일 행 curl이라면 첫 줄을 연속 명령으로 바꾼 뒤 헤더를 추가합니다.
            lines[0] = first_line + f" \\\\{newline}"
            header_line = f"  -H 'authorization: Bearer {token}'{newline}"
            lines.insert(1, header_line)

        updated = "".join(lines)

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(path)


def capture_access_token(
    *,
    portal_url: str,
    profile_dir: Path,
    timeout_seconds: int,
    headless: bool,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    captured_token: str | None = None

    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(profile_dir),
            channel="chrome",
            headless=headless,
            viewport={"width": 1440, "height": 900},
            args=["--disable-blink-features=AutomationControlled"],
        )

        def inspect_request(request) -> None:
            nonlocal captured_token
            if "api.trackmangolf.com" not in request.url:
                return

            authorization = request.headers.get("authorization", "")
            if authorization.lower().startswith("bearer "):
                token = authorization.split(" ", 1)[1].strip()
                if token:
                    captured_token = token

        context.on("request", inspect_request)
        page = context.pages[0] if context.pages else context.new_page()

        try:
            page.goto(portal_url, wait_until="domcontentloaded", timeout=60_000)
        except PlaywrightTimeoutError:
            # 로그인 화면이나 포털이 일부 리소스를 오래 기다려도 요청 감시는 계속합니다.
            pass

        print(
            "TrackMan 인증 확인 중입니다. 전용 Chrome 창이 열리면 필요한 경우 한 번 로그인하세요.",
            flush=True,
        )

        reload_at = time.monotonic() + 15
        while time.monotonic() < deadline and not captured_token:
            page.wait_for_timeout(500)

            # 로그인 완료 후 GraphQL 요청이 발생하도록 주기적으로 포털을 새로고침합니다.
            if time.monotonic() >= reload_at:
                try:
                    page.reload(wait_until="domcontentloaded", timeout=30_000)
                except PlaywrightTimeoutError:
                    pass
                reload_at = time.monotonic() + 15

        context.close()

    if not captured_token:
        raise RuntimeError(
            "TrackMan access token을 자동으로 가져오지 못했습니다. "
            "열린 Chrome 창에서 TrackMan 로그인을 완료한 뒤 다시 실행하세요."
        )

    return captured_token


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "TrackMan 포털의 로그인 세션을 재사용하여 최신 Bearer token을 감지하고 "
            "기존 cURL 파일의 인증 헤더를 자동 갱신합니다."
        )
    )
    parser.add_argument("--list-curl", required=True)
    parser.add_argument("--report-curl", required=True)
    parser.add_argument(
        "--profile-dir",
        default=".trackman_browser",
        help="Playwright 전용 Chrome 로그인 프로필 저장 경로",
    )
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="브라우저 창을 숨깁니다. 최초 로그인 전에는 사용하지 마세요.",
    )
    args = parser.parse_args()

    list_curl = Path(args.list_curl).resolve()
    report_curl = Path(args.report_curl).resolve()
    profile_dir = Path(args.profile_dir).resolve()

    try:
        portal_url = infer_portal_url(read_curl(list_curl))
        token = capture_access_token(
            portal_url=portal_url,
            profile_dir=profile_dir,
            timeout_seconds=max(30, args.timeout),
            headless=args.headless,
        )
        replace_or_add_bearer_token(list_curl, token)
        replace_or_add_bearer_token(report_curl, token)
    except Exception as exc:
        print(f"TrackMan 인증 자동 갱신 실패: {exc}", file=sys.stderr)
        return 1

    print("TrackMan 인증정보를 자동으로 갱신했습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
