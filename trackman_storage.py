from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from supabase import Client, create_client


@dataclass(frozen=True)
class StorageStatus:
    report_count: int
    last_sync: datetime | None
    cache_exists: bool
    report_dir: Path
    cloud_configured: bool
    cloud_connected: bool
    cloud_report_count: int | None
    cloud_shot_count: int | None
    cloud_updated_at: datetime | None
    cloud_error: str | None


@dataclass(frozen=True)
class CloudSyncResult:
    uploaded: int = 0
    downloaded: int = 0
    skipped: int = 0
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class TrackmanStorage:
    """로컬 JSON/Parquet 캐시와 Supabase Storage 백업을 함께 관리합니다."""

    MANIFEST_VERSION = "2.1"
    MANIFEST_NAME = "manifest.json"

    def __init__(
        self,
        base_dir: str | Path | None = None,
        *,
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        bucket: str | None = None,
        cloud_prefix: str = "reports",
    ) -> None:
        project_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.project_dir = project_dir.resolve()
        self.data_dir = self.project_dir / "data"
        self.report_dir = self.data_dir / "trackman_reports"
        self.cache_path = self.data_dir / "trackman_cache.parquet"
        self.cache_meta_path = self.data_dir / "trackman_cache_meta.json"
        self.last_run_path = self.report_dir / "_last_run.json"

        self.supabase_url = (supabase_url or os.getenv("SUPABASE_URL", "")).strip().rstrip("/")
        self.supabase_key = (supabase_key or os.getenv("SUPABASE_KEY", "")).strip()
        self.bucket = (bucket or os.getenv("SUPABASE_BUCKET", "trackman-reports")).strip()
        self.cloud_prefix = cloud_prefix.strip("/") or "reports"
        self._client: Client | None = None
        self.ensure_directories()

    @property
    def cloud_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_key and self.bucket)

    @property
    def client(self) -> Client:
        self._require_cloud()
        if self._client is None:
            self._client = create_client(self.supabase_url, self.supabase_key)
        return self._client

    @property
    def cloud_bucket(self):
        return self.client.storage.from_(self.bucket)

    def ensure_directories(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def report_files(self) -> list[Path]:
        return sorted(p for p in self.report_dir.glob("*.json") if not p.name.startswith("_"))

    def report_names(self) -> set[str]:
        return {p.name for p in self.report_files()}

    def status(self, *, check_cloud: bool = True) -> StorageStatus:
        files = self.report_files()
        last_sync = self._read_last_sync()
        if last_sync is None and files:
            last_sync = datetime.fromtimestamp(max(p.stat().st_mtime for p in files), tz=timezone.utc)

        cloud_connected = False
        cloud_count: int | None = None
        cloud_shots: int | None = None
        cloud_updated: datetime | None = None
        cloud_error: str | None = None

        if check_cloud and self.cloud_configured:
            try:
                manifest = self.read_cloud_manifest()
                if manifest:
                    cloud_count = self._safe_int(manifest.get("session_count"))
                    cloud_shots = self._safe_int(manifest.get("shot_count"))
                    cloud_updated = self._parse_datetime(manifest.get("updated_at"))
                else:
                    cloud_count = len(self.list_cloud_reports())
                cloud_connected = True
            except Exception as exc:
                cloud_error = str(exc)

        return StorageStatus(
            report_count=len(files),
            last_sync=last_sync,
            cache_exists=self.cache_path.exists(),
            report_dir=self.report_dir,
            cloud_configured=self.cloud_configured,
            cloud_connected=cloud_connected,
            cloud_report_count=cloud_count,
            cloud_shot_count=cloud_shots,
            cloud_updated_at=cloud_updated,
            cloud_error=cloud_error,
        )

    def _require_cloud(self) -> None:
        if not self.cloud_configured:
            raise RuntimeError("Supabase 설정이 없습니다. SUPABASE_URL, SUPABASE_KEY, SUPABASE_BUCKET을 등록해 주세요.")

    def _cloud_object_name(self, filename: str) -> str:
        return f"{self.cloud_prefix}/{Path(filename).name}"

    def list_cloud_reports(self) -> list[str]:
        """Supabase Storage의 reports 폴더에 있는 TrackMan JSON 파일명을 반환합니다."""
        self._require_cloud()
        offset = 0
        names: list[str] = []
        while True:
            items = self.cloud_bucket.list(
                self.cloud_prefix,
                {"limit": 1000, "offset": offset, "sortBy": {"column": "name", "order": "asc"}},
            )
            if not isinstance(items, list):
                raise RuntimeError("Supabase 파일 목록 응답 형식이 올바르지 않습니다.")
            for item in items:
                name = str(item.get("name", ""))
                if name.endswith(".json") and not name.startswith("_"):
                    names.append(Path(name).name)
            if len(items) < 1000:
                break
            offset += len(items)
        return sorted(set(names))

    def upload_report(self, path: str | Path, *, overwrite: bool = True) -> None:
        self._require_cloud()
        source = Path(path)
        if not source.exists() or source.suffix.lower() != ".json":
            raise FileNotFoundError(f"업로드할 JSON 파일을 찾을 수 없습니다: {source}")
        self.cloud_bucket.upload(
            self._cloud_object_name(source.name),
            source.read_bytes(),
            {"content-type": "application/json", "upsert": "true" if overwrite else "false"},
        )

    def download_report(self, filename: str, *, overwrite: bool = False) -> Path:
        self._require_cloud()
        target = self.report_dir / Path(filename).name
        if target.exists() and not overwrite:
            return target
        content = self.cloud_bucket.download(self._cloud_object_name(target.name))
        if not isinstance(content, (bytes, bytearray)):
            raise RuntimeError(f"{target.name} 다운로드 응답이 올바르지 않습니다.")
        self._atomic_write(target, bytes(content))
        return target

    def read_cloud_manifest(self) -> dict[str, Any] | None:
        self._require_cloud()
        try:
            content = self.cloud_bucket.download(self.MANIFEST_NAME)
        except Exception as exc:
            text = str(exc).lower()
            if "not found" in text or "404" in text or "object not found" in text:
                return None
            raise
        try:
            payload = json.loads(bytes(content).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise RuntimeError(f"Supabase manifest.json 해석 실패: {exc}") from exc
        return payload if isinstance(payload, dict) else None

    def write_cloud_manifest(self) -> dict[str, Any]:
        self._require_cloud()
        reports = self.report_files()
        payload = {
            "version": self.MANIFEST_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_count": len(reports),
            "shot_count": sum(self._count_report_shots(path) for path in reports),
            "reports": [path.name for path in reports],
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.cloud_bucket.upload(
            self.MANIFEST_NAME,
            content,
            {"content-type": "application/json", "upsert": "true"},
        )
        return payload

    def upload_local_reports(self, paths: Iterable[Path] | None = None) -> CloudSyncResult:
        if not self.cloud_configured:
            return CloudSyncResult(errors=("Supabase 설정이 없습니다.",))
        errors: list[str] = []
        uploaded = skipped = 0
        try:
            remote = set(self.list_cloud_reports())
        except Exception as exc:
            return CloudSyncResult(errors=(str(exc),))

        selected = list(paths) if paths is not None else self.report_files()
        for path in selected:
            if path.name in remote:
                skipped += 1
                continue
            try:
                self.upload_report(path, overwrite=False)
                uploaded += 1
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        if not errors:
            try:
                self.write_cloud_manifest()
            except Exception as exc:
                errors.append(f"manifest.json: {exc}")
        return CloudSyncResult(uploaded=uploaded, skipped=skipped, errors=tuple(errors))

    def pull_cloud_reports(self) -> CloudSyncResult:
        if not self.cloud_configured:
            return CloudSyncResult(errors=("Supabase 설정이 없습니다.",))
        errors: list[str] = []
        downloaded = skipped = 0
        try:
            manifest = self.read_cloud_manifest()
            remote = manifest.get("reports", []) if manifest else self.list_cloud_reports()
            remote = sorted({Path(str(name)).name for name in remote if str(name).endswith(".json")})
        except Exception as exc:
            return CloudSyncResult(errors=(str(exc),))

        local = self.report_names()
        for filename in remote:
            if filename in local:
                skipped += 1
                continue
            try:
                self.download_report(filename)
                downloaded += 1
            except Exception as exc:
                errors.append(f"{filename}: {exc}")
        if downloaded:
            self.invalidate_cache()
        return CloudSyncResult(downloaded=downloaded, skipped=skipped, errors=tuple(errors))

    def save_uploaded_json(self, filename: str, content: bytes, *, upload_cloud: bool = True) -> Path:
        """직접 추가한 TrackMan JSON을 로컬에 저장하고 선택적으로 Supabase에도 백업합니다."""
        payload = json.loads(content.decode("utf-8"))
        if not isinstance(payload, dict) or "StrokeGroups" not in payload:
            raise ValueError("StrokeGroups가 없는 파일입니다.")
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in Path(filename).stem)[:80]
        digest = hashlib.sha256(content).hexdigest()[:12]
        target = self.report_dir / f"{safe_stem}_{digest}.json"
        if not target.exists():
            normalized = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self._atomic_write(target, normalized)
            self.invalidate_cache()
        if upload_cloud and self.cloud_configured:
            self.upload_report(target, overwrite=True)
            self.write_cloud_manifest()
        return target

    def write_last_sync(self, *, source: str, details: dict[str, Any] | None = None) -> None:
        payload = {
            "finishedAt": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "details": details or {},
        }
        self._atomic_write(self.last_run_path, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_bytes(content)
        temp.replace(path)

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _count_report_shots(path: Path) -> int:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        groups = payload.get("StrokeGroups", []) if isinstance(payload, dict) else []
        if not isinstance(groups, list):
            return 0
        total = 0
        for group in groups:
            if isinstance(group, dict) and isinstance(group.get("Strokes"), list):
                total += len(group["Strokes"])
        return total

    def _read_last_sync(self) -> datetime | None:
        if not self.last_run_path.exists():
            return None
        try:
            payload = json.loads(self.last_run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        for key in ("finishedAt", "completedAt", "timestamp", "startedAt"):
            value = payload.get(key)
            if value:
                parsed = self._parse_datetime(value)
                if parsed is not None:
                    return parsed
        return datetime.fromtimestamp(self.last_run_path.stat().st_mtime, tz=timezone.utc)

    def _fingerprint(self) -> dict[str, dict[str, int]]:
        return {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in self.report_files()}

    def _read_cache_meta(self) -> dict[str, Any]:
        if not self.cache_meta_path.exists():
            return {}
        try:
            return json.loads(self.cache_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def cache_is_current(self) -> bool:
        return self.cache_path.exists() and self._read_cache_meta().get("fingerprint") == self._fingerprint()

    def invalidate_cache(self) -> None:
        for path in (self.cache_path, self.cache_meta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def load_rows(
        self,
        parser: Callable[[dict[str, Any]], list[dict[str, Any]]],
        *,
        force_rebuild: bool = False,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        if not force_rebuild and self.cache_is_current():
            try:
                frame = pd.read_parquet(self.cache_path)
                return frame.to_dict("records"), []
            except Exception:
                self.invalidate_cache()

        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for path in self.report_files():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict) or "StrokeGroups" not in payload:
                    errors.append(f"{path.name}: StrokeGroups가 없습니다.")
                    continue
                rows.extend(parser(payload))
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

        if rows:
            try:
                frame = pd.DataFrame(rows)
                frame.to_parquet(self.cache_path, index=False)
                self.cache_meta_path.write_text(
                    json.dumps(
                        {
                            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "row_count": len(frame),
                            "fingerprint": self._fingerprint(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                errors.append(f"캐시 생성 실패: {exc}")
        return rows, errors
