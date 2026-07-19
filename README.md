# TrackMan Dashboard v2.0

## 주요 변경점

- TrackMan 보고서를 로컬 `data/trackman_reports`에 저장
- 신규 보고서를 Supabase Storage에 자동 백업
- Streamlit Cloud 재시작 시 Supabase에서 자동 복원
- 로컬 전체 보고서 일괄 백업 및 클라우드 새로고침
- 직접 추가한 JSON도 영구 저장
- Parquet 캐시 유지

## 1. Supabase 버킷 생성

Supabase Dashboard > Storage에서 비공개 버킷을 생성합니다.

- Bucket name: `trackman-reports`
- Public bucket: Off
- MIME type 제한을 사용한다면 `application/json` 허용

## 2. 로컬 Secrets 설정

프로젝트에 `.streamlit/secrets.toml`을 만들고 `secrets.toml.example` 내용을 복사합니다.

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_KEY = "YOUR_KEY"
SUPABASE_BUCKET = "trackman-reports"
```

`.streamlit/secrets.toml`은 GitHub에 올리지 마세요.

## 3. Storage 정책

비공개 버킷에서 anon key를 사용할 경우 `storage.objects`에 해당 버킷의 SELECT/INSERT/UPDATE 정책이 필요합니다. 개인 전용 앱에서는 Streamlit Cloud Secrets에 service_role key를 넣는 방법도 있지만, 키가 외부에 노출되지 않도록 반드시 서버 측 Secrets에서만 사용해야 합니다.

## 4. 파일 교체

- `streamlit_app_v2.py` → `streamlit_app.py`
- `trackman_storage_v2.py` → `trackman_storage.py`
- `trackman_sync_v2.py` → `trackman_sync.py`
- `requirements_v2.txt` → `requirements.txt`

## 5. 실행

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

## 최초 데이터 이전

기존 로컬 보고서가 있다면 앱 사이드바에서 `로컬 데이터를 Supabase에 백업`을 한 번 누릅니다. 이후 신규 TrackMan 동기화 데이터는 자동으로 클라우드에 백업됩니다.
