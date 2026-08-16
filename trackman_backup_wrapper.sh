#!/bin/bash
set -u

PROJECT_DIR="/Users/justin/Desktop/python/training_golf/trackman_dashboard_project"
PYTHON="$PROJECT_DIR/.venv/bin/python"
RUNNER="$PROJECT_DIR/scheduled_dodos_pipeline.py"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/trackman_backup.log"

mkdir -p "$LOG_DIR"

echo "==================================================" >> "$LOG_FILE"
echo "[$(/bin/date '+%Y-%m-%d %H:%M:%S')] wrapper 시작" >> "$LOG_FILE"

if [ ! -x "$PYTHON" ]; then
  echo "가상환경 Python을 찾을 수 없습니다: $PYTHON" >> "$LOG_FILE"
  exit 2
fi

if [ ! -f "$RUNNER" ]; then
  echo "예약 실행 파일을 찾을 수 없습니다: $RUNNER" >> "$LOG_FILE"
  exit 2
fi

/usr/bin/caffeinate -imsu \
  "$PYTHON" -u "$RUNNER" \
  --project-dir "$PROJECT_DIR" \
  --retries 2 \
  --retry-delay 60 \
  >> "$LOG_FILE" 2>&1

EXIT_CODE=$?
echo "[$(/bin/date '+%Y-%m-%d %H:%M:%S')] wrapper 종료 코드: $EXIT_CODE" >> "$LOG_FILE"
exit "$EXIT_CODE"
