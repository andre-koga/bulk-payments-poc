#!/usr/bin/env bash
# Bootstrap + start API + Vite dev server.
# Usage: ./run.sh [--db /path/to/bulk.db] [--port 8000]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${ROOT}/backend"
FRONTEND="${ROOT}/frontend"
VENV="${BACKEND}/.venv"

DB_PATH="${BULK_DB:-/tmp/bulk.db}"
API_PORT="${API_PORT:-8000}"
VITE_PORT="${VITE_PORT:-5173}"

for cmd in python3 node npm; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "error: missing '$cmd'; install it and re-run ./run.sh" >&2
    exit 1
  }
done

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "==> Creating Python venv in backend/.venv"
  python3 -m venv "$VENV"
fi

echo "==> Installing backend (editable + dev deps)"
(
  cd "$BACKEND"
  "${VENV}/bin/pip" install -q --upgrade pip
  "${VENV}/bin/pip" install -q -e ".[dev]"
)

echo "==> Installing frontend dependencies"
(
  cd "$FRONTEND"
  npm install --silent
)

echo "==> Seeding demo database at ${DB_PATH}"
"${VENV}/bin/bulk-match" seed --db "$DB_PATH"

echo "==> Starting API on http://127.0.0.1:${API_PORT}  (Ctrl-C stops both)"
BULK_DB="$DB_PATH" "${VENV}/bin/bulk-match" serve --db "$DB_PATH" --port "$API_PORT" &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Starting Vite on http://localhost:${VITE_PORT}"
VITE_API_URL="http://127.0.0.1:${API_PORT}" \
  npm --prefix "$FRONTEND" run dev -- --port "$VITE_PORT"
