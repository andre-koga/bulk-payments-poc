#!/usr/bin/env bash
# Bootstrap deps, seed demo DB, start API + Vite dev server.
#
# Usage:
#   ./run.sh
#   ./run.sh --db /tmp/bulk.db --port 8000 --vite-port 5173
#   ./run.sh --no-seed          # skip re-seeding (keep existing DB)
#
# Environment (optional): create .env in repo root, e.g.
#   OPENAI_API_KEY=sk-...
#   LANGCHAIN_TRACING_V2=true
#   LANGSMITH_API_KEY=lsv2_...
#   LANGSMITH_PROJECT=bulk-payments-poc
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${ROOT}/backend"
FRONTEND="${ROOT}/frontend"
VENV="${VENV:-${BACKEND}/.venv}"

DB_PATH="${BULK_DB:-/tmp/bulk.db}"
API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-127.0.0.1}"
VITE_PORT="${VITE_PORT:-5173}"
NO_SEED=0

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)
      DB_PATH="$2"
      shift 2
      ;;
    --port)
      API_PORT="$2"
      shift 2
      ;;
    --vite-port)
      VITE_PORT="$2"
      shift 2
      ;;
    --no-seed)
      NO_SEED=1
      shift
      ;;
    -h | --help)
      usage 0
      ;;
    *)
      echo "error: unknown argument '$1' (try --help)" >&2
      exit 1
      ;;
  esac
done

for cmd in python3 node npm curl; do
  command -v "$cmd" >/dev/null 2>&1 || {
    echo "error: missing '$cmd'; install it and re-run ./run.sh" >&2
    exit 1
  }
done

if [[ -f "${ROOT}/.env" ]]; then
  echo "==> Loading ${ROOT}/.env"
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "==> Creating Python venv at ${VENV}"
  python3 -m venv "$VENV"
fi

echo "==> Installing backend (editable + dev + agent)"
(
  cd "$BACKEND"
  "${VENV}/bin/pip" install -q --upgrade pip
  "${VENV}/bin/pip" install -q -e ".[dev,agent]"
)

echo "==> Installing frontend dependencies"
(
  cd "$FRONTEND"
  npm install --silent
)

export BULK_DB="$DB_PATH"

if [[ "$NO_SEED" -eq 0 ]]; then
  echo "==> Seeding demo database at ${DB_PATH}"
  "${VENV}/bin/bulk-match" seed --db "$DB_PATH"
else
  echo "==> Skipping seed (--no-seed); using ${DB_PATH}"
  "${VENV}/bin/bulk-match" init-db --db "$DB_PATH" 2>/dev/null || true
fi

API_URL="http://${API_HOST}:${API_PORT}"

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "$API_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "==> Starting API at ${API_URL}"
BULK_DB="$DB_PATH" "${VENV}/bin/bulk-match" serve --db "$DB_PATH" --host "$API_HOST" --port "$API_PORT" &
API_PID=$!

echo "==> Waiting for API health"
for _ in $(seq 1 40); do
  if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
if ! curl -sf "${API_URL}/health" >/dev/null 2>&1; then
  echo "error: API did not become ready at ${API_URL}/health" >&2
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" ]] && [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "warning: OPENAI_API_KEY / ANTHROPIC_API_KEY not set — 'Resolve with AI' will fail"
fi

echo "==> Starting Vite at http://localhost:${VITE_PORT}  (Ctrl-C stops API + UI)"
export VITE_API_URL="${API_URL}"
exec npm --prefix "$FRONTEND" run dev -- --port "$VITE_PORT" --host 127.0.0.1
