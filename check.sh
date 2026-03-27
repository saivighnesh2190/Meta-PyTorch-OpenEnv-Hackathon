#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="${BASE_URL:-http://${HOST}:${PORT}}"
SERVER_LOG="${SERVER_LOG:-/tmp/delivery_env_check.log}"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing virtualenv at ${ROOT_DIR}/.venv"
  echo "Create it first:"
  echo "  python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt pytest"
  exit 1
fi

cd "${ROOT_DIR}"

echo "== tests =="
"${VENV_PYTHON}" -m pytest -q

echo "== starting server =="
"${VENV_PYTHON}" -m uvicorn api.app:app --host "${HOST}" --port "${PORT}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
sleep 3

if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "Server failed to start. Log:"
  cat "${SERVER_LOG}"
  exit 1
fi

echo "== health =="
curl -fsS "${BASE_URL}/health"
echo

echo "== tasks =="
curl -fsS "${BASE_URL}/tasks"
echo

echo "== reset =="
curl -fsS -X POST "${BASE_URL}/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy"}'
echo

echo "== sample step =="
curl -fsS -X POST "${BASE_URL}/step" \
  -H "Content-Type: application/json" \
  -d '{"action":{"action_type":"assign_order","order_id":"O-101","worker_id":"W-1"}}'
echo

echo "== baseline =="
"${VENV_PYTHON}" -m baseline.run_baseline --base-url "${BASE_URL}"

echo "== baseline endpoint =="
curl -fsS "${BASE_URL}/baseline"
echo

echo "== done =="
echo "Full verification completed successfully."
