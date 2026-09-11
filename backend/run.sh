#!/usr/bin/env bash
# Start the KKM Diagnostic Router API on http://127.0.0.1:8000
# Docs: http://127.0.0.1:8000/docs
set -euo pipefail

cd "$(dirname "$0")"

if [[ -d "../.venv" ]]; then
  # shellcheck source=/dev/null
  source "../.venv/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
  echo "Using already-active virtualenv: $VIRTUAL_ENV"
else
  echo "No virtualenv found. Run ./init_project.sh from the project root first." >&2
  exit 1
fi

# backend/.env is optional: every setting has a working default and there is
# no API key to supply. See .env.example for the overrides that exist.

CHUNKS=$(python - <<'PY'
try:
    from app.ingest import get_collection
    print(get_collection().count())
except Exception:
    print(0)
PY
)
if [[ "$CHUNKS" == "0" ]]; then
  echo "WARNING: the vector index is empty. Run: python -m app.ingest --reset" >&2
else
  echo "Vector index: $CHUNKS chunks."
fi

exec uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
