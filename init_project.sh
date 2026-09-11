#!/usr/bin/env bash
# One-shot setup for the KKM Diagnostic Router prototype.
#
#   ./init_project.sh              # venv + pip + npm + directories
#   ./init_project.sh --ingest     # also build the ChromaDB index (few minutes)
#
# Safe to re-run: nothing is deleted and existing files are left alone.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; OFF=$'\033[0m'
step() { printf "\n%s==> %s%s\n" "$BOLD" "$1" "$OFF"; }
ok()   { printf "  %s✓%s %s\n" "$GREEN" "$OFF" "$1"; }
warn() { printf "  %s!%s %s\n" "$YELLOW" "$OFF" "$1"; }
die()  { printf "  %s✗%s %s\n" "$RED" "$OFF" "$1" >&2; exit 1; }

RUN_INGEST=0
[[ "${1:-}" == "--ingest" ]] && RUN_INGEST=1

# ---------------------------------------------------------------- directories
step "Creating directories"
mkdir -p backend/data/raw_pdfs backend/data/raw_json backend/data/chroma_db \
         backend/app frontend/src/app frontend/src/components frontend/src/lib
ok "backend/data/{raw_pdfs,raw_json,chroma_db}"

# --------------------------------------------------------------------- python
step "Python environment"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null || die "python3 not found. brew install python"
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
ok "using $PYTHON_BIN (Python $PY_VERSION)"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
  ok "created .venv"
else
  ok ".venv already present"
fi
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --quiet --upgrade pip
step "Installing backend dependencies (torch + chromadb take a few minutes)"
python -m pip install --quiet -r backend/requirements.txt
ok "$(python -m pip list 2>/dev/null | wc -l | tr -d ' ') packages installed"

if [[ ! -f backend/.env ]]; then
  cp backend/.env.example backend/.env
  ok "created backend/.env (optional overrides only — no API key needed)"
else
  ok "backend/.env already present"
fi

# --------------------------------------------------------------------- inputs
step "Checking corpus inputs"
PDF_COUNT=$(find backend/data/raw_pdfs -name '*.pdf' | wc -l | tr -d ' ')
if [[ "$PDF_COUNT" == "0" ]]; then
  warn "no PDFs in backend/data/raw_pdfs/ — add the KKM CPG / QR / MTS / Paediatric PDFs"
else
  ok "$PDF_COUNT PDF(s) in backend/data/raw_pdfs/"
fi
if [[ -f backend/data/raw_json/fukkm_database.json ]]; then
  ok "fukkm_database.json present"
else
  warn "no fukkm_database.json — build it with: cd backend && python -m app.scrape_fukkm"
fi

# ------------------------------------------------------------------- frontend
step "Frontend dependencies"
command -v npm >/dev/null || die "npm not found. brew install node"
( cd frontend && npm install --no-fund --no-audit )
ok "node_modules installed"
if [[ ! -f frontend/.env.local ]]; then
  cp frontend/.env.local.example frontend/.env.local
  ok "created frontend/.env.local"
fi

# --------------------------------------------------------------------- ingest
if [[ "$RUN_INGEST" == "1" ]]; then
  step "Building the vector index"
  ( cd backend && python -m app.ingest --reset )
else
  step "Skipping ingestion (pass --ingest to build the index now)"
fi

cat <<EOF

${BOLD}Setup complete.${OFF}

  1. Build the index        ${BOLD}source .venv/bin/activate && cd backend && python -m app.ingest --reset${OFF}
  2. Start the API          ${BOLD}./backend/run.sh${OFF}                  -> http://127.0.0.1:8000/docs
  3. Start the UI           ${BOLD}cd frontend && npm run dev${OFF}        -> http://localhost:3000

The first Mode A/B request downloads the local MLX model weights from Hugging
Face (a few GB) and loads them into memory — expect it to be slow. Later
requests reuse the resident model.

EOF
