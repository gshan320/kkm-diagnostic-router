#!/usr/bin/env bash
# Everything left that the Mac can do alone (no cloud tokens). Log: data/cards/overnight.log
set -u
cd "$(dirname "$0")/.."
source ../.venv/bin/activate
echo "=== start $(date)"
python -m app.cards_local                         # 1. extract the remaining packets on the local model
python -m app.cards_build --verify                 # 2. verify every quote, merge, write condition_cards.json
python -m app.references --check-cited             # 3. confirm the KKM-cited DOIs / URLs
for t in tests/test_*.py; do                       # 4. every unit suite
  m=$(basename "$t" .py); out=$(python -m tests.$m 2>&1)
  echo "== $m: $(echo "$out" | grep -c '  PASS  ') pass, $(echo "$out" | grep -c '  FAIL  ') fail"
  echo "$out" | grep "  FAIL  \|Traceback" | head -5
done
echo "=== end $(date)"
