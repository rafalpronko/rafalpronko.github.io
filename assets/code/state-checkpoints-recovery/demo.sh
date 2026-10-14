#!/usr/bin/env bash
# Walk through two crashes and restarts in a throwaway directory.
# Usage: bash demo.sh   (from the examples/ directory, needs uv)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"
cat > tickets.json <<'JSON'
[{"id": "T-3", "text": "  Refund for DOUBLE charge  "},
 {"id": "T-1", "text": "Cannot log in after password reset"},
 {"id": "T-2", "text": "Invoice shows the wrong VAT number"}]
JSON
py() { uv run --project "$HERE" --python 3.11 python "$@"; }
worker() {
  py "$HERE/worker.py" --db "$1.db" --receiver "$1-receiver.db" \
     --input tickets.json --run-id "$1"
  echo "exit=$?"
}
show() { py -c 'import sqlite3, sys
print(sqlite3.connect(sys.argv[1]).execute(sys.argv[2]).fetchall())' "$@"; }

echo "== 1. crash inside the classify transaction, before COMMIT"
CRASH_AT=before_commit:classify worker run-a
echo "files: $(ls | tr '\n' ' ')"
show run-a.db "SELECT cursor FROM run"
show run-a.db "SELECT name FROM step_result ORDER BY step"
echo "== restart"
worker run-a

echo "== 2. effect delivered, crash before the ack is committed"
CRASH_AT=before_commit:notify worker run-b
show run-b.db "SELECT operation_id, status, receipt FROM operation"
echo "== restart replays the same operation id"
worker run-b
show run-b-receiver.db "SELECT operation_id, attempts, receipt FROM delivery"
show run-b.db "SELECT operation_id, status, receipt FROM operation"

echo "== 3. same run id, different input: refuse to resume"
echo '[]' > tickets.json
worker run-a
