#!/usr/bin/env bash
# Install the daily-refresh launchd job. Re-run safely; reload is idempotent.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
TEMPLATE="$REPO_ROOT/scripts/com.user.hedgefund.daily.plist.template"
DEST="$HOME/Library/LaunchAgents/com.user.hedgefund.daily.plist"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: $PYTHON_BIN not found. Run 'python3 -m venv .venv && .venv/bin/pip install -r requirements.txt' first." >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents"

sed -e "s|REPO_PATH|$REPO_ROOT|g" -e "s|PYTHON_PATH|$PYTHON_BIN|g" \
    "$TEMPLATE" > "$DEST"

# Reload (unload silently if not loaded)
launchctl unload "$DEST" 2>/dev/null || true
launchctl load "$DEST"

echo "Installed: $DEST"
echo "Fires:     Weekdays at 17:15 local"
echo "Logs:      $REPO_ROOT/output/launchd.{out,err}.log"
