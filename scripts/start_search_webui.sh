#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Start the Offline Search Web UI
#
# Usage:
#   bash scripts/start_search_webui.sh                   # uses search_config.yaml
#   bash scripts/start_search_webui.sh my_config.yaml    # custom config
#   bash scripts/start_search_webui.sh my_config.yaml 9090  # custom port
# ─────────────────────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-search_config.yaml}"
# Resolve to absolute path if relative
if [[ "$CONFIG" != /* ]]; then
  CONFIG="${PROJECT_ROOT}/${CONFIG}"
fi

# ── Read port from config (Python one-liner), with override from arg 2 ────────
_PY="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
_PY="${_PY:-python3}"
CFG_PORT=$("$_PY" -c "
import yaml
with open('$CONFIG') as f:
    c = yaml.safe_load(f)
print(c.get('service',{}).get('port', 8000))
" 2>/dev/null || echo "8000")
PORT="${2:-$CFG_PORT}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Offline Search Web UI"
echo "  Config : $CONFIG"
echo "  URL    : http://localhost:${PORT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$PROJECT_ROOT"

# ── Resolve Python: conda env > project venv > system python3 ────────────────
if [ -n "$CONDA_PREFIX" ]; then
  PYTHON="$CONDA_PREFIX/bin/python"
  echo "[INFO] Using conda env Python: $PYTHON  (env: ${CONDA_DEFAULT_ENV:-unknown})"
elif [ -f ".venv/bin/activate" ]; then
  echo "[INFO] Activating project venv (.venv)"
  source .venv/bin/activate
  PYTHON=python
else
  PYTHON=python3
  echo "[WARN] No conda env or .venv detected — using: $(which python3)"
fi

# ── Check pyyaml ──────────────────────────────────────────────────────────────
"$PYTHON" -c "import yaml" 2>/dev/null || {
  echo "[WARN] pyyaml not found, installing..."
  "$PYTHON" -m pip install pyyaml -q
}

# ── Auto-detect libjvm.dylib if JVM_PATH not already set ─────────────────────
if [ -z "$JVM_PATH" ]; then
  _JVM=$(find /opt/homebrew /Library/Java -name "libjvm.dylib" 2>/dev/null | head -1)
  if [ -n "$_JVM" ]; then
    export JVM_PATH="$_JVM"
    export JAVA_HOME="$(echo "$_JVM" | sed 's|/lib/server/libjvm.dylib||')"
    export PATH="$JAVA_HOME/bin:$PATH"
    echo "[INFO] JVM_PATH=$JVM_PATH"
  else
    echo "[WARN] libjvm.dylib not found. Set JVM_PATH manually if startup fails."
  fi
fi

# ── Launch ────────────────────────────────────────────────────────────────────
SEARCH_CONFIG="$CONFIG" \
  "$PYTHON" -m uvicorn scripts.search_webui:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
