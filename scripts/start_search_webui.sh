#!/usr/bin/env bash
# Usage:
#   bash scripts/start_search_webui.sh                        # 默认 search_config.yaml, port 8000
#   bash scripts/start_search_webui.sh my_config.yaml         # 自定义配置
#   bash scripts/start_search_webui.sh my_config.yaml 9090    # 自定义配置 + 端口
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-search_config.yaml}"
[[ "$CONFIG" != /* ]] && CONFIG="${PROJECT_ROOT}/${CONFIG}"

PORT="${2:-8000}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Config : $CONFIG"
echo "  URL    : http://localhost:${PORT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SEARCH_CONFIG="$CONFIG" \
  python -m uvicorn scripts.search_webui:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info
