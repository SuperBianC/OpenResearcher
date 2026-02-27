#!/usr/bin/env bash
# Usage:
#   bash scripts/start_search_webui.sh                            # 默认: search_config.yaml + YAML 里的 host/port
#   bash scripts/start_search_webui.sh my_config.yaml             # 自定义配置 + YAML 里的 host/port
#   bash scripts/start_search_webui.sh my_config.yaml 9090        # 自定义配置 + 覆盖端口
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-search_config.yaml}"
[[ "$CONFIG" != /* ]] && CONFIG="${PROJECT_ROOT}/${CONFIG}"

# 从 YAML 中读取 service.host / service.port，缺省值为 0.0.0.0:8000
read_host_port_from_yaml() {
  python - << 'PY' "$1"
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f) or {}
svc = cfg.get("service", {})
host = str(svc.get("host", "0.0.0.0"))
port = int(svc.get("port", 8000))
print(f"{host}:{port}")
PY
}

HP="$(read_host_port_from_yaml "$CONFIG")"
HOST_FROM_YAML="${HP%%:*}"
PORT_FROM_YAML="${HP##*:}"

# 允许第二个参数覆盖端口；host 始终以 YAML 为准
PORT="${2:-$PORT_FROM_YAML}"
HOST="$HOST_FROM_YAML"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Config : $CONFIG"
echo "  Host   : $HOST"
echo "  Port   : $PORT"
echo "  URL    : http://${HOST}:${PORT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

SEARCH_CONFIG="$CONFIG" \
  python -m uvicorn scripts.search_webui:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info
