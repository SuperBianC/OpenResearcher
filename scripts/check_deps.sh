#!/usr/bin/env bash
# 检查 Offline Search Service 所需的全部依赖
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT="$(dirname "$SCRIPT_DIR")"

PYTHON="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
PYTHON="${PYTHON:-python3}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok(){ echo -e "  ${GREEN}✓${NC} $1"; }
warn(){ echo -e "  ${YELLOW}?${NC} $1"; }
fail(){ echo -e "  ${RED}✗${NC} $1"; MISSING+=("$2"); }

MISSING=()

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Dependency Check — Offline Search Service"
echo "  Python: $("$PYTHON" --version 2>&1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Python packages ───────────────────────────────────────────
echo ""
echo "【Python 包】"
check_pkg() {
  local import_name="$1" pip_name="$2"
  "$PYTHON" -c "import $import_name" 2>/dev/null \
    && ok "$pip_name" \
    || fail "$pip_name   →  pip install $pip_name" "$pip_name"
}

# ── 必装（BM25 + Dense 共用）─────────────────────────────────
check_pkg fastapi        fastapi
check_pkg uvicorn        uvicorn
check_pkg yaml           pyyaml
check_pkg loguru         loguru
check_pkg duckdb         duckdb
check_pkg pyserini       pyserini
check_pkg jnius_config   pyjnius
check_pkg pydantic       pydantic

# ── 仅 Dense 模式需要 ─────────────────────────────────────────
echo ""
echo "【Dense 专用（BM25 模式可跳过）】"
check_pkg faiss          faiss-cpu
check_pkg torch          torch
check_pkg transformers   transformers

# ── Java (JDK 11+) ────────────────────────────────────────────
echo ""
echo "【Java JDK】"
if command -v java &>/dev/null; then
  ok "Java: $(java -version 2>&1 | head -1)"
else
  fail "Java 未安装，需要 JDK 11+（推荐 21）" "java"
  MISSING+=("java")
fi

# ── tevatron（含 Lucene JAR）───────────────────────────────────
echo ""
echo "【tevatron & Lucene JARs】"
if [ -d "$ROOT/tevatron" ]; then
  ok "tevatron 目录存在: $ROOT/tevatron"
  JAR_COUNT=$(find "$ROOT/tevatron" -name "lucene-*.jar" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$JAR_COUNT" -ge 3 ]; then
    ok "Lucene JARs ($JAR_COUNT 个)"
  else
    fail "Lucene JARs 缺失（找到 $JAR_COUNT 个，需要 3 个）" "lucene-jars"
    MISSING+=("lucene-jars")
  fi
else
  fail "tevatron 目录不存在，需要 git clone + pip install" "tevatron"
  MISSING+=("tevatron")
fi

# ── 数据文件 ──────────────────────────────────────────────────
echo ""
echo "【数据文件（来自 search_config.yaml）】"
CONFIG="${SEARCH_CONFIG:-$ROOT/search_config.yaml}"
if [ -f "$CONFIG" ]; then
  PARQUET_PATH=$("$PYTHON" -c "
import yaml
with open('$CONFIG') as f: c=yaml.safe_load(f)
print(c['corpus']['parquet_path'])
" 2>/dev/null || echo "")

  import glob
  FILE_COUNT=$("$PYTHON" -c "
import glob, yaml
with open('$CONFIG') as f: c=yaml.safe_load(f)
print(len(glob.glob(c['corpus']['parquet_path'])))
" 2>/dev/null || echo "0")

  if [ "$FILE_COUNT" -gt 0 ]; then
    ok "Parquet 语料库（$FILE_COUNT 个文件）"
  else
    fail "Parquet 文件未找到: $PARQUET_PATH" "corpus"
    MISSING+=("corpus-parquet")
  fi

  ENGINE=$("$PYTHON" -c "
import yaml
with open('$CONFIG') as f: c=yaml.safe_load(f)
print(c['engine']['type'])
" 2>/dev/null || echo "unknown")

  if [ "$ENGINE" = "bm25" ]; then
    INDEX_DIR=$("$PYTHON" -c "
import yaml
with open('$CONFIG') as f: c=yaml.safe_load(f)
print(c['engine']['bm25']['index_dir'])
" 2>/dev/null || echo "")
    if [ -d "$INDEX_DIR" ] && [ "$(ls -A "$INDEX_DIR" 2>/dev/null)" ]; then
      ok "BM25 索引目录: $INDEX_DIR"
    else
      fail "BM25 索引目录为空或不存在: $INDEX_DIR" "bm25-index"
      MISSING+=("bm25-index")
    fi
  fi
else
  warn "未找到 search_config.yaml，跳过数据检查"
fi

# ── 总结 ──────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ ${#MISSING[@]} -eq 0 ]; then
  echo -e "  ${GREEN}✅ 所有依赖已就绪，可以启动服务！${NC}"
  echo ""
  echo "  bash scripts/start_search_webui.sh"
else
  echo -e "  ${RED}❌ 缺失 ${#MISSING[@]} 项，请按下方命令安装${NC}"
  echo ""
  echo "  ── BM25 模式最小依赖（无需 torch/faiss/tevatron）──"
  echo "  pip install fastapi uvicorn pyyaml loguru duckdb pyserini pyjnius pydantic"
  echo ""
  echo "  ── Lucene 高亮 JAR（放到 lucene.extra_dir 目录里）──"
  echo "  LUCENE=9.9.1; DIR=/path/to/your/lucene_jars"
  echo "  wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-highlighter/\$LUCENE/lucene-highlighter-\$LUCENE.jar -P \$DIR"
  echo "  wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-queries/\$LUCENE/lucene-queries-\$LUCENE.jar    -P \$DIR"
  echo "  wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-memory/\$LUCENE/lucene-memory-\$LUCENE.jar     -P \$DIR"
  echo ""
  echo "  ── Dense 模式额外依赖（BM25 不需要）──"
  echo "  pip install torch faiss-cpu transformers"
  echo "  git clone https://github.com/texttron/tevatron.git && pip install -e tevatron/"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
