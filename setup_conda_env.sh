#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# 创建 conda 环境 offsearch 并安装 Dense Search 所需的全部依赖
#
# 用法:
#   bash setup_conda_env.sh
#
# 前置要求:
#   - conda 已安装
#   - NVIDIA GPU + CUDA 驱动 (12.1+)
#   - Java JDK 11+ (用于 pyserini / Lucene snippet highlighting)
#     如未安装: sudo apt install -y openjdk-21-jdk
# ─────────────────────────────────────────────────────────────
set -e

ENV_NAME="offsearch"
PYTHON_VERSION="3.12"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TEVATRON_DIR="${SCRIPT_DIR}/tevatron"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setting up conda env: ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. 创建 conda 环境 ──────────────────────────────────────
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "[INFO] conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[1/4] Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# ── 2. 激活环境 ──────────────────────────────────────────────
echo "[2/4] Activating conda env '${ENV_NAME}'..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "  Python: $(which python) ($(python --version))"

# ── 3. 安装 pip 依赖 ─────────────────────────────────────────
echo "[3/4] Installing pip packages..."

# --- PyTorch (CUDA 12.1) ---
pip install torch --index-url https://download.pytorch.org/whl/cu121

# --- FAISS GPU ---
pip install faiss-gpu

# --- Web service ---
pip install "fastapi>=0.115.0" "uvicorn>=0.30.0" "pydantic>=2.0.0"

# --- Config & logging ---
pip install "pyyaml>=6.0" "loguru>=0.7.0"

# --- Corpus loading ---
pip install "duckdb>=0.10.0"

# --- BM25 / Lucene (also used for snippet highlighting in dense mode) ---
pip install "pyserini>=0.22.0" "pyjnius>=1.6.0"

# --- Embedding model ---
pip install "transformers>=4.40.0" "accelerate>=0.30.0" "sentencepiece>=0.2.0"

# --- tevatron dependencies ---
pip install "qwen-omni-utils>=0.0.8" "tqdm>=4.66.0" "numpy>=1.26.0"

# --- HuggingFace datasets (for loading corpus) ---
pip install "datasets>=4.5.0"

# ── 4. 从源码安装 tevatron ───────────────────────────────────
echo "[4/4] Installing tevatron from source..."
if [ -d "${TEVATRON_DIR}" ]; then
    pip install -e "${TEVATRON_DIR}"
    echo "  tevatron installed from: ${TEVATRON_DIR}"
else
    echo "[WARN] tevatron directory not found at ${TEVATRON_DIR}"
    echo "       Please clone it manually:"
    echo "         git clone https://github.com/texttron/tevatron.git ${TEVATRON_DIR}"
    echo "         pip install -e ${TEVATRON_DIR}"
fi

# ── Done ─────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Done! To use:"
echo ""
echo "     conda activate ${ENV_NAME}"
echo ""
echo "  Verify:"
echo "     python -c \"import torch; print('CUDA:', torch.cuda.is_available())\""
echo "     python -c \"import faiss; print('FAISS GPUs:', faiss.get_num_gpus())\""
echo "     python -c \"import tevatron; print('tevatron OK')\""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
