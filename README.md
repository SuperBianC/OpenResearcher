# 🔬 GPT-OSS DeepResearch Evaluation

Evaluation framework for deep research agents with local/remote search engines and multiple LLM backends.

---

## 📋 Table of Contents

- [🗂️ Data Preparation](#data-preparation)
- [📦 Installation](#installation)
- [⚙️ Configuration](#configuration)
- [🚀 Get Started](#get-started)
- [📊 Benchmarks](#benchmarks)
- [📝 Script Parameters](#script-parameters)
- [📈 Evaluation](#evaluation)

---

## 🛠️ Environment Setup

### 💻 System Dependencies

```bash
sudo apt install -y openjdk-21-jdk
```

### 🐍 Python Dependencies

```bash
uv venv --python 3.12
source .venv/bin/activate
```

### 🔍 Tevatron (for BrowseComp-Plus local search)

```bash
uv pip install -e .

git clone https://github.com/texttron/tevatron.git
cd tevatron
uv pip install -e .
cd ..
```

---

## 🗂️ Data Preparation

### 🔎 Local Search Engine (For BrowseComp-Plus)

To run **BrowseComp-Plus** benchmark with local search, you need to download the queries, corpus, and indexes:

**1️⃣ Download BrowseComp-Plus Queries (Required):**
```bash
# Create data directory
mkdir -p Tevatron/browsecomp-plus/data

# Download queries (contains questions, answers, and relevance judgements)
hf download Tevatron/browsecomp-plus \
    --repo-type dataset \
    --local-dir Tevatron/browsecomp-plus
```

**2️⃣ Download BrowseComp-Plus Corpus:**
```bash
# Create corpus directory
mkdir -p Tevatron/browsecomp-plus-corpus/data
mkdir -p Tevatron/browsecomp-plus-indexes

# Download corpus (parquet files with document texts)
hf download Tevatron/browsecomp-plus-corpus \
    --repo-type dataset \
    --local-dir Tevatron/browsecomp-plus-corpus
```

**3️⃣ Download Search Indexes:**
```bash
# BM25 Index (lightweight, recommended)
hf download Tevatron/browsecomp-plus-indexes \
    --repo-type dataset \
    --include "bm25/*" \
    --local-dir Tevatron/browsecomp-plus-indexes

# Dense Index (optional, requires GPU)
hf download Tevatron/browsecomp-plus-indexes \
    --repo-type dataset \
    --include "qwen3-embedding-8b/*" \
    --local-dir Tevatron/browsecomp-plus-indexes
```

**4️⃣ Download Lucene JARs (for BM25 highlighting):**
```bash
mkdir -p tevatron
cd tevatron
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-highlighter/9.9.1/lucene-highlighter-9.9.1.jar
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-queries/9.9.1/lucene-queries-9.9.1.jar
wget https://repo1.maven.org/maven2/org/apache/lucene/lucene-memory/9.9.1/lucene-memory-9.9.1.jar
cd ..
```

---

## 📦 Installation

### ⚡ Quick Setup (Recommended)

Run the setup script to install all dependencies automatically:

```bash
bash setup.sh
```

**📌 This script will:**
- ✅ Install Python 3.12 virtual environment using `uv`
- ✅ Install all Python dependencies
- ✅ Install Tevatron for local search
- ✅ Download Lucene JARs
- ✅ Download corpus and indexes (optional, interactive)

---

## ⚙️ Configuration

### 🔑 API Keys

Copy the template and configure your API keys:

```bash
cp .env.template .env
```

Edit `.env`:
```bash
# Serper API (for web search when using browser_backend=serper)
SERPER_API_KEY=your_key        # Get from: https://serper.dev/

# OpenAI API (for evaluation scoring)
OPENAI_API_KEY=your_key        # Get from: https://platform.openai.com/api-keys
```

---

## 🚀 Get Started

### 📖 Example 1: BrowseComp-Plus with Local Search Engine

**Complete workflow using local Dense search:**

```bash
# Terminal 1: Start local Dense search service on port 8000
bash scripts/start_search_service.sh dense 8000

# Terminal 2: Start vLLM servers (requires 4 GPUs)
# TP=2, deploy 2 servers starting from port 8001 on GPUs 0,1,2,3
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3

# Terminal 3: Run agent
bash run_agent.sh results/browsecomp_plus/Researcher_dense 8001 2 browsecomp_plus local OpenResearcher/Nemotron-3-Nano-30B-A3B
```

**💡 What this does:**
- 🔍 Deploys Dense search on port 8000 as virtual search engine
- 🖥️ Launches 2 vLLM servers (ports 8001, 8002) with TP=2 across 4 GPUs
- 🤖 Runs agent with load balancing across both servers

### 📖 Example 2: Using Serper API (No Local Search Needed)

**Run with Serper Google Search API:**

```bash
# Terminal 1: Start vLLM servers (requires 4 GPUs)
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3

# Terminal 2: Run agent with serper search backend
bash run_agent.sh results/gaia/OpenResearcher_serper 8001 2 gaia serper OpenResearcher/Nemotron-3-Nano-30B-A3B
```

**🌐 Browser Backend Options:**
- `local` - Use local BM25/Dense search service (for BrowseComp-Plus)
- `serper` - Use Serper Google Search API (for all other benchmarks)

For other parameters, please refer to [📄 assets/docs/parameter.md](assets/docs/parameter.md).

---

## 📊 Benchmarks

| Benchmark | Dataset Key | Size | Language | Search Backend | Description |
|-----------|-------------|------|----------|----------------|-------------|
| 🌐 **BrowseComp** | `browsecomp` | 1266 | EN | serper | OpenAI public browse benchmark |
| 🔬 **BrowseComp-Plus** | `browsecomp-plus` | 830 | EN | local | Deep-research benchmark isolating retriever and LLM agent effects |
| 🤖 **GAIA-text** | `gaia` | 103 | EN | serper | Text-only subset of GAIA benchmark (dev split) |
| 🇨🇳 **XBench** | `xbench` | 100 | ZH | serper | DeepSearch benchmark with encrypted test cases |
| 📚 **SealQA-ref** | `seal_ref` | 111 | EN | serper | With reference URLs |

For other benchmarks, please refer to [📁 assets/docs](assets/docs).

### ⚡ Quick Commands

| Scenario | Command |
|----------|---------|
| 🌐 **BrowseComp** | `bash run_agent.sh results/browsecomp 8001 2 browsecomp serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| 🔬 **BrowseComp+ (BM25)** | `bash scripts/start_search_service.sh bm25 8000` then `bash run_agent.sh results/browsecomp-plus/OpenResearcher_bm25 8001 2 browsecomp-plus local OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| 🔬 **BrowseComp+ (Dense)** | `bash scripts/start_search_service.sh dense 8000` then `bash run_agent.sh results/browsecomp-plus/OpenResearcher_dense 8001 2 browsecomp-plus local OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| 🤖 **GAIA** | `bash run_agent.sh results/gaia 8001 2 gaia serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| 🇨🇳 **XBench** | `bash run_agent.sh results/xbench 8001 2 xbench serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |

---

## 📈 Evaluation

After running experiments, evaluate results:

```bash
python eval.py --input_dir results/browsecomp_plus_dense/OpenResearcher
```

---

## 📜 License

