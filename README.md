<p align="center">
  <img src="./assets/docs/openresearcher-logo.webp" height="82" style="vertical-align: middle;">
  <img src="./assets/docs/openresearcher-title.svg" height="82" style="vertical-align: middle;">
</p>

<div align="center" style="line-height: 1; margin-top: 16px;">

[![Models](https://img.shields.io/badge/Models-CC9900?style=for-the-badge&logo=huggingface&logoColor=ffffff)](https://huggingface.co/OpenResearcher)
[![Blog](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://openresearcher.github.io/)

</div>

<p align="center">
🤗 <a href="https://huggingface.co/OpenResearcher" target="_blank">HuggingFace</a> ｜
📰 <a href="https://openresearcher.github.io/" target="_blank">Blog</a>
</p>

# Introduction

Evaluation framework for deep research agents with local/remote search engines and multiple LLM backends.

## 📋 Table of Contents

- [🛠 Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Installation](#installation)
- [Configuration](#configuration)
- [Get Started](#get-started)
- [Benchmarks](#benchmarks)
- [Evaluation](#evaluation)

## 🛠 Environment Setup
### Installation 
```bash
sudo apt install -y openjdk-21-jdk
uv venv --python 3.12
source .venv/bin/activate

# install tevatron for BrowserComp-plus 
git clone https://github.com/texttron/tevatron.git
cd tevatron
uv pip install -e .
cd ..

# install all dependencies automatically
uv pip install -e .
```

### DeepResearch Benchmarks Preparation

Run the setup script to download all deepresearch benchmarks automatically:

```bash
bash setup.sh
```

**This script will:**
- ✅ Verify Python 3.12 virtual environment and automatically install any missing dependencies
- ✅ Downlaod all benchmark datasets from HuggingFace and set up the directory structure

## Configuration

### API Keys

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

## Get Started

### Example 1: BrowseComp-Plus with Local Search Engine

Complete workflow using local Dense search:

```bash
# Terminal 1: Start local Dense search service on port 8000
bash scripts/start_search_service.sh dense 8000

# Terminal 2: Start vLLM servers (requires 4 GPUs)
# TP=2, deploy 2 servers starting from port 8001 on GPUs 0,1,2,3
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3

# Terminal 3: Run agent
bash run_agent.sh results/browsecomp_plus/Researcher_dense 8001 2 browsecomp_plus local OpenResearcher/Nemotron-3-Nano-30B-A3B
```

What this does:
- Deploys Dense search on port 8000 as virtual search engine
- Launches 2 vLLM servers (ports 8001, 8002) with TP=2 across 4 GPUs
- Runs agent with load balancing across both servers

### Example 2: Using Serper API (No Local Search Needed)

Run with Serper Google Search API:

```bash
# Terminal 1: Start vLLM servers (requires 4 GPUs)
bash scripts/start_nemotron_servers.sh 2 8001 0,1,2,3

# Terminal 2: Run agent with serper search backend
bash run_agent.sh results/gaia/OpenResearcher_serper 8001 2 gaia serper OpenResearcher/Nemotron-3-Nano-30B-A3B
```

**Browser Backend Options:**
- `local` - Use local BM25/Dense search service (for BrowseComp-Plus)
- `serper` - Use Serper Google Search API (for all other benchmarks)

For other parameters, please refer to [assets/docs/parameter.md](assets/docs/parameter.md).

## Benchmarks

| Benchmark | Dataset Key | Size | Language | Search Backend | Description |
|-----------|-------------|------|----------|----------------|-------------|
| **BrowseComp** | `browsecomp` | 1266 | EN | serper | OpenAI public browse benchmark |
| **BrowseComp-Plus** | `browsecomp-plus` | 830 | EN | local | Deep-research benchmark isolating retriever and LLM agent effects |
| **GAIA-text** | `gaia` | 103 | EN | serper | Text-only subset of GAIA benchmark (dev split) |
| **XBench** | `xbench` | 100 | ZH | serper | DeepSearch benchmark with encrypted test cases |
| **SealQA-ref** | `seal_ref` | 111 | EN | serper | With reference URLs |

For other benchmarks, please refer to [assets/docs](assets/docs).

### Quick Commands

| Scenario | Command |
|----------|---------|
| **BrowseComp** | `bash run_agent.sh results/browsecomp 8001 2 browsecomp serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **BrowseComp+ (BM25)** | `bash scripts/start_search_service.sh bm25 8000` then `bash run_agent.sh results/browsecomp-plus/OpenResearcher_bm25 8001 2 browsecomp-plus local OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **BrowseComp+ (Dense)** | `bash scripts/start_search_service.sh dense 8000` then `bash run_agent.sh results/browsecomp-plus/OpenResearcher_dense 8001 2 browsecomp-plus local OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **GAIA** | `bash run_agent.sh results/gaia 8001 2 gaia serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |
| **XBench** | `bash run_agent.sh results/xbench 8001 2 xbench serper OpenResearcher/Nemotron-3-Nano-30B-A3B` |

## Evaluation

After running experiments, evaluate results:

```bash
python eval.py --input_dir results/browsecomp_plus_dense/OpenResearcher
```

## License

