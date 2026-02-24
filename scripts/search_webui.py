"""
Offline Search Web UI
=====================
A FastAPI application that combines the search backend with a browser-based UI.

Usage:
    SEARCH_CONFIG=search_config.yaml uvicorn scripts.search_webui:app --host 0.0.0.0 --port 8000
    # or simply:
    bash scripts/start_search_webui.sh [search_config.yaml]
"""

import os
import re
import sys
import glob
from pathlib import Path
from threading import Lock
from typing import Dict, List, Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from loguru import logger

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Load YAML config ──────────────────────────────────────────────────────────
CONFIG_PATH = os.getenv("SEARCH_CONFIG", str(ROOT / "search_config.yaml"))

with open(CONFIG_PATH, "r") as _f:
    _cfg = yaml.safe_load(_f)

_svc    = _cfg.get("service", {})
_corpus = _cfg["corpus"]
_engine = _cfg["engine"]
_lucene = _cfg.get("lucene", {})

# ── Set environment variables BEFORE importing jnius / backend ────────────────
ENGINE_TYPE      = _engine["type"].lower()
CORPUS_NAME      = _corpus.get("name", "Custom Corpus")
MAX_SNIPPET_LEN  = int(_svc.get("max_snippet_len", 300))
LUCENE_EXTRA_DIR = _lucene.get("extra_dir", str(ROOT / "tevatron"))

os.environ["SEARCHER_TYPE"]       = ENGINE_TYPE
os.environ["CORPUS_PARQUET_PATH"] = _corpus["parquet_path"]
os.environ["MAX_SNIPPET_LEN"]     = str(MAX_SNIPPET_LEN)
os.environ["LUCENE_EXTRA_DIR"]    = LUCENE_EXTRA_DIR

if ENGINE_TYPE == "bm25":
    os.environ["LUCENE_INDEX_DIR"] = _engine["bm25"]["index_dir"]
elif ENGINE_TYPE == "dense":
    _d = _engine["dense"]
    os.environ["DENSE_INDEX_PATH"]       = _d["index_path"]
    os.environ["DENSE_MODEL_NAME"]       = _d["model_name"]
    os.environ["GPU_IDS"]                = ",".join(str(g) for g in _d.get("gpu_ids", ["0"]))
    os.environ["CUDA_VISIBLE_DEVICES"]   = str(_d.get("cuda_visible_devices", "0"))
else:
    raise ValueError(f"Unknown engine type: {ENGINE_TYPE!r}. Must be 'bm25' or 'dense'.")

# ── Lucene JARs (must happen before any jnius import) ────────────────────────
# setup_jar only needs: pyserini + pyjnius — no torch/faiss/tevatron required
from data_utils import setup_jar
setup_jar(LUCENE_EXTRA_DIR)

from jnius import autoclass

# ── Conditional backend import: BM25 uses a lightweight inline impl ───────────
# backend.py imports torch/faiss/tevatron/transformers at the TOP LEVEL even
# for BM25 mode — we bypass it entirely when engine == bm25.
if ENGINE_TYPE == "bm25":
    # ── Minimal inline BM25 backend (deps: duckdb + pyserini only) ───────────
    import glob as _glob
    import duckdb as _duckdb
    from abc import ABC, abstractmethod
    from pydantic import Field
    from pyserini.search.lucene import LuceneSearcher as _LuceneSearcher

    class SearchResult(BaseModel):
        docid: str
        score: float
        text: str | None = None

    class BaseSearcher(ABC):
        @abstractmethod
        def search(self, query: str, topn: int) -> List[SearchResult]: ...

    class Corpus:
        def __init__(self, parquet_path: str):
            self.parquet_path = parquet_path
            self.id2url: Dict[str, str] = {}
            self.url2id: Dict[str, str] = {}
            self.docid_to_text: Dict[str, str] = {}
            self._load()

        def _load(self):
            logger.info(f"Loading corpus from {self.parquet_path} ...")
            con = _duckdb.connect(database=":memory:", read_only=False)
            for docid, url, text in con.execute(
                f"SELECT docid, url, text FROM read_parquet('{self.parquet_path}')"
            ).fetchall():
                docid = str(docid)
                self.id2url[docid] = url
                self.url2id[url] = docid
                self.docid_to_text[docid] = text
            logger.info(f"Corpus loaded: {len(self.id2url):,} docs")

        def get_url_from_id(self, docid): return self.id2url.get(str(docid))
        def get_id_from_url(self, url):   return self.url2id.get(url)
        def get_text_from_id(self, docid): return self.docid_to_text.get(str(docid))

    class _BM25Searcher(BaseSearcher):
        def __init__(self, index_dir: str, corpus: Corpus):
            logger.info(f"Initialising BM25Searcher: {index_dir}")
            self._s = _LuceneSearcher(index_dir)
            self.corpus = corpus

        def search(self, query: str, topn: int) -> List[SearchResult]:
            return [
                SearchResult(docid=str(h.docid), score=h.score,
                             text=self.corpus.get_text_from_id(str(h.docid)))
                for h in self._s.search(query, k=topn)
            ]

    def _build_searcher(corpus: Corpus) -> BaseSearcher:
        return _BM25Searcher(os.environ["LUCENE_INDEX_DIR"], corpus)

else:
    # ── Dense mode: full backend (needs torch + faiss + tevatron) ────────────
    # Install: pip install torch faiss-cpu transformers
    #          git clone https://github.com/texttron/tevatron && pip install -e tevatron/
    from backend import Corpus, get_searcher as _build_searcher, BaseSearcher, SearchResult

# ── Lucene UnifiedHighlighter (for snippet generation) ───────────────────────
_ANALYZER   = None
_UH         = None
_QP         = None
_INIT_LOCK  = Lock()

_SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u00ab": '"', "\u00bb": '"',
    "\u300c": '"', "\u300d": '"', "\u300e": '"', "\u300f": '"',
}

def _drop_unpaired_quotes(q: str) -> str:
    for k, v in _SMART_QUOTES.items():
        q = q.replace(k, '"')
    out, in_q = [], False
    for ch in q:
        if ch == '"':
            in_q = not in_q
        out.append(ch)
    return "".join(out) if not in_q else "".join(out).replace('"', "")

def _highlight(query: str, content: str, max_passages: int = 50) -> str:
    global _ANALYZER, _UH, _QP
    if _ANALYZER is None:
        with _INIT_LOCK:
            if _ANALYZER is None:
                SA  = autoclass("org.apache.lucene.analysis.standard.StandardAnalyzer")
                UH  = autoclass("org.apache.lucene.search.uhighlight.UnifiedHighlighter")
                DPF = autoclass("org.apache.lucene.search.uhighlight.DefaultPassageFormatter")
                QP  = autoclass("org.apache.lucene.queryparser.classic.QueryParser")
                _ANALYZER = SA()
                fmt = DPF("<mark>", "</mark>", " … ", True)  # True = HTML-escape body, keep <mark> tags raw
                _UH = UH.builderWithoutSearcher(_ANALYZER).withFormatter(fmt).build()
                _QP = QP("content", _ANALYZER)
    try:
        q = _QP.parse(_drop_unpaired_quotes(query))
        s = _UH.highlightWithoutSearcher("content", q, content, max_passages)
        return str(s).strip() if s else content
    except Exception:
        return content

# ── Document front-matter parser ─────────────────────────────────────────────
_FM = re.compile(
    r"---\s*"
    r"(?:title:\s*(?P<title>.+?)\s*)"
    r"(?:author:\s*(?P<author>.+?)\s*)?"
    r"(?:date:\s*(?P<date>.+?)\s*)?"
    r"---\s*"
    r"(?P<content>.*?)(?=---|$)",
    re.DOTALL,
)

def _parse(docid: str, raw: str) -> Dict[str, str]:
    m = _FM.search(raw)
    if m:
        return {
            "title":   (m.group("title") or f"Doc {docid}").strip(),
            "content": (m.group("content") or "").strip(),
        }
    return {"title": f"Doc {docid}", "content": raw.strip()}

# ── Corpus + Searcher ─────────────────────────────────────────────────────────
corpus:   Corpus       = Corpus(os.environ["CORPUS_PARQUET_PATH"])
searcher: BaseSearcher = _build_searcher(corpus=corpus)
DOC_COUNT = len(corpus.id2url)
logger.info(f"Engine: {ENGINE_TYPE.upper()}  |  Corpus: {CORPUS_NAME}  |  Docs: {DOC_COUNT:,}")

# ── Pydantic models ───────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    topn: int = 10

class SearchItem(BaseModel):
    url:     str
    title:   str
    summary: str

class FetchRequest(BaseModel):
    url: str

class FetchResponse(BaseModel):
    title:   str
    content: str

# ── Embedded HTML UI ──────────────────────────────────────────────────────────
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Offline Search</title>
<style>
:root{
  --primary:#2563eb;--primary-h:#1d4ed8;--primary-bg:#eff6ff;
  --bg:#f1f5f9;--card:#ffffff;--border:#e2e8f0;
  --text:#1e293b;--muted:#64748b;--green:#059669;
  --err:#ef4444;--r:8px;--sh:0 1px 4px rgba(0,0,0,.08);
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:var(--bg);color:var(--text);height:100vh;
     display:flex;flex-direction:column;overflow:hidden}

/* ── Header ── */
header{background:#0f172a;color:#f8fafc;padding:0 20px;
       display:flex;align-items:center;gap:12px;height:52px;flex-shrink:0;
       box-shadow:0 2px 8px rgba(0,0,0,.3)}
.logo{font-size:17px;font-weight:700;display:flex;align-items:center;gap:8px;
      white-space:nowrap}
.logo-icon{font-size:20px}
.cfg-badges{display:flex;gap:8px;margin-left:auto;align-items:center;flex-wrap:wrap}
.badge{padding:3px 11px;border-radius:20px;font-size:11px;font-weight:700;
       letter-spacing:.3px;white-space:nowrap}
.b-engine{background:#7c3aed;color:#fff}
.b-corpus{background:#0284c7;color:#fff}
.b-docs  {background:#059669;color:#fff}
.b-path  {background:#374151;color:#d1d5db;font-weight:400;font-size:10px;
          max-width:340px;overflow:hidden;text-overflow:ellipsis}

/* ── Main layout ── */
main{display:flex;flex:1;overflow:hidden}

/* ── Left panel ── */
.left{width:440px;flex-shrink:0;display:flex;flex-direction:column;
      background:var(--card);border-right:1px solid var(--border)}

/* ── Search box ── */
.search-area{padding:14px 16px;border-bottom:1px solid var(--border);
             background:var(--card)}
.search-row{display:flex;gap:8px}
#q{flex:1;padding:9px 14px;border:2px solid var(--border);border-radius:var(--r);
   font-size:14px;outline:none;transition:border-color .15s;background:var(--bg)}
#q:focus{border-color:var(--primary);background:#fff}
.btn{padding:9px 18px;border-radius:var(--r);font-size:13px;font-weight:600;
     cursor:pointer;border:none;transition:all .15s}
.btn-primary{background:var(--primary);color:#fff}
.btn-primary:hover{background:var(--primary-h)}
.btn-primary:active{transform:scale(.97)}
.search-meta{margin-top:10px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.search-meta label{font-size:12px;color:var(--muted)}
select{padding:5px 8px;border:1px solid var(--border);border-radius:6px;
       font-size:12px;background:var(--bg);cursor:pointer}
#status{font-size:12px;color:var(--muted);margin-left:auto}

/* ── Results ── */
.results-list{flex:1;overflow-y:auto}
.res-header{padding:10px 16px;font-size:12px;color:var(--muted);
            background:var(--bg);border-bottom:1px solid var(--border);
            position:sticky;top:0}
.rcard{padding:13px 16px;border-bottom:1px solid var(--border);
       cursor:pointer;transition:background .12s;border-left:3px solid transparent}
.rcard:hover{background:var(--primary-bg)}
.rcard.active{background:var(--primary-bg);border-left-color:var(--primary)}
.rtitle{font-size:13px;font-weight:600;color:var(--primary);
        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:2px}
.rurl{font-size:11px;color:var(--green);white-space:nowrap;
      overflow:hidden;text-overflow:ellipsis;margin-bottom:5px}
.rsnip{font-size:12px;color:var(--muted);line-height:1.6;
       display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}

/* ── Right panel (viewer) ── */
.right{flex:1;display:flex;flex-direction:column;overflow:hidden;background:var(--card)}
.doc-header{padding:16px 24px;border-bottom:1px solid var(--border);background:var(--card);
            flex-shrink:0}
.doc-title{font-size:17px;font-weight:700;margin-bottom:4px;line-height:1.4}
.doc-url-row{display:flex;align-items:center;gap:8px}
.doc-url{font-size:12px;color:var(--green);word-break:break-all}
.copy-btn{padding:2px 8px;font-size:11px;border:1px solid var(--border);
          border-radius:4px;background:var(--bg);cursor:pointer;
          color:var(--muted);flex-shrink:0}
.copy-btn:hover{background:var(--border)}
.doc-body-wrap{flex:1;overflow-y:auto;padding:24px}
.doc-body{white-space:pre-wrap;font-size:13.5px;line-height:1.9;
          color:var(--text);font-family:'SF Mono',Menlo,Monaco,monospace;
          max-width:900px}

/* ── Placeholder ── */
.ph{display:flex;flex-direction:column;align-items:center;
    justify-content:center;height:100%;gap:10px;color:var(--muted);padding:24px;text-align:center}
.ph-icon{font-size:44px;opacity:.5}
.ph-text{font-size:14px}

/* ── Spinner ── */
.spin{width:16px;height:16px;border:2px solid var(--border);
      border-top-color:var(--primary);border-radius:50%;
      animation:rot .6s linear infinite;display:inline-block;vertical-align:middle}
@keyframes rot{to{transform:rotate(360deg)}}

/* ── Highlight (search term) ── */
mark{background:#fde68a;color:inherit;border-radius:3px;padding:0 2px;font-weight:600}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:3px}
::-webkit-scrollbar-track{background:transparent}

/* ── Toast ── */
#toast{position:fixed;bottom:20px;right:20px;background:#1e293b;color:#fff;
       padding:8px 16px;border-radius:6px;font-size:13px;
       opacity:0;transition:opacity .3s;pointer-events:none;z-index:999}
#toast.show{opacity:1}
</style>
</head>
<body>

<header>
  <div class="logo"><span class="logo-icon">🔍</span>Offline Search Service</div>
  <div class="cfg-badges" id="badges">
    <span class="badge" style="background:#374151;color:#9ca3af">Loading config…</span>
  </div>
</header>

<main>
  <!-- ── Left panel ── -->
  <div class="left">
    <div class="search-area">
      <div class="search-row">
        <input type="text" id="q" placeholder="Enter search query…" autocomplete="off" />
        <button class="btn btn-primary" onclick="doSearch()">Search</button>
      </div>
      <div class="search-meta">
        <label>Top-N</label>
        <select id="topn">
          <option value="5">5</option>
          <option value="10" selected>10</option>
          <option value="20">20</option>
          <option value="50">50</option>
        </select>
        <span id="status"></span>
      </div>
    </div>
    <div class="results-list" id="results">
      <div class="ph">
        <div class="ph-icon">🔎</div>
        <div class="ph-text">Enter a query above and press Search<br><small>or press Enter</small></div>
      </div>
    </div>
  </div>

  <!-- ── Right panel ── -->
  <div class="right" id="viewer">
    <div class="ph">
      <div class="ph-icon">📄</div>
      <div class="ph-text">Click a result to read the document</div>
    </div>
  </div>
</main>

<div id="toast"></div>

<script>
let _active = null;

// ── Load config & populate header badges ─────────────────────────────────────
async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    const c = await r.json();
    document.getElementById('badges').innerHTML = `
      <span class="badge b-engine">${c.engine_type}</span>
      <span class="badge b-corpus">${h(c.corpus_name)}</span>
      <span class="badge b-docs">${c.doc_count.toLocaleString()} docs</span>
      <span class="badge b-path" title="${h(c.parquet_path)}">${h(c.parquet_path)}</span>
    `;
  } catch(e) {
    document.getElementById('badges').innerHTML =
      '<span class="badge" style="background:#7f1d1d;color:#fca5a5">Config load failed</span>';
  }
}

// ── Search ───────────────────────────────────────────────────────────────────
async function doSearch() {
  const q    = document.getElementById('q').value.trim();
  const topn = parseInt(document.getElementById('topn').value);
  const st   = document.getElementById('status');
  const rl   = document.getElementById('results');
  if (!q) { document.getElementById('q').focus(); return; }

  st.innerHTML = '<span class="spin"></span> Searching…';
  rl.innerHTML = '';

  try {
    const resp = await fetch('/api/search', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({query: q, topn})
    });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    st.textContent = '';
    renderResults(data);
  } catch(e) {
    st.innerHTML = `<span style="color:var(--err)">Error: ${h(e.message)}</span>`;
  }
}

function renderResults(data) {
  const rl = document.getElementById('results');
  if (!data.results || data.results.length === 0) {
    rl.innerHTML = `<div class="ph"><div class="ph-icon">🙅</div><div class="ph-text">No results found</div></div>`;
    return;
  }
  rl.innerHTML = `<div class="res-header">
    ${data.count} result${data.count!==1?'s':''} for <strong>"${h(data.query)}"</strong>
  </div>`;
  data.results.forEach((item, i) => {
    const card = document.createElement('div');
    card.className = 'rcard';
    card.id = `rc${i}`;
    card.innerHTML = `
      <div class="rtitle" title="${h(item.title)}">${h(item.title||'Untitled')}</div>
      <div class="rurl"  title="${h(item.url)}">${h(item.url)}</div>
      <div class="rsnip">${item.summary}</div>
    `;
    /* item.summary is safe: Lucene HTML-escapes the body text (escape=True),
       only the <mark>…</mark> tags are kept raw for highlighting */
    card.addEventListener('click', () => viewDoc(item, card.id));
    rl.appendChild(card);
  });
}

// ── View document ─────────────────────────────────────────────────────────────
async function viewDoc(item, cardId) {
  if (_active) document.getElementById(_active)?.classList.remove('active');
  _active = cardId;
  document.getElementById(cardId)?.classList.add('active');

  const viewer = document.getElementById('viewer');
  viewer.innerHTML = `
    <div class="doc-header">
      <div class="doc-title">${h(item.title||'Untitled')}</div>
      <div class="doc-url-row">
        <div class="doc-url">${h(item.url)}</div>
        <button class="copy-btn" onclick="copyUrl('${h(item.url).replace(/'/g,"\\'")}')">Copy URL</button>
      </div>
    </div>
    <div class="doc-body-wrap">
      <div class="ph"><span class="spin"></span><span style="margin-left:8px">Loading content…</span></div>
    </div>
  `;

  try {
    const resp = await fetch('/api/content', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({url: item.url})
    });
    if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
    const doc = await resp.json();
    viewer.innerHTML = `
      <div class="doc-header">
        <div class="doc-title">${h(doc.title||item.title||'Untitled')}</div>
        <div class="doc-url-row">
          <div class="doc-url" title="${h(item.url)}">${h(item.url)}</div>
          <button class="copy-btn" onclick="copyUrl('${h(item.url).replace(/'/g,"\\'")}')">Copy URL</button>
        </div>
      </div>
      <div class="doc-body-wrap">
        <div class="doc-body">${h(doc.content)}</div>
      </div>
    `;
  } catch(e) {
    viewer.querySelector('.doc-body-wrap').innerHTML =
      `<div class="ph"><div class="ph-icon">⚠️</div><div class="ph-text" style="color:var(--err)">${h(e.message)}</div></div>`;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function h(s){ return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;') }

function copyUrl(url) {
  navigator.clipboard.writeText(url).then(() => toast('URL copied!'));
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadConfig();
  document.getElementById('q').addEventListener('keydown', e => {
    if (e.key === 'Enter') doSearch();
  });
  // "/" to focus search box
  document.addEventListener('keydown', e => {
    if (e.key === '/' && document.activeElement !== document.getElementById('q')) {
      e.preventDefault();
      document.getElementById('q').focus();
    }
  });
});
</script>
</body>
</html>"""

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Offline Search UI", docs_url="/docs")


@app.get("/", response_class=HTMLResponse)
def ui():
    return _HTML


@app.get("/api/config")
def api_config():
    return {
        "corpus_name":    CORPUS_NAME,
        "engine_type":    ENGINE_TYPE.upper(),
        "doc_count":      DOC_COUNT,
        "parquet_path":   _corpus["parquet_path"],
        "max_snippet_len": MAX_SNIPPET_LEN,
    }


@app.post("/api/search")
def api_search(req: SearchRequest):
    hits = searcher.search(req.query, topn=req.topn)
    results: List[SearchItem] = []
    for h in hits:
        if not h.text:
            continue
        url = corpus.get_url_from_id(h.docid)
        if not url:
            continue
        parsed  = _parse(h.docid, h.text)
        try:
            summary = _highlight(req.query, parsed["content"])[:MAX_SNIPPET_LEN]
        except Exception:
            summary = parsed["content"][:MAX_SNIPPET_LEN]
        results.append(SearchItem(url=url, title=parsed["title"], summary=summary))
    return {"query": req.query, "count": len(results), "results": results}


@app.post("/api/content", response_model=FetchResponse)
def api_content(req: FetchRequest):
    docid = corpus.get_id_from_url(req.url)
    if not docid:
        raise HTTPException(404, f"URL not in corpus: {req.url}")
    text = corpus.get_text_from_id(docid)
    if not text:
        raise HTTPException(404, f"No content for docid {docid}")
    return FetchResponse(**_parse(docid, text))


# ── Legacy API aliases (compatible with deploy_search_service.py) ─────────────
@app.post("/search")
def search_compat(req: SearchRequest):
    return api_search(req)

@app.post("/get_content")
def get_content_compat(req: FetchRequest):
    return api_content(req)
