import os
import re
import sys
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
import duckdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAX_SNIPPET_LEN = int(os.getenv("MAX_SNIPPET_LEN", "180"))

from backend import Corpus, get_searcher, BaseSearcher


PARQUET_PATH = os.getenv("CORPUS_PARQUET_PATH")
assert PARQUET_PATH
corpus = Corpus(PARQUET_PATH)

searcher: BaseSearcher = get_searcher(corpus=corpus)
logger.info(f"Searcher '{type(searcher).__name__}' initialized successfully.")


class SearchRequest(BaseModel):
    query: str
    topn: int = 10

class SearchResponseItem(BaseModel):
    url: str
    title: str
    summary: str

class FetchRequest(BaseModel):
    url: str

class FetchResponse(BaseModel):
    title: str
    content: str
    

_FM_PATTERN = re.compile(
    r"---\s*"
    r"(?:title:\s*(?P<title>.+?)\s*)"
    r"(?:author:\s*(?P<author>.+?)\s*)?"
    r"(?:date:\s*(?P<date>.+?)\s*)?"
    r"---\s*"
    r"(?P<content>.*?)(?=---|$)",
    re.DOTALL
)

def _parse_content(docid: str, raw_content: str) -> Dict[str, Any]:
    matches = list(_FM_PATTERN.finditer(raw_content))
    if matches:
        m = matches[0]
        title = (m.group("title") or f"Doc {docid}").strip()
        body = (m.group("content") or "").strip()
        # simplified content
        content = body
    else:
        title = f"Doc {docid}"
        content = raw_content.strip()
    return {"title": title, "content": content}


app = FastAPI(
    title="Unified Search Service",
    version="0.3.0",
    description="Search over a configured index and return results.",
)

@app.get("/", tags=["meta"])
def root():
    return {
        "service": "Unified Search Service",
        "active_searcher": type(searcher).__name__,
        "endpoints": ["/search", "/get_content"],
    }

@app.post("/search", response_model=Dict[str, List[SearchResponseItem]], tags=["search"])
def search(request: SearchRequest):
    hits = searcher.search(request.query, topn=request.topn)
    
    results = []
    for h in hits:
        if not h.text:
            logger.warning(f"No text found for docid {h.docid}, skipping.")
            continue

        url = corpus.get_url_from_id(h.docid)
        if not url:
            logger.warning(f"No URL found for docid {h.docid}, skipping.")
            continue
        
        parsed = _parse_content(h.docid, h.text)
        summary = parsed["content"][:MAX_SNIPPET_LEN]

        results.append(
            SearchResponseItem(
                url=url,
                title=parsed["title"],
                summary=summary,
            )
        )
    return {"results": results}

@app.post("/get_content", response_model=FetchResponse, tags=["content"])
def get_content(request: FetchRequest):
    docid = corpus.get_id_from_url(request.url)
    if not docid:
        raise HTTPException(status_code=404, detail=f"URL not found in corpus: {request.url}")

    text = corpus.get_text_from_id(docid)
    if not text:
        raise HTTPException(status_code=404, detail=f"Content not found for docid: {docid} (URL: {request.url})")

    parsed = _parse_content(docid, text)
    
    return FetchResponse(
        title=parsed["title"],
        content=parsed["content"],
    )
