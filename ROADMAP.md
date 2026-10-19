# Roadmap

Goal: an agent that searches papers, news and blogs, adds what it finds to the
RAG store, and answers with citations. It should run in the cloud, not only on
one PC.

## How the finished agent works

```
User question
   │
   ▼
Search stored evidence (Chroma) ◄──────────────┐
   │                                           │
   ▼                                           │
Enough evidence? ── no ──► Search the web      │
   │                       (arXiv, Semantic    │
  yes                       Scholar, Tavily)   │
   │                          │                │
   ▼                          ▼                │
Answer with citations     Ingest new sources ──┘
(verified quotes, URLs)   (fetch, extract,     at most 3 rounds
                           chunk, embed, store)
```

The ingest and search steps reuse the existing pipeline:
`split_documents` → `embed_chunks` → `save_chunks`, then `retrieve`.
Answers are checked by `verify_claims()` in `src/ragsale/agent/graph.py`.

## Phases

Each phase ends with a "done when" check. Finish it before starting the next.

### Phase 0 – Clean up the base

- Run the full test suite and fix anything broken.
- Stop tracking `__pycache__/main.cpython-311.pyc`; ignore `ans_error/`.
- Run the LangGraph agent once with real Ollama and a real PDF.
- Optional: rename the `ragsale` package now, while the code is still small.

Done when: the tests pass and `uv run python -m ragsale.agent "..."` answers
with a citation.

### Phase 1 – Ingest one URL

- Add `ingest_url(url)`: fetch → extract text (trafilatura for web pages, the
  existing loader for PDFs) → metadata (URL, title, date, source type).
- Skip duplicates (hash of the URL or the text).
- Reuse `split_documents`, `embed_chunks` and `save_chunks` unchanged.

Done when: you ingest a blog post, ask about it, and the answer cites its URL.

### Phase 2 – Hosted services

- Choose the embedding model first. Changing it later means re-embedding
  everything already stored.
- Move Chroma from `PersistentClient` (local folder) to Chroma Cloud.
- Use a hosted LLM through `AGENT_MODEL`.
- Keep all settings in environment variables so the same code runs locally and
  in the cloud.

Done when: the app runs with nothing stored on the local PC.

### Phase 3 – Search tools (outside the agent)

- arXiv and Semantic Scholar first (free), then Tavily for web and news.
- Each returns a list of `{url, title, snippet, date}`.
- Test each tool on its own before connecting it to the agent.

Done when: each tool returns clean results for a test query.

### Phase 4 – Connect the tools to LangGraph

- Replace the project-based tools (`search_project`, `compare_projects`) with
  source-based ones: `search_stored`, `search_web`, `ingest_sources`.
- Python enforces the loop: check stored evidence first, search the web only
  when needed, at most 3 rounds (same approach as the current search and
  call limits).

Done when: a question on a new topic triggers search → ingest → cited answer,
and asking it again answers straight from the store.

### Phase 5 – Chat UI and memory

- Show the agent's steps and sources in the React UI.
- Add conversation memory with LangGraph's checkpointer.
- Compare agent and pipeline answers on the same question set (`evaluation/`).

Done when: you can see each step, click its sources, and ask follow-ups.

### Phase 6 – Deploy

- Backend on Render, Railway, Fly.io or Cloud Run; frontend on Vercel.

Done when: it works from a phone.

## Workflow for each step

1. Pick one small step from the current phase.
2. Write a test first. Agent code uses the scripted fake model, as in
   `tests/test_agent_graph.py`.
3. Write the code and run the tests:
   `uv run python -m unittest discover -s tests`
4. Try it by hand with a real model.
5. Commit with a short message and push to `main`.

## Open decisions

- **LLM provider:** Ollama for development; for the cloud, one hosted provider
  with a free tier and tool calling. `AGENT_MODEL` switches between them.
- **Budget:** start on free tiers.
- **Language:** English first. If Thai is wanted later, check that the Phase 2
  embedding model handles Thai before storing a lot of data.
