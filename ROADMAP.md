# Roadmap

Goal: an agent that searches papers, news and blogs, adds what it finds to the
RAG store, and answers with citations, in English and Thai. It should run in
the cloud, not only on one PC.

## Current status

Work happens on two PCs. Pull before starting and push after each update to
this file, so the other PC sees where things stand.

- **Phase:** 0 – Clean up the base (done, apart from the optional rename)
- **In progress:** nothing yet
- **Next step:** decide on the optional `ragsale` rename, then start
  Phase 1 (choose the embedding model)
- **Blockers / notes:** `uv sync` done on the pui117 Linux PC; the other PC
  needs `uv sync` after pulling (packages were removed).

## Progress log

Newest first. One entry per piece of work: date, PC, what changed, what is
left.

- **2026-11-04 (PC: pui117 Linux)** – First real agent run (qwen3:8b, a
  2-page text-only sample PDF). Found and fixed a bug: `submit_answer` sent
  `Claim` as a `$ref`, which qwen3:8b cannot follow, so every answer failed
  validation until the budget ran out. `as_tool` now inlines nested schemas
  (test added; 100 tests pass). After the fix, answers are cited from pages 1
  and 2. Found a known issue for Phase 4: a made-up claim passed the checks
  (see Phase 4).
- **2026-11-04 (PC: pui117 Linux)** – Removed unused files: the scratch
  scripts `TestHuggingface.py` and `DB_Data_viewer.py`, and the old Hugging
  Face dataset ingest (`main.py`, `ragsale/ingest.py`, `load_documents()`,
  the `ragsale` console script). Dropped unused packages: `datasets`,
  `streamlit`, `openai`, `langchain-community`, `langchain-chroma`,
  `unstructured[pdf]`. README updated. Tests still 99 pass, 1 skipped.
- **2026-11-04 (PC: pui117 Linux)** – Ran the test suite: 99 pass, 1
  skipped (private reference files not present), nothing to fix. Next: one
  real agent run with Ollama and a PDF.
- **2026-11-03 (PC: pui117 Linux)** – Rewrote the roadmap after a review of
  the code: embedding choice and config moved to Phase 1, cloud services
  moved to Phase 6, added checklists, security and evaluation steps, and the
  two-PC workflow. No code changed yet.
- **2026-11-03 (PC: pui117 Linux)** – Cloned the repo fresh. Added this status
  section, the progress log and `CLAUDE.md` so both PCs keep the roadmap up to
  date. Dependencies not installed yet (`uv sync`).

## Where we start

Already working (inherited from RagSale):

- PDF loading with OCR, chunking, F2LLM embeddings on CPU, local Chroma,
  hybrid dense + keyword search, and cited answers.
- A LangGraph agent over uploaded PDFs (`src/ragsale/agent/`), with Python
  limits (4 searches, 7 model calls, 180 seconds) and `verify_claims()`, which
  drops any claim whose quote is not in its cited excerpt.
- FastAPI backend, React frontend, unit tests with a scripted fake model.

Not cloud-ready yet: `chroma_db/`, `data/projects.sqlite3` and `uploads/` are
local folders, so each PC has its own data. The embedding model needs torch
and more RAM than free hosting tiers give.

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
Tick the boxes as steps are done.

```
0 Clean up ─► 1 Embedding + config ─► 2 Ingest one URL ─► 3 Search tools
                                                               │
7 Deploy ◄── 6 Cloud services ◄── 5 Chat UI + memory ◄── 4 Agent + web search
```

### Phase 0 – Clean up the base

- [ ] `uv sync` on each PC (downloads torch and OCR packages, several GB).
      Done on the pui117 Linux PC.
- [x] Run the full test suite and fix anything broken. 99 tests pass, 1
      skipped (needs private reference files in `evaluation/`).
- [x] Run the LangGraph agent once with real Ollama and a real PDF. Needed a
      fix: nested tool schemas are now inlined for qwen3:8b.
- [ ] Optional: rename the `ragsale` package now, while the code is small.
- [x] `__pycache__` files are not tracked; `ans_error/` is ignored.
- [x] Remove unused files, the old dataset ingest and unused packages.

Done when: the tests pass and `uv run python -m ragsale.agent "..."` answers
with a citation.

### Phase 1 – Embedding model and config

Decide this before storing much data: changing the embedding model later means
re-embedding everything.

- [ ] Choose the embedding model. Requirements: handles English and Thai, fast
      enough to ingest a page within a question's time budget, and runs where
      the backend will be hosted. Compare the current F2LLM (local CPU) with a
      hosted embedding API on the Thai and English smoke cases in
      `evaluation/cases/`.
- [ ] Move settings to environment variables: Chroma collection name
      (now hard-coded in `vector_store.py`), embedding model, `AGENT_MODEL`,
      data paths. Add a `.env.example`.
- [ ] One collection per embedding model; never mix models in one collection.

Done when: switching the embedding model or collection is a `.env` change, and
the smoke cases pass with the chosen model.

### Phase 2 – Ingest one URL

- [ ] Add `trafilatura` and write `ingest_url(url)`: fetch → extract text
      (trafilatura for web pages; for PDFs download to a temp file, then the
      existing `load_pdf_documents`) → metadata.
- [ ] Metadata on every chunk: `document_id` (hash of the normalised URL),
      `filename` (title or URL, so `list_documents` shows it), `url`, `title`,
      `published_date`, `source_type`. Chroma accepts only strings, numbers and
      bools, so join lists such as authors into one string.
- [ ] Skip duplicates by checking `document_id` before fetching. Keep fetch
      dates out of chunk metadata, or re-ingesting creates duplicate chunk IDs.
- [ ] Fetch safely: only http/https, block private and internal IP addresses,
      cap download size and time.
- [ ] Start the web evaluation set: 10 questions about pages you ingest, with
      the expected source URL for each.

Done when: you ingest a blog post, ask about it, and the answer cites its URL.

### Phase 3 – Search tools (outside the agent)

- [ ] arXiv and Semantic Scholar first (free), then Tavily for web and news.
- [ ] Each returns a list of `{url, title, snippet, date, source_type,
      pdf_url}`.
- [ ] Respect rate limits (arXiv about 1 request per 3 s; Semantic Scholar
      without a key is shared; Tavily free tier about 1,000 credits a month).
- [ ] Cache results, and test with recorded responses so tests use no quota.

Done when: each tool returns clean results for a test query, and its tests run
offline.

### Phase 4 – Agent with web search

- [ ] New tools: `search_stored`, `search_web`, `ingest_sources`.
- [ ] Keep project and document scoping as a filter on `search_stored` (the
      frontend picker and tests use it) instead of deleting it.
- [ ] Define "enough evidence" in Python, for example at least N excerpts
      under a distance threshold, and newer than X days for news.
- [ ] Python enforces the loop: stored evidence first, web only when needed,
      at most 3 rounds.
- [ ] Fit ingestion in the time budget: during a question, ingest only the
      abstract or best chunks; ingest full text in the background.
- [ ] Check that each claim follows from its quote, not only that the quote
      exists. Seen 2026-11-04: asked "Which PLC brand controls the line?",
      qwen3:8b answered "The PLC brand is Cognex" with a real quote about
      Cognex VisionPro software, and `verify_claims()` accepted it.
- [ ] Fetched text is untrusted: keep the prompt rule that excerpts are data,
      never instructions.

Done when: a question on a new topic triggers search → ingest → cited answer,
and asking it again answers straight from the store.

### Phase 5 – Chat UI, memory and evaluation

- [ ] Show the agent's steps and sources in the React UI.
- [ ] Conversation memory with LangGraph's checkpointer (SQLite locally;
      Postgres or another persistent store in the cloud).
- [ ] Compare agent and pipeline answers on the evaluation set (`evaluation/`).

Done when: you can see each step, click its sources, and ask follow-ups.

### Phase 6 – Cloud services

- [ ] Move Chroma from `PersistentClient` to Chroma Cloud.
- [ ] Move project assignments (`data/projects.sqlite3`) and uploads
      (`uploads/`) to hosted storage.
- [ ] Hosted LLM through `AGENT_MODEL`; hosted embeddings if chosen in
      Phase 1.
- [ ] Check hybrid search speed: scoped searches load every scoped chunk on
      each query, which is slow over the network.

Done when: the app runs with nothing stored on the local PC, and both PCs see
the same data.

### Phase 7 – Deploy

- [ ] Backend on Render, Railway, Fly.io or Cloud Run; frontend on Vercel.
- [ ] Login or rate limiting before the URL is public, so nobody else uses up
      the LLM and search credits.
- [ ] Secrets only in the host's environment settings, never in git.

Done when: it works from a phone, and only you can use it.

## Workflow for each step

1. `git pull`, then read **Current status** above.
2. Pick one small step from the current phase.
3. Write a test first. Agent code uses the scripted fake model, as in
   `tests/test_agent_graph.py`. Tests create their own data; don't rely on
   what is in `chroma_db/`, which differs between PCs until Phase 6.
4. Write the code and run the tests:
   `uv run python -m unittest discover -s tests`
5. Try it by hand with a real model.
6. Update **Current status**, add a **Progress log** entry, tick the step.
7. Commit the code and this file together with a short message, and push to
   `main`.

## Open decisions

- **Embedding model:** decided in Phase 1. Must handle Thai.
- **LLM provider:** Ollama for development; for the cloud, one hosted provider
  with a free tier and tool calling. `AGENT_MODEL` switches between them.
- **Budget:** start on free tiers.
- **Language:** English and Thai. The base already supports Thai (including a
  keyword fallback for Thai queries), so keep it working in every phase.
