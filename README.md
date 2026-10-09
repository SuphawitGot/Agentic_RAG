# RagSale

A local RAG web app for English and Thai PDF reports. Upload PDFs, then ask
questions in the chat: the backend retrieves matching excerpts and a local Qwen
model answers with numbered citations. Everything runs on your machine: FastAPI
backend, React frontend, Chroma vector store, CPU embeddings, and Ollama for the
answer model and OCR.

For project history, design decisions, and notes for moving to another PC, read
[CODEX_HANDOFF.md](CODEX_HANDOFF.md).

## How it works

1. **Extract** – `pypdf` and pdfplumber read native PDF text with layout-aware
   table handling. Pages with images or without text are read by
   [Typhoon OCR 1.5](https://huggingface.co/scb10x/typhoon-ocr1.5-2b) in Ollama.
2. **Chunk and embed** – text is split into chunks and embedded on CPU with
   `codefuse-ai/F2LLM-v2-0.6B` (1,024-dimension normalized vectors).
3. **Store** – chunks go into the Chroma collection `sales_f2llm_v2_06b` in `chroma_db/`.
4. **Answer** – each question is matched by hybrid vector + keyword retrieval,
   then `qwen3:8b` in Ollama writes an answer that cites the excerpts it used.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Node.js 22.12+ (or 20.19+) and pnpm (`npm install -g pnpm` if missing)
- [Ollama](https://ollama.com/download) with two models:
  - `qwen3:8b` – answers (about 5 GB)
  - `scb10x/typhoon-ocr1.5-3b` – OCR (3.2 GB)
- An NVIDIA GPU is strongly recommended for Ollama. Embeddings always run on CPU.
  The embedding model downloads from Hugging Face on first use.

## Quick start

Run from the project root unless noted.

1. Install Python dependencies:

   ```bash
   uv sync
   ```

2. Pull the Ollama models (Ollama must be installed and running):

   ```bash
   ollama pull qwen3:8b
   ollama pull scb10x/typhoon-ocr1.5-3b
   ```

3. Start the backend:

   ```bash
   uv run uvicorn ragsale.api.main:app --reload
   ```

4. In a second terminal, start the frontend:

   ```bash
   cd frontend
   pnpm install
   pnpm dev
   ```

5. Open http://127.0.0.1:5173. Vite forwards `/api` to the backend on port 8000.
   Interactive API docs are at http://127.0.0.1:8000/docs.

See [frontend/README.md](frontend/README.md) for frontend structure and production builds.

## Configuration

Settings are read from environment variables when FastAPI starts. Restart the
backend after changing them.

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server for answers and OCR |
| `QWEN_MODEL` | `qwen3:8b` | Answer and project-clarification model |
| `TYPHOON_OCR_MODEL` | `scb10x/typhoon-ocr1.5-3b` | OCR model |

PowerShell:

```powershell
$env:QWEN_MODEL = "qwen3:8b"
uv run uvicorn ragsale.api.main:app --reload
```

bash:

```bash
export QWEN_MODEL=qwen3:8b
uv run uvicorn ragsale.api.main:app --reload
```

Any model you configure must already be pulled in Ollama.

## Project structure

```text
frontend/                     # React + Vite app; see frontend/README.md
src/ragsale/
  api/
    main.py                   # FastAPI app; all routes under /api
    health.py                 # GET  /api/health
    documents.py              # GET  /api/documents, POST /api/documents/upload
    search.py                 # POST /api/search (retrieval only)
    chat.py                   # POST /api/chat (answers and clarification)
    scope.py                  # POST /api/chat/scope (older filename-only preview)
    projects.py               # Project listing and assignment
  rag/
    loader.py                 # PDF pages / dataset rows -> Documents
    layout.py                 # Layout-aware native text and tables
    ocr.py                    # Typhoon OCR via Ollama
    pdf_ingestion.py          # Upload pipeline: extract -> chunk -> embed -> store
    chunking.py               # Documents -> chunks
    embedding.py              # F2LLM CPU embeddings
    vector_store.py           # Persistent Chroma storage
    retrieval.py              # Vector search
    keyword_search.py         # Keyword ranking for hybrid retrieval
    answering.py              # Grounded Qwen answers with citations
    project_chat.py           # Project scope detection and clarification
    project_names.py          # Project name matching in questions
    projects.py               # Project registry (data/projects.sqlite3)
    clarification.py          # Filename-based document-scope rules
  ingest.py                   # Demo dataset ingestion workflow
evaluation/                   # Extraction, chunking and embedding evaluations
tests/                        # Backend unit tests
main.py                       # Runs ingest.py
DB_Data_viewer.py             # Inspect the Chroma database
```

These are ignored by Git and stay on the local machine: `chroma_db/` (vectors),
`uploads/` (uploaded PDFs), `data/` (project assignments) and `output/`
(evaluation results). Include them in your backups.

## Uploading PDFs

Select one or more PDFs in the upload form and click **Upload PDFs**. Files are
processed one at a time, each with its own result; a failed file does not stop
the rest. Keep the page open until the batch finishes, because the queue does not
resume after a refresh. To retry, select only the failed files.

- Each PDF must be 50 MiB or smaller, 200 pages or fewer, non-empty, unencrypted,
  and start with a PDF header.
- The request stays open until the PDF is fully indexed; long scanned reports
  can take many minutes. Background jobs are planned.
- A successful upload returns `status: ready` with `text_pages`, `ocr_pages` and
  `chunks`. A failed upload removes its saved file and any partly stored chunks.
- Every upload gets a new document ID, so uploading the same file twice creates
  a duplicate document. Existing documents are not re-processed when the
  extraction code changes; upload them again to use new extraction.
- Behind a reverse proxy, allow at least 50 MB request bodies (nginx:
  `client_max_body_size 50m;`) and a long read timeout.

### Native text and layout

`rag/layout.py` uses pdfplumber coordinates and ruled-table cells. Two-column
table rows become `Label: Value`; wider tables keep cell separators. Wrapped
labels are joined without rewriting numbers, `label = value` lines become
`label: value`, and different-colored diagram labels stay separate from prose.
Short aligned caption columns keep each multi-line caption together.

Pages with Thai or rotated text keep the original pypdf text, because spatial
sorting can damage Thai combining marks; only exact-matching Latin table rows
are repaired there. Page metadata records `layout_method`, `table_count` and
`repaired_rows`.

### OCR (Typhoon OCR 1.5)

Pages with embedded images, and pages with no native text, are rendered with
`pypdfium2` at 1,800 px on the longest side (the size the model was trained on)
and sent to Typhoon OCR in Ollama with the model card's required prompt. Typhoon
returns Markdown with HTML tables, LaTeX equations, and a Thai description of
each figure, stored as a `[Figure] ...` line. Page numbers are dropped.

- Native text is kept. OCR lines already present in the native text are skipped,
  ignoring case, whitespace and Markdown/HTML/LaTeX markup. This is best-effort:
  text that Typhoon words differently can still appear twice.
- Pages run one at a time, with a 10-minute timeout and at most 8,192 output
  tokens per page. Expect seconds to a minute per page on a GPU, and minutes on CPU.
- Typhoon can hallucinate, especially in figure descriptions. Check important
  numbers against the PDF.
- Vector-only diagrams on pages that have native text are not sent to OCR.
- Each chunk records `extraction_method` (`native_text`, `ocr` or
  `native_text+ocr`), `ocr_attempted` and `ocr_version` (`typhoon-ocr1.5-page-v1`
  or `none`).
- If Ollama is not running or the model is not pulled, the upload fails with a
  503 and setup instructions. Unreadable PDFs and OCR timeouts return 422.

Preview a PDF's extraction without embedding it or changing Chroma:

```bash
uv run python -m ragsale.rag.ocr "path/to/report.pdf" --output output/extraction/report.json
```

## Asking questions

Each uploaded PDF is one project by default, named after its file.

- **Named projects** – a full project name or filename in the question (for
  example "Summarize Scope of Work of copper pipe test") selects that project.
  Matching ignores case, spacing, underscores and hyphens. Distinctive shortened
  names also work ("washer inspection" matches "IAI_Washer Inspection") when
  they contain at least two adjacent title words, including a distinctive word.
  Document markers may be omitted ("test ยา" matches "Test Report ยา").
  Full names take priority over shortened ones.
- **Clarification** – when no project is named, the backend retrieves candidate
  excerpts from each project and asks Qwen whether several fit. If so, the chat
  shows project buttons and **Compare these projects**; **Ask a different
  question** cancels. For example, if Camera A appears in two reports, "What is
  Camera A's accuracy?" asks which project. Up to 12 projects are assessed
  before a manual choice is required. Duplicate names always ask.
- **Retrieval** – scoped retrieval combines vector ranking with literal keyword
  ranking, and reserves up to half the result budget for the strongest keyword
  matches. There are no synonym or translation lists, and Thai word
  segmentation is limited. Answers use `top_k` chunks per project, capped at 18
  chunks in total for comparisons.
- **Answers** – Qwen receives numbered excerpts as untrusted context and must
  return a structured answer with valid citation numbers, or say the evidence is
  insufficient. This checks that references exist, not that every claim is
  supported, so verify answers against the shown excerpts. Requests time out
  after 180 seconds; the first question may be slow while Ollama loads the model.
- Each question is independent. History is shown in the browser but not sent
  to the model, and refreshing the page clears it.

## API

All routes are under `/api`; see http://127.0.0.1:8000/docs for schemas.

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Returns `{"status": "ok"}` |
| GET | `/documents` | Indexed documents |
| POST | `/documents/upload` | Upload and index one PDF (multipart field `file`) |
| POST | `/search` | Retrieval only: `question` (1–2000 chars), `top_k` (1–10); returns excerpts with cosine distance |
| POST | `/chat` | `question`, optional `selected_project_ids` and `top_k` (default 3); returns `type: "answer"` or `type: "clarification"` with `project_options` |
| POST | `/chat/scope` | Older filename-only scope preview |
| GET | `/projects` | Projects and their document IDs |
| PUT | `/documents/{document_id}/project` | Group PDFs: `{"project_name": "Project 1"}`; stored in `data/projects.sqlite3` |

To continue after a clarification, resend the original question with the chosen
`selected_project_ids`.

## Command-line tools

```bash
# Search stored chunks (retrieval only, no Qwen)
uv run python -m ragsale.rag.retrieval "What training materials are available?" --top-k 3

# Inspect the Chroma database
uv run python DB_Data_viewer.py

# Ingest the demo Hugging Face dataset (mtybilly/apex-r1-real-world-documents)
uv run python main.py
```

Evaluation scripts live in `evaluation/`; see [evaluation/README.md](evaluation/README.md).
For example, to score extraction on a folder of PDFs without touching the database:

```bash
uv run python -m evaluation.evaluate_documents --pdf-dir "path/to/pdfs" --output-dir output/extraction/layout-improved
```

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

Frontend tests use Node's built-in test runner. From `frontend/`:

```bash
node --test tests/documents.test.js
```

On Windows, a few backend tests that create temporary Chroma/SQLite files or
assume `/tmp` paths currently fail because of file locking and path handling.

## Limitations and next steps

- Uploads run synchronously; a background job queue would remove long-open
  requests and timeouts.
- Re-uploading creates duplicates; there is no re-index or replace step yet.
- No authentication or deployment configuration yet. Keep models, Chroma and
  credentials on the backend, and serve the frontend from the same origin or
  configure allowed origins explicitly.
- Retrieval and answer quality still need evaluation on representative
  questions; nearest matches are not guaranteed to contain the answer.
- Thai table reconstruction, OCR confidence filtering and header removal are
  not implemented.
