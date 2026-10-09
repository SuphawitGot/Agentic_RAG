# RagSale — current project handoff

Last updated: 2026-10-02.

## Repository connection — 2026-10-02

This working project now uses `https://github.com/SuphawitGot/Rag_SummarizeSheet.git`
as its only remote (`origin`), for both fetch and push. Current source files were
copied by cloning RagSale_management earlier; they are retained here. The unrelated
histories were joined using an `ours` merge of the target's initial commit
`f23be26`, preserving the current application files and both histories. The old
beginner project remains accessible in that initial commit. This permits a normal
push to Rag_SummarizeSheet without overwriting its history. RagSale_management is
not a configured remote and is unaffected by this integration.

## Windows dependency setup — 2026-10-02

Installed the existing lockfiles in `D:\GithubRepo\PROJECT\RagLLM` using
`uv sync --locked` and `pnpm install --frozen-lockfile` in `frontend/`.
Python 3.11.9: 197 installed packages including the local project. Frontend:
19 installed packages including transitive dependencies. No lockfile changes.
Validation: `uv pip check`, backend/core RAG imports, FastAPI `/api/health`
through TestClient, and `pnpm build` passed. The frontend build required execution
outside the tool sandbox because process spawning was blocked inside it.
No model weights were downloaded or document ingestion run. Ollama and Tesseract
were not found on the current PATH; Poppler was found in the Codex bundled runtime.
Next setup step: configure these external runtimes and required models/language
packs for chat and OCR, then verify an end-to-end PDF upload and answer.

## Maintenance agreement

The user explicitly requested that this file be kept current. After meaningful
implementation changes, test runs, reversals, or project decisions, update this
file in the same task. Record what changed, validation results and limitations,
and the next unfinished step. Distinguish implemented behavior from proposals.
Do not record conceptual examples as real IDs, files, or implemented features.
Verify current code/database before relying on historical counts. This is a
working summary, not a transcript; keep it concise and remove stale instructions.

## Purpose and user preferences

A local React + FastAPI RAG website for sales/industrial inspection PDFs containing
English, Thai, tables, screenshots and photos. The user is learning the pipeline
and wants plain-language, step-by-step explanations. The project is intended for
this computer. Docker setup was requested, then explicitly canceled and reverted;
do not reintroduce it unless asked. Preserve unrelated edits, PDFs and database
records. Ask before destructive cleanup unless already authorized.

## Implemented pipeline and code map

Upload PDF → extract native text/layout + OCR → page Documents → chunks → CPU
MiniLM embeddings → local ChromaDB. Question → same-model embedding → top-K
cosine search → local Ollama Qwen → answer with numbered sources.

- `frontend/`: React/Vite; PDF upload, scrollable chat, sources and theme switch.
- `src/ragsale/api/documents.py`: upload endpoint; one PDF at a time, 50 MiB and
  200-page limits. Original filename is retained; uploaded file gets a UUID.
- `src/ragsale/rag/pdf_ingestion.py`: orchestration and per-document chunk indices.
- `rag/loader.py`, `layout.py`, `ocr.py`: extraction; paths here are relative to
  `src/ragsale/`. pypdf is retained alongside pdfplumber, not replaced entirely.
- `rag/chunking.py`: RecursiveCharacterTextSplitter, 1000 characters/200 overlap.
- `rag/embedding.py`: `sentence-transformers/all-MiniLM-L6-v2`, CPU, normalized,
  384 dimensions. Embeds chunk text, not its metadata.
- `rag/vector_store.py`: root `chroma_db/`, collection `sales_minilm_l6_v2`, cosine
  metric. SHA-256 of text + metadata becomes the record ID; batched upsert.
- `rag/retrieval.py`: searches all documents; default top_k=3. Returned distance
  is lower-is-closer, not a probability or calibrated confidence score.
- `rag/answering.py`: Ollama default `qwen3:8b`, overridden by QWEN_MODEL;
  OLLAMA_BASE_URL defaults to localhost:11434. Structured output, thinking off,
  180-second timeout. Citation-number validation does not prove factual support.
- `api/search.py`: raw `/api/search`; `api/chat.py`: `/api/chat` with generation.

Filename is captured at upload, attached to extracted pages, inherited by chunks
and stored in metadata. Metadata may include document_id, source, filename, page
(1-based), chunk_index (0-based), extraction_method, ocr_attempted, layout_method,
table_count and repaired_rows. Chunk ID, text and vector are separate fields.
Test exports use document_id="evaluation"; this is not a real uploaded document ID.
Repeated uploads get new IDs and can duplicate data. Extractor edits do not update
already indexed chunks automatically.

## Extraction status

pdfplumber handles positioned English text and ruled tables, joins wrapped labels,
and keeps labels with values. Thai/rotated pages preserve pypdf text with selective
exact-match Latin table repairs. Local Typhoon OCR 1.5 (Ollama) supplements native
text and adds Thai figure descriptions; scanned tables, handwriting, noisy
OCR and visual defect interpretation remain limitations.

Three original PDFs were at `/home/pui117/Desktop/Test raw data/Test_Rag/`:
IAI_Washer Inspection.pdf, Imco foodpack V1a 2.pdf,
SHARP AUTO LABEL INSPECTION-REVISE1.pdf.

Nine reference pages/34 fields improved from 26/34 to 34/34 after layout changes
using unchanged scoring. These are development-set field-association checks,
not whole-document CER, held-out accuracy or end-to-end RAG accuracy.
References and commands are documented in `evaluation/README.md`.

## Vector checks completed — 2026-09-22

Reports: `output/vectors/report.md` and `report.json`.

- Existing database snapshot: 166 records across six documents. All vectors were
  finite, nonzero, normalized and 384-dimensional; IDs matched text/metadata hashes.
- Re-embedding stored text matched all 166 stored vectors within atol=1e-5,
  rtol=1e-4; largest absolute difference was approximately 1.3e-7.
- Generated embeddings for 105 improved chunks from the three test PDFs: all
  passed count/dimension/finite/nonzero/normalization checks.
- Only 77/105 improved text+filename+page combinations matched stored records;
  28 did not. This is not a comparison of identical document sets: the 166
  database records cover six documents, the improved exports cover three.
- Improved embeddings were generated IN MEMORY ONLY, not stored. Chroma unchanged.
- The vector check was a one-off script; no reusable vector-evaluation CLI was
  added at that time. Improved-batch storage and reopen verification were later
  completed on 2026-09-24, as recorded below.

The intended sequence is checked chunks → generate embeddings → validate vectors
→ store → read back by ID and compare text/metadata/vectors → reopen database and
verify persistence → retrieval evaluation. Unit-length vectors are expected
because normalization is enabled; cosine search can work without prenormalization.
Numerical validity does not establish semantic retrieval accuracy.

## Retrieval and project clarification — implemented 2026-09-25

User confirmed: start with ONE PDF PER PROJECT. Default project IDs are
`document:<document_id>`; display names are PDF filename stems. No existing
Chroma records or embeddings were rewritten. `rag/projects.py` supplies this
mapping and optionally persists explicit multi-PDF assignments in local
`data/projects.sqlite3` (ignored). GET /api/projects lists projects;
PUT /api/documents/{document_id}/project assigns a project_name if later needed.
Project metadata is joined onto retrieval results, not stored into old chunks.

POST /api/chat now calls rag/project_chat.py. With no explicit selection it
retrieves two excerpts per project (one query embedding reused), then asks local
Qwen for structured relevance/scope decisions. IDs are validated; multiple
relevant projects without explicit scope/comparison produce project choices.
Explicit named/task scope can resolve directly; clear comparisons use multiple
projects. Missing evidence produces an insufficient-evidence response. Failed
scope calls offer real project choices rather than silently searching all data.
Above 12 projects, ask for manual scope before model assessment. Final retrieval
uses document_id filters and a per-project budget capped at 18 total chunks.
Selected projects are revalidated on every request. /api/search remains unscoped.
/api/chat/scope remains the earlier filename-rule preview, NOT the active
retrieval-based project classifier.

React displays project buttons, Compare these projects, and Ask a different
question. It keeps the original pending question and resends it with selected
project IDs. Retries preserve this scope. State is in the current browser page;
reload loses it, no cross-question conversational memory or typed follow-up
resolution is implemented. Users choose buttons while clarification is pending.
The active chat does not depend on the old filename-only decide_scope rules
except recognizing standalone greetings.

Known limits: candidate retrieval can miss relevant evidence; Qwen scope
classification is not guaranteed correct. No calibrated confidence threshold.
Names use PDF filenames, not automatic project/task extraction. Two candidate
chunks per project may miss an entity/metric. Synthetic tests are not production
retrieval-accuracy measurements. The historical Washer wrong-answer incident
still needs representative retrieval/answer evaluation, even with new filters.

Next: build a labeled project-ambiguity/retrieval evaluation set from actual PDFs;
measure missed and unnecessary clarification, correct project choice, evidence
recall and grounded answers. Older entries below describe historical stages.

## UI and FigJam updates

Chat panel made taller: desktop 110dvh, min720px/max1600px; mobile 100dvh/min560px.
Change is in `frontend/src/styles.css`; frontend production build passed.

Board: https://www.figma.com/board/KlSZjBbe6QEFyZdoI7YqcS/Rag_Sale-Workflow
User prefers TOP-TO-BOTTOM diagrams.
Vector/storage testing diagram added and rearranged vertically (starting node
77:116), including checks, retry branches, storage readback and retrieval testing.
Question-answering section 52:320 now includes scope decision 98:199, clarification
question, user reply/recheck loop and confirmation before embedding. New behavior
is labeled proposed/not implemented. Existing unrelated board content preserved.

## Local operation and checks

```bash
uv sync
uv run uvicorn ragsale.api.main:app --reload
# Separate terminal:
cd frontend
pnpm dev
```

Ollama must be running with qwen3:8b available; check QWEN_MODEL overrides. Frontend
uses localhost:5173 and proxies /api to localhost:8000. OCR needs Ollama with
scb10x/typhoon-ocr1.5-3b pulled (TYPHOON_OCR_MODEL overrides). Do not copy machine-specific
Codex runtime PATH instructions as general installation guidance.

```bash
uv run python -m evaluation.evaluate_chunking
uv run python -m unittest discover -s tests -p test_chunking_evaluation.py -v
uv run python -m evaluation.evaluate_documents --pdf-dir "/home/pui117/Desktop/Test raw data/Test_Rag"
```

Chunking evaluation reads existing exports; regenerate exports after changing the
splitter. Extraction evaluation does not embed or write Chroma. Consult each
module/README for arguments and required files. The historical main.py ingestion
path had an empty document list; do not assume it indexes the demo HF dataset.

## Reversals, local assets, and Git

Docker files and documentation were removed at the user's request. The Docker
build was stopped and its RagSale frontend image removed. Download/build caches
may remain; no RagSale containers were left running. Host Docker itself existed
before this work and was not uninstalled. Local RAG data was untouched.

Uploads, ChromaDB, model caches, .venv, .tools and extraction previews are local
assets, not something a Git clone automatically transfers. Some chunk/vector
reports may be tracked or untracked; inspect git status before committing, and do
not indiscriminately include document contents. Never claim changes were pushed
without verifying the push result. Preserve user files such as ans_error/.

## Chunking test detail (retained)

### 2026-09-22: repeatable chunking test added

`evaluation/evaluate_chunking.py` now checks the existing improved exports against
saved extraction and reference fields. Run `uv run python -m evaluation.evaluate_chunking`.
It does not regenerate exports or index data. Read evaluation/README.md for
arguments, required local inputs, scope and exit codes.

All 105 chunks passed text coverage/metadata/size/index checks; 34/34 existing
fields and 11/11 newly selected context groups passed. Nine split-paragraph
warnings remain: Washer page 2 (NDA), Imco pages 25/27/28/50 (OCR), SHARP pages
17–20 (OCR). Warnings need review, not automatic chunk-size increases. Context
cases are development checks sourced from extraction, not independent PDF truth
or proof of image understanding. Retrieval and Chroma contents are unchanged.

Reports: `output/chunks/chunking-test/report.md` and `.json` (local outputs).
New evaluator regression tests: `tests/test_chunking_evaluation.py`, nine passing
cases including repeated-text coverage, incorrect metadata, split label/value,
conflicting values and lost heading context.

## Current storage status — completed 2026-09-24

User authorized completing the checked-chunk workflow through storage in the
existing collection. Added `evaluation/store_checked_chunks.py`, which validates
exports against extraction/field/context checks, generates and checks embeddings,
backs up the collection logically, preserves existing document IDs, upserts the
improved records, verifies readback, and then deletes obsolete IDs for those
three reports only. It attempts restoration of affected records on failure.
This is a maintenance command: stop concurrent uploads/writes before rerunning;
it is not a transaction or a multi-writer migration system.

Fresh inventory had 178 records across seven documents, including a newer
Logo & barcode_SBD.pdf upload. Stored all 105 improved chunks for Washer/Imco/SHARP,
removed 105 obsolete IDs (metadata changed even where text was unchanged), and
preserved all 73 unrelated records. Final collection count: 178. Exact ID sets,
text and metadata, plus vectors within numerical tolerance, passed both immediate
readback and verification in a separate Python process after reopening Chroma.
This tested reopening the database, not restarting the running web server.

Report and backup: `output/extraction/storage/20260924T052517Z/report.md`,
`report.json`, `backup.json`, `expected.json` (ignored local artifacts).
The 2026-09-22 vector report remains a historical snapshot, not current state.
Next task: retrieval baseline with clear questions and known evidence. No need
to create another collection or repeat the already completed storage work.

### FigJam wording clarification — 2026-09-24

Updated existing Embedding Test and ChromaDB Checking nodes to distinguish RAM
from persistent storage: generate once in RAM → validate → keep expected paired
records → write checked records to Chroma → read back and compare → reopen DB
and repeat verification → retrieval tests. Comparison does not require another
embedding generation. Storage section 79:203 renamed to clarify its purpose.

### FigJam failure recovery guidance — 2026-09-24

Added two visually checked sections to the existing FigJam board:
- Readback failure cases (107:221): missing IDs, text/metadata mismatch,
  vector mismatch, read errors and unexpected extra records, with recovery actions.
- Persistence failure cases (107:318): missing database/collection, missing or
  changed records after reopening, open errors and missing expected reference.

Existing failure nodes now point readers to these sections. Each section includes
retry/check-again steps and pass versus stop/report outcomes. These are planned
recovery guidance, not newly implemented automatic repairs. Persistence comparison
uses a saved expected-record reference because original process RAM is gone.
No application or database changes were made for this diagram request.

### FigJam vector failure guidance — 2026-09-24

Added Vector check failure cases (108:333), alongside readback/persistence guides.
Seven cases cover embedding errors, count, dimensions, NaN/infinity, zero vectors,
normalization and ID/text/metadata pairing. Each gives a corrective action;
rerun all vector checks before storage, and rerun chunk checks if text changes.
Existing embedding failure node 77:140 points to this section. Guidance is planned
recovery documentation, not new automatic repair code. No database changes.

### Document choices endpoint — 2026-09-24

Implemented GET /api/documents in api/documents.py, returning a JSON array of
{document_id, filename}. rag/vector_store.py:list_documents reads only metadata,
deduplicates by document_id, and sorts by filename then ID. Same-named uploads
with different IDs remain separate. Legacy records lacking usable IDs/filenames
are excluded; conflicting filenames for one ID fail rather than invent a choice.
The API runs the read in a worker thread and returns a generic 503 on errors.
Three focused tests passed (grouping/legacy, empty/conflicts, API/error handling).
Live metadata read returned seven indexed documents. No embeddings regenerated
or stored records modified. Frontend choices, clarification state and retrieval
filtering are still NOT implemented; those are the next steps.

### Clarification decision step — 2026-09-24

Added rag/clarification.py:decide_scope and POST /api/chat/scope (api/scope.py).
This is a standalone decision endpoint; POST /api/chat and React are not wired
into it yet. It returns status ready/clarification/greeting/no_documents, reason,
original question, resolved document_ids, clarification_question and real options.
Selected IDs are validated against the current document list; search_all and
selected IDs are mutually exclusive. Filename/title token rules resolve named
documents, conservative ambiguous cases ask for scope, named comparisons can
resolve multiple documents, and explicit all-document requests are supported.
No Qwen call, embeddings, retrieval or database writes occur in this step.
Rules are an English-oriented baseline, not semantic ambiguity detection or a
confidence score. It can over-ask for general concept questions and comparisons
within one report. A sole available document is assumed when none is named.
Ready means scope resolved, not that evidence exists. Missing evidence must be
handled later by retrieval/answering rather than another scope question.
Next: wire scope decisions into chat, apply resolved IDs as retrieval filters,
preserve pending questions, and display/respond to clarification choices in React.
Validation: six clarification tests plus the existing chat API test passed.
Live metadata checks confirmed ambiguous results ask, IAI Washer resolves to its
actual ID, Washer/Imco comparison resolves both, and a greeting bypasses scope.

### Project clarification validation — 2026-09-25

57 backend tests passed, including 10 new project-chat/registry/model-contract/
API/filtering tests. Real temporary Chroma tests checked scope isolation and
balanced candidate counts. Production frontend build passed. Browser verified
project buttons, preserved original question, single selection and compare-all
using a temporary synthetic API fixture (not real answer accuracy).
Local Qwen synthetic Camera A tests: ambiguous car/person accuracy -> clarify;
person-detection scope -> answer that project; compare both -> answer both after
fixing unnecessary clarification using explicit comparison_scope_clear.
Live cached CPU MiniLM retrieval over the seven existing PDFs returned two
candidate chunks per project; every chunk's document_id matched its project.
No existing document records were modified. Temporary UI test server stopped.

### Next clarification improvement — discussed 2026-09-25, not implemented

User highlighted that listing 100–200 document buttons is not usable. Current
fallback (including catalogs above 12 projects) still exposes all project choices.
Next work: resolve clear names such as "Summarize copper pipe test" before model
scope assessment; retrieve a bounded shortlist across projects instead of querying
every project; ask discriminating evidence-based questions; support typed replies.
Do not claim this scalable workflow is implemented. The screenshot's generic
scope failure message is an error fallback, not proof the question is ambiguous.

Local startup diagnosis: Docker container open-terminal occupies host port 8000.
No container was stopped or project port changed during that diagnosis.

### User-requested document reset — 2026-09-29

User requested deleting all project document data and ChromaDB to re-upload from
scratch. Stopped the running RagSale uvicorn reloader (PID 11671) before reset;
backend must be restarted by the user. Removed all 13 PDFs from uploads/, the
local ChromaDB (178 records), and all 35 generated files in output/ including
extraction/chunk exports, reports and logical storage backups. Checked for and
cleared project-assignment database files if present (none existed).
Original PDFs outside the repository, application code, tests and evaluation
reference fixtures were preserved. Earlier storage counts and validation reports
in this handoff are HISTORICAL; generated report paths no longer exist after reset.
Recreated the empty Chroma collection and checked document/project lists.
Re-upload PDFs through the website; avoid dataset ingestion or maintenance
replacement scripts when starting a fresh PDF-only catalog. Reload the browser
to discard old in-memory chat and pending project selections.

### Multiple PDF selection — implemented 2026-09-29

frontend FileUpload now accepts multiple PDFs and displays per-file waiting,
processing, ready or failed status plus batch progress. api/documents.js adds
uploadDocuments, a sequential queue using the existing single-file backend
endpoint, so each PDF completes extraction/OCR/chunking/embedding/storage before
the next begins. Failures do not abandon later files. Client checks filename,
empty file and 10 MiB size per file; backend still enforces 100 pages. No new
parallel ingestion or backend batch endpoint. Queue requires keeping the page
open; no persisted/resumable queue. Select only failed PDFs to retry.
Three frontend tests passed: sequential processing and error continuation,
invalid-file rejection, and network-error continuation. Production build passed.
No sample uploads were made; the user's reset database remains untouched.

### Upload connection failure resolved — 2026-09-29

All five PDF attempts failed because Vite was running on 5173 but no backend was
listening on 8000 (proxy returned 502). Restarted RagSale via .venv Python uvicorn,
127.0.0.1:8000 --reload (reloader PID 24207 at startup). Left backend running for
user uploads. Verified /api/health through Vite and an empty POST to upload route
(422 expected, no file saved). Updated upload client to explain non-JSON 502/503/
504 connection failures; preserves specific backend JSON errors such as OCR setup.
Five frontend tests and production build passed. User must reselect the failed
PDFs to upload; no documents were uploaded by these diagnostics.

### Evaluation references removed — 2026-09-29

User explicitly requested deletion after being informed of benchmark dependencies.
Deleted all 12 JSON files shown under evaluation/references, including the
three-documents manifest/page references, chunk_contexts.json and copper-pipe
reference. Removed the empty directories. Evaluation scripts remain but their old
benchmark commands require new references and regenerated exports. Added a notice
to evaluation/README.md. This supersedes the earlier reset note that references
were preserved. No uploaded PDFs or ChromaDB records were changed in this step.

### Clear project names bypass unnecessary clarification — 2026-09-29

Added rag/project_names.py and wired it into active project_chat before candidate
retrieval, model scope assessment and the 12-project catalog limit. Match COMPLETE
normalized project names or PDF stems; do not guess from a lone shared keyword.
One match goes directly to answer_in_projects with the original question and
existing document filters. Multiple matches ask only about matching projects.
Comparisons, all/other-project requests and exclusions retain evidence-based scope
assessment. Existing explicit user selections retain priority. This is not the
broader 100–200-project semantic shortlisting or typed-reply feature.
Six new regression cases cover the exact copper-pipe question, normalization,
partial/shared words, comparison/exclusion, duplicate names, filename aliases,
and direct routing with a 200-project catalog. Full suite: 62 passed, one legacy
benchmark error because the user deleted evaluation/references/three-documents/
manifest.json. References were not restored. No stored data was changed.
Live verification: POST /api/chat with "Can you summarize Scope of Work of copper
pipe test ?" returned type=answer directly, no clarification, and cited only
copper pipe.pdf. This verifies routing, not the factual accuracy of every summary
claim; PDF-grounded answer evaluation remains separate.

### Shortened project names — 2026-09-29

Screenshot "Summary washer inspection" exposed that complete-name matching still
required the IAI prefix. Extended project_names.py with contiguous two-word title
matches containing a nongeneric, nonnumeric word of at least three characters.
Full names take priority; multiple shortened matches return only matching choices.
Generic phrases and single words do not force scope. Comparisons/exclusions retain
the previous guard. Added regressions for exact screenshot question, duplicate
washer projects, full-name priority and generic/single-word nonmatches.
Validation: all 19 project-related tests passed. Live POST /api/chat with
"Summary washer inspection" returned type=answer directly and cited only
IAI_Washer Inspection.pdf. This check verifies routing, not summary completeness.

### Mixed Thai/English software question — 2026-09-29

Traced "test ยา ใช้ software หรือ technology อะไรบ้าง". Stored Test Report ยา
contains technology comparison on p5 and software/hardware labels on p6, but
original dense top3 returned p1, p20, p16; an English reformulation also missed
software in its top5. Name resolver did not recognize omitted "Report".
Added explicit document-marker/core-name matching (test ยา -> Test Report ยา).
Added keyword_search.py: length-normalized TF/IDF keyword ranking, English token
boundaries, Thai substring matching, and software/technology English–Thai aliases.
Scoped retrieve now fuses dense and keyword top12 rankings via reciprocal rank
fusion (k=60), preserving document filters and computing cosine distance for
lexical-only selected hits. Global unfiltered retrieval remains dense-only.
No vector migration/re-embedding. Reads scoped text per request; this is not yet
an indexed lexical service for very large corpora or general Thai segmentation.
Live hybrid top3 became p1,p5,p6. Initial generation then wrongly called both
comparison alternatives used; strengthened answer instructions to distinguish
compared vs selected technologies and prioritize explicitly named software.
Tests: 20 project tests, 3 keyword/retrieval tests, 7 answering tests passed.

### Software caption repair and keyword recall safeguard — 2026-09-29

Visually inspected uploaded Test Report ยา.pdf pages 5 and 6. Page 6 contains
three separate captions: Software deep learning studio; Industrial computer
24/7 operation; Industrial Machine Vision Camera & Lighting. Spatial extraction
interleaved their lines. layout.py now groups simple aligned short caption columns
only when native text corroborates each complete phrase. Six layout tests passed.
Answer instructions distinguish hardware/software and comparison alternatives.

Refreshed ONLY page 6 of document 551621d721254b2a849a0aa265d629b1: regenerated
its embedding, saved/read back verified text and metadata, then removed the old
chunk. Total collection count remained 157. Other records were preserved.
The temporary repair script was /tmp/ragsale_refresh_medicine_page6.py; it included
concurrency checks and rollback. This is not a general reindex command.

Re-embedding changed the dense rank and exposed a fusion problem: page 6 dropped
out despite being keyword rank 2. Scoped retrieval now reserves up to ceil(top_k/2)
positions for top keyword results, filling the rest with fused ranks. Four keyword
tests passed, including competing overlapping weak hits, two keyword-only topics,
scope isolation and distance integrity. This heuristic needs broader retrieval
evaluation; it is not calibrated confidence or a general multilingual solution.

Direct live generation now includes Software deep learning studio and cites pages
5 and 6 rather than abstaining. Still imperfect: hardware is grouped under the
broad word technology, and p5's selection note loses its column association in
flattened text. Visually p5 selects Deep Learning, but stored text alone is ambiguous.
Euresys is visible as a logo on p6 and missed by OCR. No brand was manually injected
into evidence. Do not claim comprehensive factual accuracy or OCR perfection.
Final running POST /api/chat check returned type=answer directly, naming Software
deep learning studio and Industrial Machine Vision Camera & Lighting with a valid
page-6 citation. It omitted the p5 technology discussion, so answer completeness
still needs evaluation. git diff --check passed.

### FigJam workflow updated — 2026-09-30

Updated existing board KlSZjBbe6QEFyZdoI7YqcS in place, wrapper 52:267,
preparation section 52:295 and answering section 52:320. Added page metadata
inheritance, chunk IDs, vector/RAM/readback details, project filtering, hybrid
retrieval, current retrieval evaluation focus, and citation versus factual
validation. Preserved top-to-bottom layout and existing detailed failure panels.
Widened truncated labels and visually verified both sections. Evaluation checks
are conceptual checkpoints, not a claim that every live upload/question executes
the full benchmark. No application or database changes in this step.

### SHARP equipment question retrieval — 2026-09-30

Screenshot question: "Sharp auto label ใช้อุปกรณ์อะไรบ้าง". Reproduced top-3
pages 2,3,6; p2 is only a system-purpose description. Chroma already contains
p9's equipment diagram OCR, so this is primarily evidence retrieval failure.
Keyword search treated the unspaced Thai question as an unmatched token and
ranked English project-name matches. Added small Thai alias splitting and
explicit equipment/hardware query expansion into industrial-vision component
terms (camera, lighting, lens, computer, PLC, robot/servo/motor). Component groups
receive 2x lexical weight for explicit equipment intent to counter title matches.
This is a domain heuristic, not general Thai segmentation or evidence insertion.
No page numbers or filenames are hardcoded in the retrieval logic.
Five keyword/temporary-Chroma tests passed, including Thai/English equipment
questions and a no-equipment-intent control. Live generation with retrieved
p9 cited p9 and listed cameras, RS485 and Vision Inspection. The answer remains
incomplete; RS485 is an interface, not a separate equipment item. OCR missed the
5 in "Camera 5 MP (Front)" visible in the user's screenshot. Did not patch source
text, re-embed, or modify database records. Retrieval and extraction completeness
need separate evaluation. No claim that this establishes corpus-wide accuracy.
Final retrieval check after weighting: SHARP top3 pages [2,9,20]; medicine
software regression top3 [1,5,6]. The relevant page is retained, although the
introductory page still ranks first. Broader ranking evaluation remains pending.

### Vocabulary workarounds removed at user request — 2026-09-30

Removed software/technology/equipment alias dictionaries, special Thai alias
splitting, equipment component expansion and its 2x weights from keyword_search.py.
Keyword ranking now uses literal normalized query terms only, with existing generic
stop words and token boundaries. Vector search, rank fusion, reserved keyword slots,
project scope/name resolution and database records are unchanged. Four focused
keyword/temporary-Chroma tests passed, including no implicit translation/expansion
and scoped keyword rescue. README updated. Earlier SHARP/medicine improvements
that depended on aliases are historical, not a guarantee for this version.
Removing vocabulary rules does not resolve the separate 12-project scope fallback
or per-request scoped-text scan; 100–200-PDF scalability still needs evaluation.
The previously shared visualization/FigJam may still show the removed aliases.

### Isolated F2LLM embedding experiment — 2026-10-01

User changed the live embedding model name to codefuse-ai/F2LLM-v2-0.6B while
sales_minilm_l6_v2 still contained 157 384-dimensional MiniLM vectors. Explained
and restored live MiniLM loading to keep the existing collection compatible.
F2LLM is tested separately by `python -m evaluation.compare_embeddings`.
Downloaded model revision 8786315a8711c242ee03ec67c74dd9ad0a61e2cf to HF cache.

Evaluator reads live IDs/text/metadata/vectors but never mutates live records.
It creates a new timestamped collection in chroma_db_f2llm_test/, regenerates
1024-dimensional normalized document embeddings on CPU float32, uses the bundled
query prompt with encode_query, and checks test readback and a before/after
fingerprint of all live record content. Report paths and test DB are gitignored.
It refuses destination paths that overlap the live database. Five focused tests
passed for path guards, vector validation, fingerprints and page-based scoring.
Four smoke cases in evaluation/cases/embedding_smoke.json cover SHARP equipment
p9 and medicine software/technology p5–6, each in English and Thai. Tests bypass
scope resolution, hybrid search and Qwen; they compare exact cosine over stored
MiniLM vectors with cosine Chroma retrieval over F2LLM vectors. Both document and
global scopes are reported. Results are not full RAG accuracy or a broad benchmark.
Completed run: output/embedding-comparison/20261001T020305534105Z/report.json
(and report.md). Test collection f2llm_v2_06b_20261001t020305534105z, 157 records.
Live source fingerprint unchanged; test readback passed. CPU document embedding
182.0s, total query encoding for four questions MiniLM 0.12s vs F2LLM 1.99s.
Document-scoped Page Recall@5: SHARP English MiniLM100% vs F2LLM0%; SHARP Thai
both0%; medicine English MiniLM0% vs F2LLM100%; medicine Thai MiniLM0% vs F2LLM50%.
Global expected-page Recall@5 was0% for all four questions with both models.
Do NOT promote F2LLM based on this mixed small sample. Current app stays MiniLM.
Fresh-process reopen also verified live fingerprint unchanged and all 157 isolated
records persisted with the same IDs/text/metadata and 1024-dimensional vectors.

### Qwen answers from F2LLM test retrieval — 2026-10-01

Added evaluation/generate_test_answers.py. Explicit --retrieval-report is required;
replays candidate ranked IDs from the report, reads actual source text from the
isolated test collection, and calls production generate_answer. Defaults top5,
document scope; --top-k 3 and --scope global supported. It does not load embeddings,
run project resolution/hybrid search, or open the live database. Expected pages
are only for reporting and are never sent to Qwen. Saves incremental answers.json
and answers.md including all input chunks, cited sources, status/errors, elapsed
time, prompt hash and before/after test fingerprint. No automatic factual score.
Three tests passed for input evidence/answer-key separation, bad sources and errors.

Ran against output/embedding-comparison/20261001T021036665225Z/report.json with
qwen3:8b. Outputs: answers-20261001T022346553822Z/{answers.json,answers.md,review.md}
under that report directory. Four answers completed; citation IDs passed and test
collection unchanged. SHARP EN gives only system purpose (p3); SHARP TH uses p20
camera compatibility specs, not the missing p9 overview. Medicine EN names software
(p6); medicine TH describes supervised learning (p9) but omits missing p6 software.
Compatibility is not proof of selected deployment. No blanket correctness claim.
First generation83.12s; others2.97/1.74/2.11s. Live app/database unchanged this turn.

### Live switch to F2LLM and complete Chroma reset — 2026-10-01

User explicitly requested deleting all Chroma data and switching the live app.
Stopped old API PID13837. Deleted sales_minilm_l6_v2 (157 records) from chroma_db
and both F2LLM experiment collections (157 records each) from chroma_db_f2llm_test.
Created empty live sales_f2llm_v2_06b. PDFs in uploads, project-assignment registry,
model caches and historical report files were preserved. No PDFs were reindexed.

embedding.py now loads codefuse-ai/F2LLM-v2-0.6B on CPU float32; documents use
encode_document (batch2, normalized), both retrieval entry points use encode_query
with the bundled query prompt. Dimension1024. Qwen remains unchanged. Three encoder
unit tests and ten project-chat/scope tests passed. Real cached-model document/query
smoke test passed for dimension, finite values and unit norms without storing data.
Restarted uvicorn --reload on127.0.0.1:8000 (reloader58111).

Historical evaluation scripts/reports depend on now-deleted collections; README
and evaluation README flag this. Earlier notes saying live MiniLM is retained are
superseded by this explicit user switch. User must upload PDFs again to populate
F2LLM. This switch is a trial, not an assertion F2LLM won the earlier evaluation.

### SHARP page 9 extraction audit - 2026-10-01

User has reuploaded documents after reset. Read-only audit of current SHARP PDF
uploads/4bbefb012c7e41db858e06c4c8822fb8.pdf, page9 (Vision concept).
Native text is only the title; equipment labels are inside a1257x584 diagram image.
Current image OCR loses5 in Camera5MP, adds graphic noise, and merges back-camera
function text with RS485. Both12MP camera labels and major program/control labels
are present. Recomputed current extraction exactly matches live F2LLM collection's
page9 text, stored in one chunk (chunk_index8). No storage loss/chunk split found.
Existing full-page OCR helper (300DPI/max4000px,PSM11) recovers5MP and cleaner labels,
but still has noise and does not preserve diagram relationships. Current image
OCR usesPSM6. Trial changes both rendering and segmentation, not an isolated test.
Saved output/extraction/sharp-page9-audit/{report.md,native.txt,current.txt,
full-page-ocr.txt,stored.json}. No pipeline/DB mutations. Extraction defects are
confirmed; retrieval ranking/generation causality is not established by this audit.

### OCR image layout improvement - 2026-10-01

ocr.py now uses PSM11 and up to2x enlargement for wide embedded images
(width/height>=1.8); compact images retain original resolution/PSM6. Both cap
longest side at4000px. Generic geometry heuristic, no filenames/vocabulary rules.
An all-sparse trial broke copper metrics label/value grouping, so compact-image
behavior is deliberately preserved. No semantic diagram classifier/vision added.
loader metadata records ocr_version=layout-images-v2 (none for native-only pages).
SHARP p9 now includes Camera5MP and all10 manually selected main labels; graphic
noise/reading-order issues remain. Copper p8 metrics/matrix section unchanged.
Also compared Washer p4/p6 and Imco p10 (mostly background/logo image smoke tests,
not evidence of broad OCR accuracy). Outputs in output/extraction/ocr-v2-comparison
with report.md and raw before/after OCR text. 11 OCR+6 layout+2 upload tests passed.
No live Chroma mutation/reindex performed. Existing indexed documents still have
old OCR; replace/reindex them before claiming improved retrieval. New uploads use
new code after backend reload. Do not duplicate-upload existing documents blindly.

### Clear live index for OCR testing - 2026-10-01

User explicitly requested clearing old Chroma data before testing new uploads.
Deleted all157 records belonging to6 documents from sales_f2llm_v2_06b using
record IDs; retained the collection configuration. Count after deletion:0.
Uploaded PDF files, project registry, model caches and evaluation reports retained.
User can now upload fresh copies to test layout-images-v2 OCR with F2LLM.

### Post-upload SHARP retrieval diagnosis - 2026-10-01

User reuploaded; live count159. SHARP document74058bdcf34c4315b1885c1954d7094c
page9 confirms ocr_version=layout-images-v2 and improved Camera5MP/12MP labels.
Read-only replay of both screenshot questions, scoped to that document, default
top_k3: Thai/mixed query page9 dense rank10; English query rank11. Lexical rank11
for both. Final hybrid pages are3,2,6 for both, so page9 is not passed to generation.
Page9 does enter dense candidate pool12 but loses final selection. Reports saved
output/retrieval/ocr-v2/sharp-ranks.json with all dense/lexical ranks and final texts.
No Qwen generation replay or retrieval code change this turn. Next evaluate
semantic reranking of candidates before final3, with held-out questions; no
document/page-specific boosts. OCR remains noisy, but storage/new-version confirmed.
