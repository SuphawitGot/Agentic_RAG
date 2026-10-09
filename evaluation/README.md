> **2026-10-01 reset:** The live app now uses F2LLM (`sales_f2llm_v2_06b`).
> The old MiniLM and F2LLM experiment collections were deleted at the user's
> request. Reports remain historical. The comparison and answer-replay commands
> below depend on those old collections; recreate an appropriate isolated
> benchmark before using them again. Do not mix models in the new live collection.

# Isolated embedding comparison

Run from the repository root:

```bash
uv run python -m evaluation.compare_embeddings
```

This reads the existing `sales_minilm_l6_v2` collection, reuses all chunk text and
metadata, and embeds it with `codefuse-ai/F2LLM-v2-0.6B` on CPU (float32, batch size
2). The first run downloads model weights. The live app stays on MiniLM.

- Source: `chroma_db/`, opened using `get_collection`, with no record mutations.
- Destination: `chroma_db_f2llm_test/`, a new timestamped collection each run.
- Reports: `output/embedding-comparison/<timestamp>/report.json` and `report.md`.
- Test cases: `evaluation/cases/embedding_smoke.json`. The four questions cover two
  known evidence targets, in English and Thai. They are a smoke test, not a broad
  benchmark. Extend the cases only after reviewing the original PDFs.

The evaluator verifies vector count, dimension, finite values and normalization;
reads the test records back; compares copied text/metadata and generated vectors;
and fingerprints ALL live record IDs, text, metadata and vectors before/after.
Avoid concurrent uploads or deletes during a comparison: they would invalidate
the unchanged-source check. No collection is deleted or replaced by this script.

Question encoding uses the candidate's bundled query prompt via `encode_query`;
documents use `encode_document`. Existing MiniLM vectors are queried with MiniLM
question embeddings. The test excludes aliases, keyword fusion, project selection
and Qwen generation so it can examine vector retrieval. It runs both filename-scoped
and global searches. Baseline search is exact cosine in memory over saved vectors;
candidate search uses the test Chroma cosine index. Page Recall@K counts unique
expected PDF pages, not chunk precision or answer accuracy. Timings exclude model
loading and are single-run observations, not throughput benchmarks.

The separate database and output paths are ignored by Git. To experiment with
paths, use `--test-db` and `--output`; neither can overlap the live database. A run
creates another collection rather than clearing an older experiment. Do not point
the application at a candidate collection until retrieval and latency are evaluated.

## Qwen answers from an existing F2LLM experiment

```bash
uv run python -m evaluation.generate_test_answers \
  --retrieval-report output/embedding-comparison/20261001T021036665225Z/report.json
```

Use your own completed report path. This replays the saved F2LLM ranked IDs,
fetches their actual text from that isolated test collection, and calls the same
`generate_answer()` function used by the application. It does not re-embed chunks,
run hybrid search, resolve project scope or access the live database.

Defaults: document scope and top 5, to match the prior retrieval report. Use
`--top-k 3` to test the application's default context budget or `--scope global`
to replay global search. These are separate diagnostic runs, not the complete
live chat pipeline. Ollama must be running with `qwen3:8b` (or `QWEN_MODEL`).

Results are saved incrementally in `answers-<timestamp>/answers.json` and
`answers.md` beside the retrieval report. Each includes the question, expected
pages, ALL supplied chunks, cited sources, answer/abstention/error and elapsed
generation time. Expected pages never go into the generation prompt. Read both
the retrieved context and original PDF to review factual correctness; citation
validation alone does not score accuracy. The evaluator verifies the test
collection fingerprint before/after and never opens the production database.

## Historical extraction benchmarks

> **2026-09-29:** The old reference JSON files were deleted at the user's request.
> Commands below describe the historical benchmarks and require new reference files
> (and regenerated extraction/chunk exports) before they can run again.

# Extraction field evaluation

`references/copper-pipe-page-08.json` is a visually verified answer key for the
five metrics in page 8's Metrics panel. It does not cover the confusion matrix.
Keep reference values independent from OCR output; correct a reference only after
checking the original PDF.

From the project root, generate a fresh extraction preview:

```bash
uv run python -m ragsale.rag.ocr "/home/pui117/Desktop/Test raw data/Vision test report/copper pipe.pdf" \
  --output output/extraction/copper-pipe.json
```

Then evaluate it:

```bash
uv run python evaluation/evaluate_extraction.py \
  --reference evaluation/references/copper-pipe-page-08.json \
  --extraction output/extraction/copper-pipe.json \
  --output output/extraction/copper-pipe-page-08-evaluation.json
```

The evaluator selects the exact filename and page, then checks each complete
label and its associated value/unit on the same line. It tolerates case,
whitespace, an optional colon, and trailing `@`/`*` screenshot icon noise.
It preserves numeric formatting (including decimal places) and units. Missing
labels, wrong numbers, missing units, conflicting duplicates, and values separated
from their labels fail. This intentionally conservative line-based rule is for
these metric fields; more complex tables need their own structured evaluation.

The report lists expected and observed values, reasons for failures and
`correct fields / total fields`. Exit code 0 means all fields pass, 1 means
at least one fails, and 2 means invalid input. Neither command changes ChromaDB.

The initial existing preview scores 5/5 (100%) on these five fields only. This is
not overall extraction accuracy, CER, table accuracy, or RAG answer accuracy.
Add independently verified references for other pages to expand coverage.

## Three-document benchmark

Nine visually checked pages, 34 reference fields:

| Document | Pages | Coverage |
|---|---|---|
| IAI_Washer Inspection.pdf | 1, 4, 5 | Report date/code, hardware and specification tables |
| Imco foodpack V1a 2.pdf | 4, 10, 47 | Scope table, Thai checkpoint, five screenshot metrics |
| SHARP AUTO LABEL INSPECTION-REVISE1.pdf | 4, 11, 21 | Scope table, diagram camera labels/specs, lead time/warranty |

Run from the project root:

```bash
uv run python -m evaluation.evaluate_documents \
  --pdf-dir "/home/pui117/Desktop/Test raw data/Test_Rag"
```

This runs the current extraction pipeline on all three PDFs, without embedding,
calling Qwen, or writing to Chroma. OCR can take several minutes. References are
in `evaluation/references/three-documents/`; each contains a PDF SHA-256 so a
changed source cannot silently be evaluated against an outdated reference.

Outputs are under `output/extraction/three-documents/`:

- `washer.json`, `imco.json`, `sharp.json`: raw page extraction previews.
- `summary.json`: all field results, source/extraction hashes, errors and scores.
- `summary.md`: readable scores and failing fields.

Use `--reuse-extractions` only to rescore existing previews; it does not rerun OCR,
and old previews may not represent current extractor settings. The report records
whether this option was used. Exit codes: 0 all fields pass; 1 benchmark failures;
2 extraction/source/setup errors. A failed document is counted as zero passing
fields and remains in the denominator instead of disappearing from the score.

Interpretation: strict same-line **field association** accuracy on selected fields.
Line-wrapped labels or separated table columns can fail even if every character
is present elsewhere. Inspect the failing page before deciding whether to fix
OCR, layout handling, or extend the evaluator's declared matching rules. Do not
change references to copy OCR mistakes or relax matching to simply find a number
anywhere on the page. These samples are not statistically representative accuracy
estimates. Photo interpretation, whole-page CER, confusion matrices in screenshots,
and most Thai prose are outside this initial benchmark.

### Layout extraction improvement (2026-09-21)

Using the same nine reference pages and unchanged scoring rules:

| Document | Previous | Layout-aware extraction |
|---|---:|---:|
| Washer | 11/12 | 12/12 |
| Imco | 7/10 | 10/10 |
| SHARP | 8/12 | 12/12 |
| Total | 26/34 (76.5%) | 34/34 (100%) |

Detailed outputs: `output/extraction/layout-improved/`. These are development-set
results on the documents used to investigate the failures, not held-out accuracy.
Scanned table structure, defect-photo understanding, most Thai prose, and unseen
layouts remain outside these 34 checks. The reference JSONs and evaluator were
not changed to obtain this improvement.

## Chunking regression checks

From the repository root, test the saved improved chunk exports:

```bash
uv run python -m evaluation.evaluate_chunking
```

This reads `output/extraction/layout-improved/{washer,imco,sharp}.json` and
`output/chunks/{washer,imco,sharp}-improved-chunks.json`. It does not regenerate
chunks: after changing the splitter, export fresh chunks first. Missing input
files cause an explicit error; they are not silently skipped. Override locations
with `--extraction-dir`, `--chunks-dir`, and `--output-dir`; use `--max-size` when
checking exports made with another chunk size. The check does not verify that
actual overlap equals the configured overlap; recursive splitting need not
produce a fixed overlap at every boundary.

The test checks:

1. Chunk indices, nonempty text, maximum character count and source metadata.
2. Exact membership in the source text and non-whitespace coverage. It aligns
   successive chunks to increasing source positions, so one occurrence cannot
   falsely cover every repeated occurrence. This is a conservative check for
   these ordered exports, not a general alignment algorithm for rewritten text.
3. The existing 34 label/value/unit references: each must occur intact in at
   least one chunk on the correct page. Conflicting duplicates fail.
4. Eleven context groups in `references/chunk_contexts.json`: related headings
   and phrases must coexist in a correctly attributed chunk. These were selected
   from the current extraction, not independently transcribed from PDFs.
5. Source lines/paragraphs split between chunks, listed as manual-review warnings.
   A long paragraph often must split; this alone is not a semantic failure.

Reports are `output/chunks/chunking-test/report.md` and `report.json`. JSON includes
input SHA-256 hashes and the chunk indices containing each reference/context.
Exit 0 means hard checks pass (warnings may remain), 1 means a check failed,
and 2 means an input/setup error. Source extraction failures are distinguished
from chunk failures using `source_passed` in field/context results.

On 2026-09-22: all 105 improved chunks passed integrity checks, 34/34 fields and
11/11 selected context groups passed, and nine split paragraphs were flagged
for review. Eight flags were OCR blocks (Imco pages 25/27/28/50 and SHARP pages
17–20); the remaining flag was Washer's NDA page 2. These results are development
regressions, not a whole-document accuracy percentage or a retrieval benchmark.
Do not increase chunk size solely to eliminate these warnings.

Check the evaluator itself with intentionally damaged examples:

```bash
uv run python -m unittest discover -s tests -p test_chunking_evaluation.py -v
```

Nine tests cover text loss/invention, repeated content, wrong provenance,
separated fields/context, conflicting values, overlap, size and chunk indices.
No embeddings, database writes, model calls or OCR runs occur in these checks.

## Store checked improved exports in the existing database

`uv run python -m evaluation.store_checked_chunks` is an intentional maintenance
write, not a read-only evaluation. Stop concurrent uploads/writers before running.
It requires exactly one existing document ID per benchmark filename, validates
the saved improved exports, embeds them, checks vectors, saves a full logical
backup, upserts the improved data and verifies it before removing obsolete IDs.
Existing IDs at the document level and all unrelated records are preserved.
It checks for changes before writing and attempts rollback on failure, but is
not an atomic transaction and does not support concurrent writers.

Results and backup live under output/extraction/storage/<UTC timestamp>/.
On 2026-09-24, 105 improved chunks were stored; 73 unrelated records were preserved
(total 178). A separate Python process reopened Chroma and compared all expected
IDs, text, metadata and vectors successfully. Retrieval accuracy is still pending.
