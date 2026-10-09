"""Check exported chunks against saved extraction and reviewed field references.

No model downloads, embeddings, database writes, or LLM calls are performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from .evaluate_extraction import evaluate, normalize


def source_key(metadata):
    return metadata.get('filename'), metadata.get('page')


def evaluate_chunks(pages, chunks, references, contexts=(), max_size=1000):
    """Evaluate exact text coverage, provenance, field association and context.

    Align chunks in chunk_index order to successive source occurrences. Never
    mark every occurrence of a repeated substring covered by one chunk.
    Whitespace omitted by the splitter is intentionally ignored.
    """
    errors = []
    page_map = {source_key(p['metadata']): p for p in pages}
    if len(page_map) != len(pages):
        raise ValueError('Duplicate filename/page in extraction; evaluate one document at a time.')
    indices = [c['metadata'].get('chunk_index') for c in chunks]
    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(chunks))):
        errors.append('Chunk indices must be unique consecutive integers starting at zero.')
    for position, chunk in enumerate(chunks):
        text, metadata = chunk['text'], chunk['metadata']
        page = page_map.get(source_key(metadata))
        if not text.strip():
            errors.append(f'Chunk {position}: empty text.')
        if len(text) > max_size:
            errors.append(f'Chunk {position}: {len(text)} characters exceeds {max_size}.')
        if page is None:
            errors.append(f'Chunk {position}: filename/page absent from extraction.')
        elif any(metadata.get(k) != v for k, v in page['metadata'].items()):
            errors.append(f'Chunk {position}: source metadata changed.')
    page_checks = []
    review_items = []
    for key, page in page_map.items():
        text = page['text']
        selected = [c for c in chunks if source_key(c['metadata']) == key]
        selected.sort(key=lambda c: c['metadata'].get('chunk_index') if type(c['metadata'].get('chunk_index')) is int else -1)
        covered = bytearray(len(text))
        previous_start = -1
        for chunk in selected:
            # Earliest increasing occurrence gives deterministic conservative
            # alignment. Arbitrary/reordered exports can fail this check.
            start = text.find(chunk['text'], previous_start + 1) if chunk['text'] else -1
            if start < 0:
                errors.append(f'{key}: chunk {chunk["metadata"].get("chunk_index")} cannot align to source in order.')
                continue
            end = start + len(chunk['text'])
            covered[start:end] = b'\1' * (end-start)
            previous_start = start
        missing = [i for i, ch in enumerate(text) if not ch.isspace() and not covered[i]]
        page_checks.append({'filename':key[0], 'page':key[1], 'chunks':len(selected),
                            'missing_nonwhitespace_characters':len(missing),
                            'missing_preview': ''.join(text[i] for i in missing)[:160]})
        if missing:
            errors.append(f'{key}: {len(missing)} non-whitespace characters are uncovered.')
        # These are review warnings, not semantic accuracy judgments. A line
        # may validly be longer than the configured chunk size.
        for line in text.splitlines():
            if line.strip() and not any(normalize(line) in normalize(c['text']) for c in selected):
                review_items.append({'filename':key[0], 'page':key[1],
                                     'type':'source_line_split', 'text':line.strip()})
        for paragraph in re.split(r"\n\s*\n", text):
            if len(paragraph.strip()) >= 80 and not any(normalize(paragraph) in normalize(c['text']) for c in selected):
                review_items.append({'filename':key[0], 'page':key[1],
                                     'type':'source_paragraph_split',
                                     'text':paragraph.strip()[:220],
                                     'source_characters':len(paragraph.strip())})
    field_checks = []
    for reference in references:
        for field in reference['fields']:
            single = {**reference, 'fields':[field]}
            extraction_result = evaluate(single, pages)['results'][0]
            aggregate = evaluate(single, chunks)['results'][0]
            matches = [c['metadata'].get('chunk_index') for c in chunks
                       if evaluate(single, [c])['correct_fields'] == 1]
            passed = bool(matches) and aggregate['passed']
            field_checks.append({'filename':reference['filename'], 'page':reference['page'],
                                 'label':field['label'], 'expected':field['value']+field['unit'],
                                 'source_passed':extraction_result['passed'], 'passed':passed,
                                 'chunk_indices':matches, 'reason':aggregate['reason']})
    context_checks = []
    for case in contexts:
        def contains(entry):
            text = normalize(entry['text'])
            return all(normalize(phrase) in text for phrase in case['required_phrases'])
        selected_pages = [p for p in pages if source_key(p['metadata']) == (case['filename'],case['page'])]
        selected_chunks = [c for c in chunks if source_key(c['metadata']) == (case['filename'],case['page'])]
        matches = [c['metadata'].get('chunk_index') for c in selected_chunks if contains(c)]
        context_checks.append({'id':case['id'], 'filename':case['filename'], 'page':case['page'],
                               'source_passed':any(contains(p) for p in selected_pages),
                               'passed':bool(matches), 'chunk_indices':matches})
    return {'chunks':len(chunks), 'integrity_passed':not errors, 'integrity_errors':errors,
            'page_checks':page_checks, 'field_checks':field_checks,
            'context_checks':context_checks, 'boundary_review':review_items,
            'passed':not errors and all(f['passed'] for f in field_checks) and all(c['passed'] for c in context_checks)}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extraction-dir',type=Path,default=Path('output/extraction/layout-improved'))
    parser.add_argument('--chunks-dir',type=Path,default=Path('output/chunks'))
    parser.add_argument('--output-dir',type=Path,default=Path('output/chunks/chunking-test'))
    parser.add_argument('--max-size',type=int,default=1000)
    args=parser.parse_args()
    if args.max_size<1: parser.error('--max-size must be positive')
    reference_dir=Path(__file__).parent/'references/three-documents'
    inputs={}
    def tracked(path):
        value=read_json(path)
        inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return value
    try:
        contexts=tracked(Path(__file__).parent/'references/chunk_contexts.json')['cases']
        documents=[]
        for doc in tracked(reference_dir/'manifest.json'):
            pages=tracked(args.extraction_dir/f'{doc["id"]}.json')
            exported=tracked(args.chunks_dir/f'{doc["id"]}-improved-chunks.json')
            if exported['count'] != len(exported['chunks']):
                raise ValueError(f'{doc["id"]}: declared export count differs from actual count')
            references=[tracked(reference_dir/name) for name in doc['references']]
            result=evaluate_chunks(pages,exported['chunks'],references,
                                   [c for c in contexts if c['filename']==doc['filename']],args.max_size)
            documents.append({'filename':doc['filename'],**result})
    except (OSError,ValueError,KeyError,TypeError) as exc:
        parser.error(str(exc))
    report={'scope':'Saved export tests; no fresh PDF transcription, OCR, retrieval, or answer evaluation.',
            'max_chunk_characters':args.max_size,'input_sha256':inputs,'documents':documents,
            'passed':all(d['passed'] for d in documents)}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Chunking test results','','Checks on saved improved exports; source extraction is not independently revalidated.','',
           '| Document | Chunks | Integrity | Reference fields | Context groups | Boundary review candidates |',
           '|---|---:|---|---:|---:|---:|']
    for d in documents:
        fields=d['field_checks']; contexts=d['context_checks']
        lines.append(f'| {d["filename"]} | {d["chunks"]} | {"PASS" if d["integrity_passed"] else "FAIL"} | {sum(f["passed"] for f in fields)}/{len(fields)} | {sum(c["passed"] for c in contexts)}/{len(contexts)} | {len(d["boundary_review"])} |')
    lines+=['','Integrity checks: exact ordered source membership, non-whitespace coverage, source metadata, chunk indices and maximum size.',
            '', 'Context cases test phrases co-occurring in a chunk on the correct page. They were selected from current extraction and are development regression checks, not independent OCR ground truth.',
            '', 'A passing result does not measure overall semantic quality, visual understanding, retrieval or generated answers. Boundary warnings are review candidates, not automatic errors.']
    for d in documents:
        lines+=['',f'## {d["filename"]}']
        for error in d['integrity_errors']:lines.append(f'- FAIL: {error}')
        for c in d['context_checks']:lines.append(f'- {"PASS" if c["passed"] else "FAIL"}: {c["id"]}, page {c["page"]}, chunks {c["chunk_indices"]}.')
        for f in d['field_checks']:
            if not f['passed']:lines.append(f'- FAIL field: {f["label"]}, page {f["page"]}: {f["reason"]}.')
        for item in d['boundary_review']:
            lines+=['',f'Boundary review ({item["type"]}), page {item["page"]}:','', '> '+item['text'].replace('\n',' ')]
    (args.output_dir/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:10]))
    print(f'Report: {args.output_dir / "report.md"}')
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
