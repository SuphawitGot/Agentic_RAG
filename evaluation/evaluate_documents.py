"""Run the three-document extraction benchmark without touching ChromaDB."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from evaluation.evaluate_extraction import evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(pdf_dir, output_dir, reuse=False):
    reference_dir = Path(__file__).parent / 'references' / 'three-documents'
    manifest = json.loads((reference_dir / 'manifest.json').read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    documents = []
    for case in manifest:
        name = case['filename']
        print(f'Evaluating {name}', flush=True)
        references = [json.loads((reference_dir / path).read_text()) for path in case['references']]
        result = {'filename': name, 'reference_fields': sum(len(r['fields']) for r in references)}
        try:
            source = pdf_dir / name
            source_hash = digest(source)
            if any(r['source_sha256'] != source_hash for r in references):
                raise ValueError('PDF differs from the visually verified reference. Review the reference before testing.')
            extraction_path = output_dir / (case['id'] + '.json')
            if not reuse:
                from ragsale.rag.loader import load_pdf_documents
                docs = load_pdf_documents(source, 'evaluation', name)
                extraction = [{'text': d.page_content, 'metadata': d.metadata} for d in docs]
                extraction_path.write_text(json.dumps(extraction, ensure_ascii=False, indent=2), encoding='utf-8')
            else:
                extraction = json.loads(extraction_path.read_text(encoding='utf-8'))
            pages = [evaluate(ref, extraction) for ref in references]
            passed = sum(p['correct_fields'] for p in pages)
            result.update(status='evaluated', source_sha256=source_hash,
                          extraction_sha256=digest(extraction_path), pages=pages,
                          correct_fields=passed, field_accuracy_percent=100*passed/result['reference_fields'])
        except Exception as error:
            result.update(status='error', error=str(error), correct_fields=0, field_accuracy_percent=0)
        documents.append(result)
        print(f"  {result['correct_fields']}/{result['reference_fields']} fields ({result['field_accuracy_percent']:.1f}%)", flush=True)
    total = sum(d['reference_fields'] for d in documents)
    correct = sum(d['correct_fields'] for d in documents)
    report = {'scope': 'Strict same-line label/value/unit checks on nine selected pages. Not whole-document OCR accuracy, CER, visual understanding or RAG accuracy.',
              'reused_extractions': reuse, 'correct_fields': correct, 'total_fields': total,
              'field_accuracy_percent': 100*correct/total, 'documents': documents}
    (output_dir/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# Extraction benchmark', '', report['scope'], '',
             '| Document | Correct fields | Field accuracy |', '|---|---:|---:|']
    for d in documents:
        lines.append(f"| {d['filename']} | {d['correct_fields']}/{d['reference_fields']} | {d['field_accuracy_percent']:.1f}% |")
    lines += ['', f'Overall: {correct}/{total} ({report["field_accuracy_percent"]:.1f}%).', '',
              'Failures below require review: a strict line-association failure does not necessarily mean the value is absent.', '']
    for d in documents:
        lines += [f"## {d['filename']}", '']
        if d['status']=='error': lines += [d['error'], '']; continue
        for p in d['pages']:
            for field in p['results']:
                if not field['passed']:
                    lines += [f"- Page {p['page']}, **{field['label']}**: expected `{field['expected']}`; {field['reason']}. Observed: `{field['observed']}`."]
        lines.append('')
    (output_dir/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('output/extraction/three-documents'))
    parser.add_argument('--reuse-extractions', action='store_true', help='Score existing previews instead of running extraction; these may be stale.')
    args=parser.parse_args()
    report=run(args.pdf_dir,args.output_dir,args.reuse_extractions)
    print(f"Overall: {report['correct_fields']}/{report['total_fields']} ({report['field_accuracy_percent']:.1f}%). Report: {args.output_dir/'summary.md'}")
    return 2 if any(d['status']=='error' for d in report['documents']) else int(report['correct_fields'] != report['total_fields'])


if __name__=='__main__':
    sys.exit(main())
