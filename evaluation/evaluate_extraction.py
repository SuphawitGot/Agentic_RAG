"""Evaluate reference metric fields against an OCR preview using only stdlib.

Conservative same-line matching: label, value and unit must remain associated.
This is field accuracy, not whole-document accuracy or character error rate.
"""

import argparse
import json
from pathlib import Path
import re
import unicodedata


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def evaluate(reference, extraction):
    fields = reference["fields"]
    if not fields:
        raise ValueError("The reference must contain at least one field.")
    pages = [entry for entry in extraction
             if entry["metadata"].get("filename") == reference["filename"]
             and entry["metadata"].get("page") == reference["page"]]
    lines = [normalize(line) for entry in pages for line in entry["text"].splitlines()]
    results = []
    for field in fields:
        label = normalize(field["label"])
        # Anchor at line start: 'Image detection accuracy' must not match
        # the different field 'Best achievable image detection accuracy'.
        pattern = re.compile(r"^" + re.escape(label) + r"\s*:?\s*(.*)$")
        observed = [match.group(1) for line in lines if (match := pattern.match(line))]
        expected = normalize(field["value"] + field["unit"])
        # Ignore decorative screenshot icons recognized as @ or *, but do
        # not discard words, numbers or units that might change the value.
        values = [normalize(value.rstrip(" @*" )).replace(" ", "") for value in observed]
        passed = bool(values) and all(value == expected.replace(" ", "") for value in values)
        reason = ("match" if passed else "source page missing" if not pages else
                  "label missing or not on the same line as its value" if not observed else
                  "value/unit mismatch or conflicting duplicate")
        results.append({"label": field["label"], "expected": field["value"] + field["unit"],
                        "observed": observed, "passed": passed, "reason": reason})
    correct = sum(result["passed"] for result in results)
    return {"filename": reference["filename"], "page": reference["page"],
            "correct_fields": correct, "total_fields": len(fields),
            "field_accuracy_percent": 100 * correct / len(fields), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON evaluation report")
    args = parser.parse_args()
    try:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        extraction = json.loads(args.extraction.read_text(encoding="utf-8"))
        report = evaluate(reference, extraction)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(f"{report['filename']} — page {report['page']}")
    for result in report["results"]:
        print(f"{'PASS' if result['passed'] else 'FAIL'} {result['label']}: expected {result['expected']}")
        if not result["passed"]:
            print(f"     {result['reason']}; observed: {result['observed']}")
    print(f"Field accuracy: {report['correct_fields']}/{report['total_fields']} "
          f"({report['field_accuracy_percent']:.1f}%)")
    print("Scope: reference fields only; not overall OCR or RAG accuracy.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if report["correct_fields"] == report["total_fields"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
