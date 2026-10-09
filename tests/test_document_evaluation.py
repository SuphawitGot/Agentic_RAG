import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.evaluate_documents import run
from evaluation.evaluate_extraction import evaluate


class DocumentBenchmarkTests(unittest.TestCase):
    def test_missing_documents_are_reported_and_not_removed_from_denominator(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            report = run(root / 'missing-pdfs', root / 'results')
            self.assertEqual(len(report['documents']), 3)
            self.assertEqual(report['total_fields'], 34)
            self.assertEqual(report['correct_fields'], 0)
            self.assertTrue(all(d['status'] == 'error' for d in report['documents']))
            self.assertTrue((root / 'results/summary.md').exists())
            self.assertEqual(json.loads((root / 'results/summary.json').read_text())['total_fields'], 34)

    def test_thai_field_and_wrong_value(self):
        reference = {'filename': 'thai.pdf', 'page': 10, 'fields': [
            {'label': 'จุดที่ตรวจสอบ', 'value': 'ปากขวดไม่เรียบ', 'unit': ''}]}
        def score(text):
            return evaluate(reference, [{'metadata': {'filename': 'thai.pdf', 'page': 10}, 'text': text}])['correct_fields']
        self.assertEqual(score('จุดที่ตรวจสอบ : ปากขวดไม่เรียบ'), 1)
        self.assertEqual(score('จุดที่ตรวจสอบ : ภายในขวด'), 0)

    def test_separated_table_columns_are_not_assumed_to_be_associated(self):
        reference = {'filename': 'spec.pdf', 'page': 1, 'fields': [
            {'label': 'WD', 'value': '800', 'unit': 'mm'}]}
        extraction = [{'metadata': {'filename': 'spec.pdf', 'page': 1}, 'text': 'FOV\nWD\n212 mm x 155 mm\n800 mm'}]
        self.assertEqual(evaluate(reference, extraction)['correct_fields'], 0)


if __name__ == '__main__':
    unittest.main()
