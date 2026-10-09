import unittest
from evaluation.generate_test_answers import prepare_matches, run_case
from ragsale.rag.answering import GenerationError


class TestAnswerExperiment(unittest.TestCase):
    def setUp(self):
        self.row = {'case': 'case', 'question': 'Which equipment?', 'scope': 'document',
                    'expected_filename': 'test.pdf', 'expected_pages': [9],
                    'top5': [{'id': 'a', 'filename': 'test.pdf', 'page': 2, 'cosine_distance': .3}]}
        self.records = {'a': {'text': 'System introduction', 'metadata': {'filename': 'test.pdf', 'page': 2}}}

    def test_only_retrieved_evidence_is_sent_without_answer_key(self):
        matches = prepare_matches(self.row, self.records, 5)
        def fake(question, evidence):
            self.assertEqual(question, 'Which equipment?')
            self.assertEqual(evidence, matches)
            self.assertNotIn('expected_pages', evidence[0])
            return {'answer': 'Insufficient', 'insufficient_evidence': True, 'sources': []}
        result = run_case(self.row, matches, fake)
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(result['expected_page_recall'], 0)
        self.assertEqual(result['retrieved_sources'][0]['citation'], 1)

    def test_stale_or_out_of_scope_sources_rejected(self):
        for records in [{}, {'a': {'text': 'changed', 'metadata': {'filename': 'other.pdf', 'page': 2}}}]:
            with self.assertRaises(ValueError):
                prepare_matches(self.row, records, 5)

    def test_generation_error_is_recorded(self):
        def fail(*args):
            raise GenerationError('Invalid citation')
        r = run_case(self.row, prepare_matches(self.row, self.records, 3), fail)
        self.assertEqual(r['status'], 'error')
        self.assertEqual(r['error'], 'Invalid citation')
        self.assertEqual(r['factual_review'], 'not_scored_requires_source_review')
