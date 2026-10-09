import unittest
from unittest.mock import AsyncMock, patch

import httpx

from ragsale.api.main import app
from ragsale.rag.clarification import decide_scope


DOCS = [
    {'document_id': 'washer', 'filename': 'IAI_Washer Inspection.pdf'},
    {'document_id': 'imco', 'filename': 'Imco foodpack V1a 2.pdf'},
    {'document_id': 'sharp', 'filename': 'SHARP AUTO LABEL INSPECTION-REVISE1.pdf'},
]


class ScopeTests(unittest.TestCase):
    def test_ambiguous_question_offers_real_documents(self):
        result = decide_scope('What is the inspection result?', DOCS)
        self.assertEqual(result['status'], 'clarification')
        self.assertEqual(result['options'], DOCS)
        self.assertEqual(result['document_ids'], [])

    def test_named_document_and_no_substring_match(self):
        for question in ['What about IAI Washer Inspection?', 'Washer result?', 'SHARP result?']:
            self.assertEqual(decide_scope(question, DOCS)['status'], 'ready')
        self.assertEqual(decide_scope('What about dishwashers?', DOCS)['status'], 'clarification')
        self.assertEqual(decide_scope('Washer result?', DOCS)['document_ids'], ['washer'])

    def test_comparison_and_all_documents(self):
        self.assertEqual(decide_scope('Compare Washer and Imco', DOCS)['document_ids'], ['washer', 'imco'])
        self.assertEqual(decide_scope('Compare inspection results', DOCS)['status'], 'clarification')
        self.assertEqual(decide_scope('Compare Washer', DOCS)['status'], 'clarification')
        self.assertEqual(len(decide_scope('Summarize all documents', DOCS)['document_ids']), 3)
        self.assertEqual(len(decide_scope('Summarize results', DOCS, search_all=True)['document_ids']), 3)

    def test_selection_validation_and_duplicate_names(self):
        self.assertEqual(decide_scope('Result?', DOCS, ['imco'])['document_ids'], ['imco'])
        with self.assertRaises(ValueError):
            decide_scope('Result?', DOCS, ['deleted-id'])
        with self.assertRaises(ValueError):
            decide_scope('Result?', DOCS, ['imco'], True)
        duplicates = DOCS + [{'document_id': 'washer2', 'filename': DOCS[0]['filename']}]
        self.assertEqual(decide_scope('Washer result?', duplicates)['status'], 'clarification')

    def test_empty_greeting_single_document(self):
        self.assertEqual(decide_scope('Result?', [])['status'], 'no_documents')
        self.assertEqual(decide_scope('Hello!', DOCS)['status'], 'greeting')
        self.assertEqual(decide_scope('Hello, inspection results?', DOCS)['status'], 'clarification')
        self.assertEqual(decide_scope('Result?', DOCS[:1])['document_ids'], ['washer'])


class ScopeAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_contract_and_failures(self):
        with patch('ragsale.api.scope.list_documents', return_value=DOCS), patch(
            'ragsale.api.scope.run_in_threadpool', new=AsyncMock(return_value=DOCS)
        ) as worker:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                r = await client.post('/api/chat/scope', json={'question': 'Result?'})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()['status'], 'clarification')
                r = await client.post('/api/chat/scope', json={'question': 'Result?', 'selected_document_ids': ['imco']})
                self.assertEqual(r.json()['document_ids'], ['imco'])
                for body in [{'question': ' '}, {'question': 'Result?', 'selected_document_ids': ['missing']},
                             {'question': 'Result?', 'selected_document_ids': ['imco'], 'search_all': True}]:
                    self.assertEqual((await client.post('/api/chat/scope', json=body)).status_code, 422)
                worker.side_effect = RuntimeError('private details')
                with self.assertLogs('ragsale.api.scope', level='ERROR'):
                    r = await client.post('/api/chat/scope', json={'question': 'Result?'})
                self.assertEqual(r.status_code, 503)
                self.assertNotIn('private details', r.text)
