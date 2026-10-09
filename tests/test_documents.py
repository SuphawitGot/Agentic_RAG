import unittest
from unittest.mock import AsyncMock, patch

import httpx

from ragsale.api.main import app
from ragsale.rag.vector_store import list_documents


class DocumentListTests(unittest.TestCase):
    def test_groups_by_id_and_skips_legacy_metadata(self):
        with patch('ragsale.rag.vector_store.get_collection') as get_collection:
            collection = get_collection.return_value
            collection.get.return_value = {'metadatas': [
                {'document_id': 'b', 'filename': 'washer.pdf', 'page': 1},
                {'document_id': 'b', 'filename': 'washer.pdf', 'page': 2},
                {'document_id': 'a', 'filename': 'washer.pdf'},
                {'document_id': 'c', 'filename': 'Imco.pdf'},
                {'source': 'legacy dataset'}, None,
                {'document_id': ' ', 'filename': 'bad.pdf'},
            ]}
            self.assertEqual(list_documents(), [
                {'document_id': 'c', 'filename': 'Imco.pdf'},
                {'document_id': 'a', 'filename': 'washer.pdf'},
                {'document_id': 'b', 'filename': 'washer.pdf'},
            ])
            collection.get.assert_called_once_with(include=['metadatas'])

    def test_empty_and_conflicting_metadata(self):
        with patch('ragsale.rag.vector_store.get_collection') as get_collection:
            collection = get_collection.return_value
            collection.get.return_value = {'metadatas': []}
            self.assertEqual(list_documents(), [])
            collection.get.return_value = {'metadatas': [
                {'document_id': 'a', 'filename': 'one.pdf'},
                {'document_id': 'a', 'filename': 'two.pdf'},
            ]}
            with self.assertRaises(ValueError):
                list_documents()


class DocumentListAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_and_database_failure(self):
        documents = [{'document_id': 'a', 'filename': 'washer.pdf'}]
        with patch('ragsale.api.documents.run_in_threadpool',
                   new=AsyncMock(return_value=documents)) as worker:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url='http://test'
            ) as client:
                response = await client.get('/api/documents')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), documents)
                worker.return_value = []
                self.assertEqual((await client.get('/api/documents')).json(), [])
                worker.side_effect = RuntimeError('private database path')
                with self.assertLogs('ragsale.api.documents', level='ERROR'):
                    response = await client.get('/api/documents')
                self.assertEqual(response.status_code, 503)
                self.assertNotIn('private database path', response.text)
