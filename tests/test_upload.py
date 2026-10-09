import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

import httpx

from ragsale.api.main import app
from ragsale.api import documents


class UploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with patch.object(documents, 'UPLOAD_DIR', directory), patch.object(documents, 'run_in_threadpool', new=AsyncMock(return_value={'text_pages': 1, 'chunks': 1})):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                    # Header-shaped bytes test receipt only, not PDF parsing.
                    content = b'%PDF-1.4\nfixture'
                    response = await client.post('/api/documents/upload', files={'file': ('../../sales.pdf', content, 'application/pdf')})
                    self.assertEqual(response.status_code, 201, response.text)
                    data = response.json()
                    self.assertEqual(data['filename'], 'sales.pdf')
                    self.assertEqual(data['status'], 'ready')
                    self.assertEqual((directory / (data['document_id'] + '.pdf')).read_bytes(), content)
                    for name, body, expected in [('x.txt', b'hello', 415), ('x.pdf', b'hello', 415), ('x.pdf', b'', 400)]:
                        response = await client.post('/api/documents/upload', files={'file': (name, body)})
                        self.assertEqual(response.status_code, expected)
                    with patch.object(documents, 'MAX_UPLOAD_BYTES', 5):
                        response = await client.post('/api/documents/upload', files={'file': ('x.pdf', content)})
                        self.assertEqual(response.status_code, 413)
                    response = await client.post('/api/documents/upload')
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(len(list(directory.iterdir())), 1)

    async def test_missing_ocr_returns_setup_error_and_removes_uploaded_file(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(documents, 'UPLOAD_DIR', Path(folder)), patch.object(
                documents, 'run_in_threadpool',
                new=AsyncMock(side_effect=documents.OCRUnavailableError('Install OCR language packs')),
            ):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                    response = await client.post('/api/documents/upload', files={'file': ('scan.pdf', b'%PDF-1.4 fixture')})
                self.assertEqual(response.status_code, 503)
                self.assertIn('language packs', response.json()['detail'])
                self.assertEqual(list(Path(folder).iterdir()), [])
