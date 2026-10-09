import unittest
from unittest.mock import AsyncMock, patch
import httpx
from ragsale.api.main import app
from ragsale.rag.answering import GenerationError


class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_and_failure(self):
        result={'answer':'Not enough information.','sources':[],'insufficient_evidence':True}
        with patch('ragsale.api.chat.run_in_threadpool', new=AsyncMock(return_value=result)) as worker:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                r=await client.post('/api/chat',json={'question':'What happened?'})
                self.assertEqual(r.status_code,200)
                self.assertEqual(r.json()['answer'], result['answer'])
                self.assertEqual(r.json()['type'], 'answer')
                self.assertEqual(r.json()['project_options'], [])
                self.assertEqual((await client.post('/api/chat',json={'question':' '})).status_code,422)
                worker.side_effect=GenerationError('Ollama unavailable')
                self.assertEqual((await client.post('/api/chat',json={'question':'What?'})).status_code,503)
