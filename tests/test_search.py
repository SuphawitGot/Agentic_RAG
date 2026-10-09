import unittest
from unittest.mock import AsyncMock, patch
import httpx
from ragsale.api.main import app


class SearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_contract_and_validation(self):
        match = {'id': 'one', 'text': 'Sales increased.', 'metadata': {'filename': 'sales.pdf', 'page': 2}, 'distance': 0.2}
        with patch('ragsale.api.search.run_in_threadpool', new=AsyncMock(return_value=[match])) as worker:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                response = await client.post('/api/search', json={'question': ' sales? ', 'top_k': 3})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {'question': 'sales?', 'matches': [match]})
                self.assertEqual(worker.call_args.args[1:], ('sales?', 3))
                for body in [{}, {'question': ' '}, {'question': 'x' * 2001}, {'question': 'x', 'top_k': 0}, {'question': 'x', 'top_k': 11}]:
                    self.assertEqual((await client.post('/api/search', json=body)).status_code, 422)
                worker.return_value=[]
                self.assertEqual((await client.post('/api/search', json={'question':'sales'})).json()['matches'], [])
                worker.side_effect=RuntimeError('internal detail')
                response=await client.post('/api/search', json={'question':'sales'})
                self.assertEqual(response.status_code, 503)
                self.assertNotIn('internal detail',response.text)
