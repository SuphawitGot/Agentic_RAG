export async function searchDocuments(question, topK = 3) {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const message = typeof detail === 'string' ? detail
      : Array.isArray(detail) ? detail.map(item => item.msg).join('; ')
      : 'Search failed. Please try again.';
    throw new Error(message);
  }
  if (!Array.isArray(data?.matches)) throw new Error('Unexpected search response.');
  return data;
}
