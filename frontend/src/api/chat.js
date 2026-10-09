export async function askQuestion(question, topK = 3, scope = {}) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK, ...scope }),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const message = typeof detail === 'string' ? detail
      : Array.isArray(detail) ? detail.map(item => item.msg).join('; ')
      : 'Answer generation failed. Please try again.';
    throw new Error(message);
  }
  if (typeof data?.answer !== 'string' || !Array.isArray(data.sources)) throw new Error('Unexpected chat response.');
  if (data.type === 'clarification' && (!Array.isArray(data.project_options) || !data.project_options.length)) {
    throw new Error('No project choices were returned. Please ask again.');
  }
  return data;
}
