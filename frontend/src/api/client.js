// Relative URLs go through Vite's /api proxy during development.
export async function getHealth(signal) {
  const response = await fetch('/api/health', { signal });
  if (!response.ok) throw new Error(`Backend returned ${response.status}`);
  const data = await response.json();
  if (data.status !== 'ok') throw new Error('Unexpected health response');
  return data;
}
