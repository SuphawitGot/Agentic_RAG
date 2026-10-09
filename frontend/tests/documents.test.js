import test from 'node:test';
import assert from 'node:assert/strict';
import { uploadDocument, uploadDocuments } from '../src/api/documents.js';

test('unavailable development proxy gives an actionable backend error', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response('', { status: 502 }));
  await assert.rejects(uploadDocument(new File(['pdf'], 'one.pdf')), /Start FastAPI on port 8000/);
});

test('backend service errors retain their specific explanation', async t => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ detail: 'Install OCR language packs' }), { status: 503 }));
  await assert.rejects(uploadDocument(new File(['pdf'], 'one.pdf')), /Install OCR language packs/);
});

test('batch waits for each response and continues after a failed PDF', async t => {
  const files = ['one.pdf', 'broken.pdf', 'three.pdf'].map(name => new File(['%PDF-fixture'], name));
  const sent = [];
  let active = 0;
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url, '/api/documents/upload');
    assert.equal(++active, 1, 'uploads must be sequential');
    const name = options.body.get('file').name;
    sent.push(name);
    await new Promise(resolve => setTimeout(resolve, 5));
    active--;
    return name === 'broken.pdf'
      ? new Response(JSON.stringify({ detail: 'PDF could not be read.' }), { status: 422 })
      : new Response(JSON.stringify({ status: 'ready', filename: name, chunks: 2 }));
  });
  const updates = [];
  const result = await uploadDocuments(files, (index, update) => updates.push([index, update.status]));
  assert.deepEqual(sent, files.map(f => f.name));
  assert.deepEqual(result.map(r => r.status), ['ready', 'error', 'ready']);
  assert.equal(result[1].error, 'PDF could not be read.');
  assert.deepEqual(updates, [[0, 'uploading'], [0, 'ready'], [1, 'uploading'], [1, 'error'], [2, 'uploading'], [2, 'ready']]);
});

test('invalid files do not upload; a later valid file still uploads', async t => {
  const fetch = t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ status: 'ready' })));
  const files = [new File(['text'], 'bad.txt'), new File([], 'empty.pdf'),
    { name: 'large.pdf', size: 50 * 1024 * 1024 + 1 }, new File(['%PDF'], 'valid.PDF')];
  const result = await uploadDocuments(files);
  assert.deepEqual(result.map(r => r.status), ['error', 'error', 'error', 'ready']);
  assert.equal(fetch.mock.callCount(), 1);
});

test('network failure is reported without abandoning the remaining files', async t => {
  let count = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    if (++count === 1) throw new TypeError('Failed to fetch');
    return new Response(JSON.stringify({ status: 'ready' }));
  });
  const result = await uploadDocuments([new File(['pdf'], 'one.pdf'), new File(['pdf'], 'two.pdf')]);
  assert.match(result[0].error, /Cannot reach the backend/);
  assert.equal(result[1].status, 'ready');
});
