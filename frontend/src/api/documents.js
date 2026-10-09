export async function uploadDocument(file) {
  const formData = new FormData();
  // 'file' must match the FastAPI parameter name.
  formData.append('file', file);
  const response = await fetch('/api/documents/upload', {
    method: 'POST',
    body: formData,
    // The browser sets multipart/form-data and its boundary automatically.
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    if ([502, 503, 504].includes(response.status) && typeof data?.detail !== 'string') {
      throw new Error('Cannot reach the RagSale backend. Start FastAPI on port 8000, then retry the failed files.');
    }
    throw new Error(typeof data?.detail === 'string' ? data.detail : 'Upload failed. Please try again.');
  }
  if (data?.status !== 'ready') throw new Error('Unexpected upload response.');
  return data;
}

export async function uploadDocuments(files, onProgress = () => {}) {
  const results = [];
  // Wait for indexing to finish before starting the next file: OCR and embeddings
  // are expensive, and each response reports that document's final result.
  for (const [index, file] of files.entries()) {
    let result;
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      result = { status: 'error', error: 'Please select a PDF file.' };
    } else if (file.size === 0 || file.size > 50 * 1024 * 1024) {
      result = { status: 'error', error: 'Choose a non-empty PDF no larger than 50 MiB.' };
    } else {
      onProgress(index, { status: 'uploading' });
      try {
        result = { status: 'ready', result: await uploadDocument(file) };
      } catch (error) {
        result = { status: 'error', error: error instanceof TypeError
          ? 'Cannot reach the backend. Check that FastAPI is running.' : error.message };
      }
    }
    results.push(result);
    onProgress(index, result);
  }
  return results;
}
