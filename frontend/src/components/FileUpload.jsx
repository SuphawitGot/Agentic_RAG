import { useRef, useState } from 'react';
import { uploadDocuments } from '../api/documents';

export default function FileUpload() {
  const input = useRef(null);
  const busy = useRef(false);
  const [files, setFiles] = useState([]);
  const [rows, setRows] = useState([]);
  const [uploading, setUploading] = useState(false);
  const completed = rows.filter(row => row.status === 'ready' || row.status === 'error').length;
  const succeeded = rows.filter(row => row.status === 'ready').length;
  const failed = rows.filter(row => row.status === 'error').length;

  async function submit(event) {
    event.preventDefault();
    if (!files.length || busy.current) return;
    busy.current = true;
    setUploading(true);
    try {
      await uploadDocuments(files, (index, update) => {
        setRows(current => current.map((row, i) => i === index ? { ...row, ...update } : row));
      });
      setFiles([]);
      input.current.value = '';
    } finally {
      busy.current = false;
      setUploading(false);
    }
  }

  return <section className="panel upload-panel" aria-labelledby="upload-heading">
    <h2 id="upload-heading">Upload documents</h2>
    <p id="upload-help">Select multiple PDFs. Each file can be up to 50 MiB and 200 pages. Files are processed one at a time with extraction, OCR when needed, embedding and storage. Keep this page open until the batch finishes.</p>
    <form onSubmit={submit} aria-busy={uploading}>
      <label htmlFor="pdf-file">PDF files</label>
      <input ref={input} id="pdf-file" type="file" accept=".pdf,application/pdf" multiple
        aria-describedby="upload-help" disabled={uploading}
        onChange={event => {
          const selected = Array.from(event.target.files ?? []);
          setFiles(selected);
          setRows(selected.map(file => ({ name: file.name, status: 'queued' })));
        }} />
      <div><button type="submit" disabled={!files.length || uploading}>
        {uploading ? `Processing ${Math.min(completed + 1, rows.length)} of ${rows.length}…` : `Upload ${files.length || ''} PDF${files.length === 1 ? '' : 's'}`}
      </button></div>
    </form>
    {rows.length > 0 && <>
      <p role="status">{completed} of {rows.length} finished · {succeeded} ready · {failed} failed</p>
      <progress value={completed} max={rows.length} aria-label="Files processed" />
      <ul className="upload-results">
        {rows.map((row, index) => <li key={index}>
          <strong>{row.name}</strong> — {row.status === 'queued' ? 'Waiting' : row.status === 'uploading' ? 'Uploading and preparing…' : row.status === 'ready' ? 'Ready for search' : 'Failed'}
          {row.error && <p className="upload-error">{row.error}</p>}
          {row.result && <p>{row.result.chunks} chunks · {row.result.text_pages} pages with text · {row.result.ocr_pages ?? 0} OCR pages</p>}
        </li>)}
      </ul>
      {!uploading && failed > 0 && <p>Other files were processed independently. Fix the failed files and select only those to try again.</p>}
    </>}
  </section>;
}
