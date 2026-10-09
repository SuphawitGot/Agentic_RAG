# React frontend

React + Vite using JavaScript and JSX.

```text
src/
  api/client.js              # Backend HTTP requests
  components/ChatInput.jsx   # Question form
  components/MessageList.jsx # Conversation display
  components/SourceList.jsx  # Source references
  pages/ChatPage.jsx         # Chat layout and backend health check
  App.jsx                    # Root component
  main.jsx                   # Browser entry point
  styles.css                 # Shared styles
public/                      # Static assets
index.html                   # HTML entry point
vite.config.js               # React plugin and /api development proxy
package.json                 # Frontend dependencies and commands
```

## Development

Use Node.js 22.12+ (or 20.19+) and pnpm. From this folder:

```bash
pnpm install
pnpm dev
```

In another terminal, from the project root:

```bash
uv run uvicorn ragsale.api.main:app --reload
```

Visit http://127.0.0.1:5173. Vite forwards `/api` requests to
http://127.0.0.1:8000, so the health check works without adding CORS settings.
The interface displays connection status and lets you retry the check.

The PDF upload form calls `POST /api/documents/upload` and saves files on the backend.
The question form sends JSON to `/api/chat` and displays Qwen answers with sources.
Do not show fabricated answers while backend integration is unfinished.

## Production build

```bash
pnpm build
pnpm preview
```

The build is written to `dist/`. The development proxy is not a production
backend configuration. Configure your deployed web server to route `/api` to
FastAPI. Keep Qwen, embeddings, Chroma, and model credentials in the backend.

## PDF upload

`src/components/FileUpload.jsx` selects a PDF and shows status.
`src/api/documents.js` sends multipart FormData with a field named `file`.
The backend stores PDFs in `uploads/<document_id>.pdf` and returns a 201 JSON response.
Files must be non-empty, no larger than 50 MiB, have a .pdf extension, and begin
with a PDF header. PDF parsing and text extraction follow these checks. Readable uploaded PDFs are indexed before success is returned; scanned PDFs need OCR. The size check runs after multipart parsing; a deployed
server should also enforce request limits at its proxy.

PDF uploads now return `status: ready`, `text_pages`, and `chunks` after indexing.
The request stays open while processing. No-text, malformed, or encrypted PDFs
return an error. A failed upload removes its saved file; failed Chroma writes
attempt cleanup by document ID. Each new upload receives a new ID, so uploading
the same file again creates another document. Large-scale processing will need
a job queue; this version processes one request through completion.

`src/api/search.js` sends questions; `ChatPage.jsx` manages search state;
`SourceList.jsx` displays excerpts, filenames, page numbers, and distances.
Results are retrieval excerpts, not generated answers.

## Answer generation

The question form now uses `src/api/chat.js` to call `/api/chat`.
It displays Qwen's answer with numbered excerpts beneath it. Ollama/model
availability errors appear with a retry button. Raw search remains available
through the backend `/api/search` endpoint. Start Ollama before asking questions.
