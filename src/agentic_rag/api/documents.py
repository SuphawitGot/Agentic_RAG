"""Receive PDFs and index their text for retrieval."""

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from ..rag.pdf_ingestion import index_pdf
from ..rag.ocr import OCRUnavailableError
from ..rag.vector_store import list_documents
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])
UPLOAD_DIR = Path(__file__).resolve().parents[3] / "uploads"
MAX_UPLOAD_BYTES = 50  * 1024 * 1024


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    status: str
    text_pages: int
    chunks: int
    ocr_pages: int = 0


class DocumentSummary(BaseModel):
    document_id: str
    filename: str


@router.get("", response_model=list[DocumentSummary])
async def get_documents():
    """Return document choices from indexed chunk metadata."""
    try:
        return await run_in_threadpool(list_documents)
    except Exception as error:
        logger.exception("Could not list indexed documents")
        raise HTTPException(
            503, "Could not load indexed documents. Please try again."
        ) from error


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile):
    try:
        # Use the original name for display only, never as a storage path.
        filename = (file.filename or "").replace("\\", "/").split("/")[-1]
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(415, "Please select a PDF file.")
        if file.size is not None and file.size > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "PDF must be 50 MiB or smaller.")
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "PDF must be 50 MiB or smaller.")
        if not content:
            raise HTTPException(400, "The uploaded file is empty.")
        # Basic signature check, not full PDF parsing or text extraction.
        if not content.startswith(b"%PDF-"):
            raise HTTPException(415, "The file does not have a PDF header.")

        document_id = uuid4().hex
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        destination = UPLOAD_DIR / f"{document_id}.pdf"
        try:
            with destination.open("xb") as output:
                output.write(content)
        except OSError:
            destination.unlink(missing_ok=True)
            raise HTTPException(500, "Could not save the PDF. Please try again.")
        try:
            # CPU work runs in a worker thread, leaving the event loop free.
            counts = await run_in_threadpool(index_pdf, destination, document_id, filename)
        except OCRUnavailableError as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            logger.exception("PDF indexing failed for %s", document_id)
            destination.unlink(missing_ok=True)
            raise HTTPException(500, "PDF indexing failed. Please try again.") from error
        return UploadResponse(
            document_id=document_id,
            filename=filename,
            size_bytes=len(content),
            status="ready",
            **counts,
        )
    finally:
        await file.close()
