"""Load dataset rows or PDF pages (including local OCR) as documents."""

from datasets import load_dataset
from langchain_core.documents import Document
from .layout import extract_layout
from .ocr import OCR_VERSION, check_ocr_available, merge_page_text, ocr_page


def load_documents(dataset_id="mtybilly/apex-r1-real-world-documents", split="train"):
    dataset = load_dataset(dataset_id, split=split)
    # Each row is text, not necessarily a complete original document.
    return [
        Document(
            page_content=row["text"],
            metadata={
                "source": f"https://huggingface.co/datasets/{dataset_id}",
                "split": split,
                "row_index": index,
            },
        )
        for index, row in enumerate(dataset)
        if row["text"] and row["text"].strip()
    ]


def load_pdf_documents(path, document_id, filename):
    """Preserve native text and supplement scanned/mixed pages with OCR."""
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")
        if len(reader.pages) > 200:
            raise ValueError("Please upload PDFs with 200 pages or fewer for OCR processing.")
        import pdfplumber

        documents = []
        ocr_checked = False
        with pdfplumber.open(path) as layout_pdf:
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text, layout_metadata = extract_layout(layout_pdf.pages[page_number - 1], text)
                method = "native_text"
                # A heading does not mean screenshot/diagram text was extracted.
                images = page.images
                needs_ocr = not text.strip() or bool(images)
                if needs_ocr:
                    if not ocr_checked:
                        check_ocr_available()
                        ocr_checked = True
                    # The whole page is read, so images keep their surrounding context.
                    recognized = ocr_page(path, page_number)
                    text, method = merge_page_text(text, recognized)
                if text.strip():
                    documents.append(Document(page_content=text, metadata={
                        **layout_metadata,
                        "document_id": document_id,
                        "source": filename,
                        "filename": filename,
                        "page": page_number,
                        "extraction_method": method,
                        "ocr_attempted": needs_ocr,
                        "ocr_version": OCR_VERSION if needs_ocr else "none",
                    }))
    except PyPdfError as error:
        raise ValueError("Could not read this PDF. Please upload a valid PDF.") from error
    if not documents:
        raise ValueError("No readable text found, even after OCR. Photos without text need vision descriptions.")
    return documents
