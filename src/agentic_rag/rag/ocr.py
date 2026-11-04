"""Local English/Thai OCR with Typhoon OCR served by Ollama.

Pages are rendered in-process and sent to the Ollama server in OLLAMA_BASE_URL
(this machine by default), so no document content leaves the machine.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path
import re
from threading import Semaphore
import unicodedata
from urllib import error, request


_OCR_SLOT = Semaphore(1)
OCR_VERSION = "typhoon-ocr1.5-page-v1"
DEFAULT_MODEL = "scb10x/typhoon-ocr1.5-3b"
# Typhoon OCR v1.5 was trained on page images whose longest side is 1,800 px.
PAGE_SIZE = 1800
TIMEOUT_SECONDS = 600
# The model card states the model only works with this exact prompt.
PROMPT = """Extract all text from the image.

Instructions:
- Only return the clean Markdown.
- Do not include any explanation or extra text.
- You must include all information on the page.

Formatting Rules:
- Tables: Render tables using <table>...</table> in clean HTML format.
- Equations: Render equations using LaTeX syntax with inline ($...$) and block ($$...$$).
- Images/Charts/Diagrams: Wrap any clearly defined visual areas (e.g. charts, diagrams, pictures) in:

<figure>
Describe the image's main elements (people, objects, text), note any contextual clues (place, event, culture), mention visible text and its meaning, provide deeper analysis when relevant (especially for financial charts, graphs, or documents), comment on style or architecture if relevant, then give a concise overall summary. Describe in Thai.
</figure>

- Page Numbers: Wrap page numbers in <page_number>...</page_number> (e.g., <page_number>14</page_number>).
- Checkboxes: Use ☐ for unchecked and ☑ for checked boxes."""


class OCRUnavailableError(RuntimeError):
    """The server needs Ollama running with the Typhoon OCR model pulled."""


def _model():
    return os.getenv("TYPHOON_OCR_MODEL", DEFAULT_MODEL)


def _base_url():
    return os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def check_ocr_available():
    model = _model()
    try:
        with request.urlopen(_base_url() + "/api/tags", timeout=10) as response:
            names = {item["name"] for item in json.load(response)["models"]}
    except (error.URLError, OSError, ValueError, KeyError, TypeError) as exc:
        raise OCRUnavailableError(
            "OCR is unavailable: cannot reach Ollama. Start Ollama and run "
            f"'ollama pull {model}' (see README)."
        ) from exc
    if model not in names and f"{model}:latest" not in names:
        raise OCRUnavailableError(f"The OCR model is missing. Run 'ollama pull {model}' on the backend.")


def _render_page(path, page_number):
    """Render one page as a base64 PNG with the longest side at PAGE_SIZE."""
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        raise ValueError("Could not render this PDF page for OCR.") from exc
    try:
        page = document[page_number - 1]
        width, height = page.get_size()
        image = page.render(scale=PAGE_SIZE / max(width, height)).to_pil().convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    except pdfium.PdfiumError as exc:
        raise ValueError("Could not render this PDF page for OCR.") from exc
    finally:
        document.close()


def _recognize(image):
    payload = {
        "model": _model(),
        "stream": False,
        "messages": [{"role": "user", "content": PROMPT, "images": [image]}],
        # Model card settings. num_predict bounds a runaway repetition loop.
        "options": {"temperature": 0.1, "top_p": 0.6, "repeat_penalty": 1.1,
                    "num_ctx": 16384, "num_predict": 8192},
    }
    req = request.Request(_base_url() + "/api/chat", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
            body = json.load(response)
    except error.HTTPError as exc:
        if exc.code == 404:
            raise OCRUnavailableError(f"The OCR model is missing. Run 'ollama pull {_model()}' on the backend.") from exc
        raise ValueError("PDF OCR failed. Check the Ollama server logs.") from exc
    except TimeoutError as exc:
        raise ValueError("PDF OCR timed out. Try a smaller PDF or fewer pages.") from exc
    except error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise ValueError("PDF OCR timed out. Try a smaller PDF or fewer pages.") from exc
        raise OCRUnavailableError("OCR is unavailable: cannot reach Ollama. Check that it is running.") from exc
    except OSError as exc:
        raise OCRUnavailableError("OCR is unavailable: the Ollama connection failed. Check that it is running.") from exc
    try:
        return body["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise ValueError("PDF OCR returned an invalid response. Please retry.") from exc


def _clean(markdown):
    """Drop page numbers; keep each figure description on one line for merging."""
    text = re.sub(r"<page_number>.*?</page_number>", "", markdown, flags=re.S)

    def figure(match):
        description = " ".join(match.group(1).split())
        return f"\n[Figure] {description}\n" if description else "\n"

    text = re.sub(r"<figure>(.*?)</figure>", figure, text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def ocr_page(path, page_number):
    """Read a whole page: text, tables, equations and figure descriptions."""
    # One page at a time; Ollama runs one generation per model anyway.
    with _OCR_SLOT:
        return _clean(_recognize(_render_page(path, page_number)))


# Markdown/HTML/LaTeX markup Typhoon adds around text that native extraction lacks.
_MARKUP = re.compile(r"</?[A-Za-z][^>]*>|[#*_`|$]")


def _normalized(text):
    return "".join(_MARKUP.sub("", unicodedata.normalize("NFKC", text)).casefold().split())


def merge_page_text(native_text, ocr_text):
    """Keep native text; add OCR lines not already present (best-effort dedup)."""
    native_text = native_text.strip()
    if not native_text:
        return ocr_text.strip(), "ocr"
    native_normalized = _normalized(native_text)
    known_lines = {_normalized(line) for line in native_text.splitlines()}
    additions = []
    for line in ocr_text.splitlines():
        key = _normalized(line)
        # Short numeric cells must not disappear just because their digits
        # happen to occur somewhere else in the native page text.
        already_present = key in known_lines or (len(key) >= 20 and key in native_normalized)
        if key and not already_present:
            additions.append(line.strip())
            known_lines.add(key)
    if not additions:
        return native_text, "native_text"
    return native_text + "\n\n[Additional OCR text]\n" + "\n".join(additions), "native_text+ocr"


def main():
    parser = argparse.ArgumentParser(description="Preview PDF extraction without embedding or writing to Chroma.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .loader import load_pdf_documents

    documents = load_pdf_documents(args.pdf, "preview", args.pdf.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([
        {"text": document.page_content, "metadata": document.metadata}
        for document in documents
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(documents)} pages with text to {args.output}")


if __name__ == "__main__":
    main()
