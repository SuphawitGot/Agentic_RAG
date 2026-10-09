import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib import error

from ragsale.rag import loader, ocr
from ragsale.rag.chunking import split_documents


def ollama_response(body):
    return io.BytesIO(json.dumps(body).encode())


class OCRTests(unittest.TestCase):
    def test_page_ocr_sends_required_prompt_and_rendered_page(self):
        markdown = ("# บทที่ 3\n$F = ma$\n<figure>\nแผนภาพแรง\nบนวัตถุ\n</figure>\n"
                    "<page_number>14</page_number>")
        with patch.object(ocr, "_render_page", return_value="cGFnZQ==") as render, \
             patch.object(ocr.request, "urlopen",
                          return_value=ollama_response({"message": {"content": markdown}})) as urlopen:
            result = ocr.ocr_page("lesson.pdf", 3)
        render.assert_called_once_with("lesson.pdf", 3)
        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(payload["model"], ocr.DEFAULT_MODEL)
        self.assertEqual(payload["messages"], [{"role": "user", "content": ocr.PROMPT, "images": ["cGFnZQ=="]}])
        self.assertEqual(payload["options"]["temperature"], 0.1)
        self.assertEqual(result, "# บทที่ 3\n$F = ma$\n\n[Figure] แผนภาพแรง บนวัตถุ")

    def test_render_page_uses_trained_page_size(self):
        import base64
        import tempfile
        from pathlib import Path
        import pypdfium2 as pdfium
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a4.pdf"
            document = pdfium.PdfDocument.new()
            document.new_page(595, 842)
            document.save(path)
            document.close()
            encoded = ocr._render_page(path, 1)
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            self.assertEqual(image.size[1], ocr.PAGE_SIZE)

    def setUp(self):
        # These tests isolate OCR routing; real layout behavior has separate tests.
        self.pdf = patch('pdfplumber.open')
        self.layout = patch.object(loader, 'extract_layout', side_effect=lambda page, text: (text, {}))
        self.pdf.start()
        self.layout.start()
        self.addCleanup(self.pdf.stop)
        self.addCleanup(self.layout.stop)

    def test_merge_preserves_native_and_adds_screenshot_text(self):
        text, method = ocr.merge_page_text(
            "Inspection report\nLead time: 3 months",
            "INSPECTION REPORT\nLead time: 3 months\nAccuracy: 96.835%",
        )
        self.assertEqual(text.count("3 months"), 1)
        self.assertIn("Accuracy: 96.835%", text)
        self.assertEqual(method, "native_text+ocr")

    def test_merge_ignores_markdown_markup_around_native_text(self):
        text, method = ocr.merge_page_text(
            "Newton's second law\nF = ma",
            "## Newton's **second** law\n$F = ma$\n[Figure] แผนภาพแรง",
        )
        self.assertEqual(text, "Newton's second law\nF = ma\n\n[Additional OCR text]\n[Figure] แผนภาพแรง")
        self.assertEqual(method, "native_text+ocr")

    def test_short_numeric_cell_is_not_matched_inside_another_number(self):
        text, _ = ocr.merge_page_text("Invoice 1599", "59")
        self.assertTrue(text.endswith("\n59"))

    def test_scanned_and_mixed_pages_get_ocr_but_text_only_does_not(self):
        pages = [
            SimpleNamespace(extract_text=lambda: "Plain native text", images=[]),
            SimpleNamespace(extract_text=lambda: "", images=["scan"]),
            SimpleNamespace(extract_text=lambda: "Metrics", images=["screenshot"]),
            SimpleNamespace(extract_text=lambda: "", images=[]),
        ]
        reader = SimpleNamespace(is_encrypted=False, pages=pages)
        with patch("pypdf.PdfReader", return_value=reader), \
             patch.object(loader, "check_ocr_available") as check, \
             patch.object(loader, "ocr_page", side_effect=["ภาษาไทย 123", "Metrics\nAccuracy 96.835%", ""]) as render:
            docs = loader.load_pdf_documents("sample.pdf", "doc-1", "report.pdf")
        self.assertEqual([doc.metadata["page"] for doc in docs], [1, 2, 3])
        self.assertEqual([call.args for call in render.call_args_list],
                         [("sample.pdf", 2), ("sample.pdf", 3), ("sample.pdf", 4)])
        check.assert_called_once()
        self.assertEqual(docs[1].metadata["extraction_method"], "ocr")
        self.assertEqual(docs[2].metadata["extraction_method"], "native_text+ocr")
        self.assertEqual(docs[1].metadata["ocr_version"], ocr.OCR_VERSION)
        for chunk in split_documents(docs, chunk_size=30, chunk_overlap=5):
            self.assertEqual(chunk.metadata["document_id"], "doc-1")
            self.assertIn("extraction_method", chunk.metadata)

    def test_missing_model_is_an_explicit_setup_error(self):
        with patch.object(ocr.request, "urlopen",
                          return_value=ollama_response({"models": [{"name": "qwen3:8b"}]})):
            with self.assertRaisesRegex(ocr.OCRUnavailableError, "ollama pull scb10x/typhoon-ocr1.5-3b"):
                ocr.check_ocr_available()

    def test_pulled_model_with_latest_tag_is_available(self):
        with patch.object(ocr.request, "urlopen", return_value=ollama_response(
                {"models": [{"name": "scb10x/typhoon-ocr1.5-3b:latest"}]})):
            ocr.check_ocr_available()

    def test_stopped_ollama_is_an_explicit_setup_error(self):
        with patch.object(ocr.request, "urlopen", side_effect=error.URLError(ConnectionRefusedError())):
            with self.assertRaisesRegex(ocr.OCRUnavailableError, "cannot reach Ollama"):
                ocr.check_ocr_available()

    def test_timeout_is_actionable(self):
        with patch.object(ocr.request, "urlopen", side_effect=TimeoutError()):
            with self.assertRaisesRegex(ValueError, "timed out"):
                ocr._recognize("cGFnZQ==")

    def test_no_text_after_ocr_does_not_produce_empty_documents(self):
        page = SimpleNamespace(extract_text=lambda: "", images=[])
        with patch("pypdf.PdfReader", return_value=SimpleNamespace(is_encrypted=False, pages=[page])), \
             patch.object(loader, "check_ocr_available"), \
             patch.object(loader, "ocr_page", return_value=""):
            with self.assertRaisesRegex(ValueError, "No readable text"):
                loader.load_pdf_documents("photo.pdf", "doc", "photo.pdf")

    def test_text_only_pdf_does_not_require_ocr_installation(self):
        page = SimpleNamespace(extract_text=lambda: "Text only", images=[])
        with patch("pypdf.PdfReader", return_value=SimpleNamespace(is_encrypted=False, pages=[page])), \
             patch.object(loader, "check_ocr_available") as check:
            loader.load_pdf_documents("text.pdf", "doc", "text.pdf")
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
