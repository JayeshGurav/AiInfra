import io
from pathlib import Path
from typing import Any

import fitz
import pdfplumber
import pytesseract
import structlog
from PIL import Image
from unstructured.partition.pdf import partition_pdf

logger = structlog.get_logger(__name__)


class PDFExtractor:
    def __init__(self, tesseract_cmd: str, image_dir: Path) -> None:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.image_dir = image_dir
        image_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, path: Path) -> list[dict[str, Any]]:
        self._extract_embedded_images(path)
        try:
            extracted = self._extract_unstructured(path)
        except Exception:
            logger.exception("unstructured_extraction_failed_using_pdfplumber", document=path.name)
            return self._extract_fallback(path)
        if not extracted:
            return self._extract_fallback(path)

        fallback = self._extract_fallback(path)
        pages = {page["page_number"]: page["text"] for page in extracted}
        for page in fallback:
            current = pages.get(page["page_number"], "")
            if len(current.strip()) < 20 and page["text"].strip():
                pages[page["page_number"]] = page["text"]
        return [
            {"page_number": page_number, "text": text}
            for page_number, text in sorted(pages.items())
            if text.strip()
        ]

    def _extract_embedded_images(self, path: Path) -> None:
        with fitz.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                    image = pdf.extract_image(image_info[0])
                    image_path = (
                        self.image_dir
                        / f"{path.stem}_page_{page_number}_image_{image_index}.{image['ext']}"
                    )
                    image_path.write_bytes(image["image"])

    def _extract_unstructured(self, path: Path) -> list[dict[str, Any]]:
        elements = partition_pdf(
            filename=str(path),
            strategy="hi_res",
            infer_table_structure=True,
            extract_images_in_pdf=True,
            extract_image_block_types=["Image", "Table"],
            extract_image_block_output_dir=str(self.image_dir),
        )
        pages: dict[int, list[str]] = {}
        for element in elements:
            metadata = element.metadata.to_dict()
            page_number = int(metadata.get("page_number") or 1)
            category = getattr(element, "category", "")
            text = str(element).strip()
            if category == "Table":
                table_html = metadata.get("text_as_html")
                text = f"[TABLE]\n{table_html or text}\n[/TABLE]"
            if text:
                pages.setdefault(page_number, []).append(text)
        if not pages:
            return self._extract_fallback(path)
        return [{"page_number": page, "text": "\n\n".join(items)} for page, items in sorted(pages.items())]

    def _extract_fallback(self, path: Path) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        with pdfplumber.open(path) as pdf:
            page_data = []
            for index, page in enumerate(pdf.pages, start=1):
                parts = [page.extract_text() or ""]
                for table in page.extract_tables() or []:
                    table_text = "\n".join(
                        " | ".join("" if cell is None else str(cell) for cell in row)
                        for row in table
                    )
                    if table_text.strip():
                        parts.append(f"[TABLE]\n{table_text}\n[/TABLE]")
                page_data.append((index, "\n\n".join(part for part in parts if part.strip())))

        with fitz.open(path) as pdf:
            for page_index, extracted_text in page_data:
                text = extracted_text
                if len(text.strip()) < 20:
                    page = pdf[page_index - 1]
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                    try:
                        ocr_text = pytesseract.image_to_string(image)
                    except pytesseract.TesseractNotFoundError as exc:
                        raise RuntimeError(
                            "Scanned PDF requires Tesseract OCR; install it and configure TESSERACT_CMD"
                        ) from exc
                    text = ocr_text.strip()
                output.append({"page_number": page_index, "text": text})
        return [page for page in output if page["text"].strip()]
