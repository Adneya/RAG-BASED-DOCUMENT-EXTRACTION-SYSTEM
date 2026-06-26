from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import UploadFile

from app.config import settings
from app.schemas import ExtractedContent
from app.tools.audio import audio_duration_seconds, transcribe_audio
from app.tools.ocr import ocr_image_bytes
from app.tools.text_processing import clean_text


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PDF_EXTENSIONS = {".pdf"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".xml", ".html", ".py", ".js", ".ts"}


async def extract_upload(upload: UploadFile) -> ExtractedContent:
    data = await upload.read()
    filename = Path(upload.filename or "upload").name
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        return ExtractedContent(
            filename=filename,
            source_type="unknown",
            warnings=[f"File exceeds the {settings.max_upload_mb} MB upload limit."],
        )
    source_type = classify_file(filename, upload.content_type or "")

    if source_type == "image":
        text, confidence, warnings = ocr_image_bytes(data)
        return ExtractedContent(
            filename=filename,
            source_type="image",
            text=text,
            ocr_confidence=confidence,
            warnings=warnings,
        )
    if source_type == "pdf":
        text, confidence, warnings = extract_pdf(data)
        return ExtractedContent(
            filename=filename,
            source_type="pdf",
            text=text,
            ocr_confidence=confidence,
            warnings=warnings,
        )
    if source_type == "audio":
        text, warnings = transcribe_audio(data, filename)
        return ExtractedContent(
            filename=filename,
            source_type="audio",
            text=text,
            duration_seconds=audio_duration_seconds(data, filename),
            warnings=warnings,
        )
    if source_type == "text":
        return ExtractedContent(
            filename=filename,
            source_type="text",
            text=decode_text(data),
        )

    return ExtractedContent(
        filename=filename,
        source_type="unknown",
        warnings=["Unsupported file type. Supported types are text, JPG/PNG, PDF, MP3, WAV, and M4A."],
    )


def classify_file(filename: str, content_type: str = "") -> str:
    suffix = Path(filename).suffix.lower()
    content_type = (content_type or "").lower()
    if suffix in IMAGE_EXTENSIONS or content_type.startswith("image/"):
        return "image"
    if suffix in PDF_EXTENSIONS or content_type == "application/pdf":
        return "pdf"
    if suffix in AUDIO_EXTENSIONS or content_type.startswith("audio/"):
        return "audio"
    if suffix in TEXT_EXTENSIONS or content_type.startswith("text/"):
        return "text"
    return "unknown"


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return clean_text(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return clean_text(data.decode("utf-8", errors="ignore"))


def extract_pdf(data: bytes) -> tuple[str, float | None, list[str]]:
    warnings: list[str] = []
    pages: list[str] = []
    ocr_confidences: list[float] = []

    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover - dependency should be present
        return "", None, [f"PDF parser is unavailable: {exc}"]

    try:
        with pdfplumber.open(BytesIO(data)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = clean_text(page.extract_text() or "")
                if text:
                    pages.append(f"[Page {index}]\n{text}")
                    continue

                ocr_text, confidence, ocr_warnings = ocr_pdf_page(data, index - 1)
                warnings.extend(f"Page {index}: {warning}" for warning in ocr_warnings)
                if ocr_text:
                    pages.append(f"[Page {index} OCR]\n{ocr_text}")
                if confidence is not None:
                    ocr_confidences.append(confidence)
    except Exception as exc:
        return "", None, [f"PDF extraction failed: {exc}"]

    confidence_score = (
        round(sum(ocr_confidences) / len(ocr_confidences), 3)
        if ocr_confidences
        else None
    )
    if not pages:
        warnings.append("No extractable text was found in the PDF.")
    return clean_text("\n\n".join(pages)), confidence_score, warnings


def ocr_pdf_page(data: bytes, page_index: int) -> tuple[str, float | None, list[str]]:
    try:
        import fitz
    except Exception as exc:  # pragma: no cover - optional dependency
        return "", None, [f"PDF OCR rendering dependency is unavailable: {exc}"]

    document = None
    try:
        document = fitz.open(stream=data, filetype="pdf")
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        png_bytes = pixmap.tobytes("png")
        return ocr_image_bytes(png_bytes)
    except Exception as exc:  # pragma: no cover - depends on PDF shape
        return "", None, [f"PDF page OCR failed: {exc}"]
    finally:
        if document is not None:
            document.close()
