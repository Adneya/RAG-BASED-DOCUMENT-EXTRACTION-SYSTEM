from __future__ import annotations

from io import BytesIO

from .text_processing import clean_text


def ocr_image_bytes(data: bytes) -> tuple[str, float | None, list[str]]:
    warnings: list[str] = []
    try:
        from PIL import Image
        import pytesseract
        from pytesseract import Output
    except Exception as exc:  # pragma: no cover - depends on optional install
        return "", None, [f"OCR dependencies are unavailable: {exc}"]

    try:
        image = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:
        return "", None, [f"Image could not be opened for OCR: {exc}"]

    try:
        data_dict = pytesseract.image_to_data(image, output_type=Output.DICT)
        words: list[str] = []
        confidences: list[float] = []
        for text, conf in zip(data_dict.get("text", []), data_dict.get("conf", [])):
            cleaned = (text or "").strip()
            try:
                confidence = float(conf)
            except (TypeError, ValueError):
                confidence = -1.0
            if cleaned:
                words.append(cleaned)
            if confidence >= 0:
                confidences.append(confidence)
        transcript = clean_text(" ".join(words))
        confidence_score = (
            round(sum(confidences) / len(confidences) / 100, 3)
            if confidences
            else None
        )
        if not transcript:
            transcript = clean_text(pytesseract.image_to_string(image))
        return transcript, confidence_score, warnings
    except Exception as exc:  # pragma: no cover - depends on local binary
        warnings.append(f"OCR failed: {exc}")
        return "", None, warnings

