# ocr.py
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# Try PDF-native text first
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

def _preprocess_pil(img: Image.Image) -> Image.Image:
    """
    Light, safe preprocessing that improves OCR without adding heavy deps.
    - grayscale
    - binary threshold (if Pillow-only), or OTSU if OpenCV available
    - mild denoise
    """
    try:
        import cv2, numpy as np  # optional
        arr = np.array(img.convert("L"))
        arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        arr = cv2.medianBlur(arr, 3)
        return Image.fromarray(arr)
    except Exception:
        # Pillow-only fallback
        return img.convert("L")

def _tesseract(img: Image.Image) -> str:
    # OEM 1 (LSTM), PSM 6 (block of text). Adjust if needed.
    config = "--oem 1 --psm 6 -l eng"
    return pytesseract.image_to_string(img, config=config)

def _ocr_images(images: List[Image.Image]) -> str:
    chunks: List[str] = []
    for im in images:
        prep = _preprocess_pil(im)
        chunks.append(_tesseract(prep))
    return "\n".join(chunks)

def extract_info_from_file(file_path: str) -> str:
    """
    Returns extracted text string, raising ValueError with a helpful message on failure.
    Strategy:
      1) If PDF and has selectable text -> return it.
      2) Else render PDF at 300 DPI and OCR.
      3) For images -> OCR with light preprocessing.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    text_data = ""

    # --- PDFs ---
    if ext == ".pdf":
        # Try native text first (fast & cheap)
        if PdfReader is not None:
            try:
                reader = PdfReader(str(path))
                native = []
                for page in reader.pages:
                    native.append(page.extract_text() or "")
                native_text = "\n".join(native).strip()
                if native_text:
                    return native_text
            except Exception:
                # Fall back to OCR below
                pass

        # OCR fallback
        try:
            images = convert_from_path(str(path), dpi=300)  # poppler required
            text_data = _ocr_images(images)
        except Exception as e:
            raise ValueError(f"PDF OCR failed: {e}")

    # --- Images ---
    elif ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        try:
            img = Image.open(str(path))
            text_data = _tesseract(_preprocess_pil(img))
        except Exception as e:
            raise ValueError(f"Image OCR failed: {e}")

    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image.")

    text_data = (text_data or "").strip()
    if not text_data:
        raise ValueError("No text could be extracted (empty after OCR).")

    return text_data