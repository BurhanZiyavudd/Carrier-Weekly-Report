from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pytesseract
from PIL import Image, ImageOps
from pdf2image import convert_from_path

# Optional fast PDF text & layout
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# Optional OpenCV
try:
    import cv2, numpy as np
except Exception:
    cv2 = None
    np = None


@dataclass
class PageNote:
    index: int
    source: str              # "pdf_native" | "ocr"
    warning: Optional[str] = None


@dataclass
class ExtractResult:
    text: str
    source: str              # "pdf_native", "pdf_mixed", "image_ocr"
    pages: List[PageNote] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _preprocess_pil(img: Image.Image) -> Image.Image:
    """Robust but light preprocessing (grayscale → binarize → denoise → deskew)."""
    img = img.convert("L")
    if cv2 is None:
        # Pillow-only fallback
        return ImageOps.autocontrast(img)
    arr = np.array(img)
    # OTSU binarization
    _, arr = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Mild denoise
    arr = cv2.medianBlur(arr, 3)
    # Deskew via Hough transform (fast, safe)
    coords = np.column_stack(np.where(arr < 128))
    if coords.size > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 0.5:
            (h, w) = arr.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(arr)


def _ocr_with_osd(img: Image.Image, lang: str) -> Tuple[str, Optional[str]]:
    """OCR with orientation detection; returns (text, warning)."""
    warning = None
    try:
        osd = pytesseract.image_to_osd(img)
        # crude parse
        for line in str(osd).splitlines():
            if "Rotate:" in line:
                deg = int(line.split(":")[1].strip())
                if deg:
                    img = img.rotate(-deg, expand=True)
                break
    except Exception:
        warning = "Orientation detection failed; OCR without rotation."
    config = f"--oem 1 --psm 6 -l {lang}"
    return pytesseract.image_to_string(_preprocess_pil(img), config=config), warning


def _pdf_native_text(path: Path) -> Optional[List[str]]:
    """Return list of per-page native text if available, else None."""
    # Prefer PyMuPDF for reliability; fall back to pypdf if needed
    if fitz is not None:
        try:
            doc = fitz.open(str(path))
            if doc.is_encrypted:
                # Try to open without password; if fails, error out
                try:
                    doc.authenticate("")  # no-op but triggers error if needed
                except Exception:
                    raise ValueError("PDF appears to be password-protected.")
            pages = []
            for p in doc:
                txt = p.get_text("text") or ""
                pages.append(txt)
            doc.close()
            if any(s.strip() for s in pages):
                return pages
            return None
        except ValueError:
            raise
        except Exception:
            return None
    else:
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = [(pg.extract_text() or "") for pg in reader.pages]
            if any(s.strip() for s in pages):
                return pages
            return None
        except Exception:
            return None


def extract_info_from_file(
    file_path: str,
    *,
    lang: str = "eng",
    poppler_path: Optional[str] = None,
    dpi: int = 300,
    max_pages: int = 200,
) -> ExtractResult:
    """
    Robust text extraction with structured result.
    Strategy:
      • PDF: try native text per page; OCR only pages with no text.
      • Image: OCR with OSD/deskew.
    Raises ValueError with human-friendly messages.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    notes: List[PageNote] = []
    warnings: List[str] = []

    # ---- PDFs ----
    if ext == ".pdf":
        # 1) Native text per page
        native_pages = None
        try:
            native_pages = _pdf_native_text(path)
        except ValueError as e:
            raise ValueError(str(e))  # encrypted, etc.

        if native_pages is not None:
            # Some pages may still be empty → OCR them only
            need_ocr = [i for i, t in enumerate(native_pages) if not (t or "").strip()]
            text_out = list(native_pages)
            if need_ocr:
                if len(native_pages) > max_pages:
                    warnings.append(f"PDF has {len(native_pages)} pages; capped OCR to first {max_pages}.")
                ocr_targets = [i for i in need_ocr if i < max_pages]
                images = convert_from_path(
                    str(path), dpi=dpi, first_page=min(ocr_targets, default=0)+1,
                    last_page=max(ocr_targets, default=0)+1, poppler_path=poppler_path
                ) if ocr_targets else []
                # Map rendered list to absolute page indexes
                base = min(ocr_targets, default=0)
                for idx, img in zip(ocr_targets, images):
                    txt, warn = _ocr_with_osd(img, lang)
                    if warn: warnings.append(f"p{idx+1}: {warn}")
                    text_out[idx] = txt
                    notes.append(PageNote(index=idx, source="ocr"))
            # Mark native pages
            for i, t in enumerate(text_out):
                if i not in [n.index for n in notes]:
                    notes.append(PageNote(index=i, source="pdf_native"))
            return ExtractResult(
                text="\n".join(text_out).strip(),
                source=("pdf_mixed" if need_ocr else "pdf_native"),
                pages=sorted(notes, key=lambda n: n.index),
                warnings=warnings,
            )

        # 2) No native text at all → full OCR
        try:
            images = convert_from_path(str(path), dpi=dpi, poppler_path=poppler_path)
            if len(images) > max_pages:
                warnings.append(f"PDF has {len(images)} pages; OCR capped at {max_pages}.")
                images = images[:max_pages]
        except Exception as e:
            raise ValueError(f"PDF rendering for OCR failed: {e}")

        out_chunks: List[str] = []
        for i, im in enumerate(images):
            txt, warn = _ocr_with_osd(im, lang)
            if warn: warnings.append(f"p{i+1}: {warn}")
            out_chunks.append(txt)
            notes.append(PageNote(index=i, source="ocr"))
        final = "\n".join(out_chunks).strip()
        if not final:
            raise ValueError("No text could be extracted from this PDF (even after OCR).")
        return ExtractResult(text=final, source="pdf_mixed", pages=notes, warnings=warnings)

    # ---- Images ----
    elif ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
        try:
            img = Image.open(str(path))
        except Exception as e:
            raise ValueError(f"Could not open image: {e}")
        txt, warn = _ocr_with_osd(img, lang)
        if not (txt or "").strip():
            raise ValueError("No text could be extracted from this image.")
        if warn: warnings.append(warn)
        return ExtractResult(text=txt.strip(), source="image_ocr", pages=[PageNote(index=0, source="ocr")], warnings=warnings)

    else:
        raise ValueError("Unsupported file type. Please upload a PDF or image.")