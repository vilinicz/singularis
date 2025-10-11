from __future__ import annotations
from pathlib import Path
from typing import List
import fitz  # PyMuPDF
from .models import TextBlock

def extract_text_blocks(pdf_path: str | Path) -> List[TextBlock]:
    """Извлекает текстовые блоки из PDF со страниц и координатами."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    blocks: List[TextBlock] = []
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            for b in page.get_text("blocks"):
                x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
                text = b[4] if len(b) > 4 else ""
                if not text.strip():
                    continue
                blocks.append(
                    TextBlock(
                        page=page_idx,
                        bbox=(float(x0), float(y0), float(x1), float(y1)),
                        text=text.strip(),
                        kind="body",
                    )
                )
    # сортировка: страница → сверху вниз → слева направо
    blocks.sort(key=lambda b: (b.page, b.bbox[1], b.bbox[0]))
    return blocks