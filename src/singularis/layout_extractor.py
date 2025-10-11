from __future__ import annotations
from pathlib import Path
from typing import List
import re
import fitz       # PyMuPDF
import camelot
from .models import TextBlock

# === 1. Текст и подписи ===
def extract_text_and_captions(pdf_path: Path) -> List[TextBlock]:
    blocks: List[TextBlock] = []
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            for b in page.get_text("blocks"):
                x0, y0, x1, y1 = b[:4]
                text = b[4].strip()
                if not text:
                    continue
                kind = "body"
                if re.match(r"^(Fig\.|Figure|Table)\s*\d+", text, re.IGNORECASE):
                    kind = "caption"
                blocks.append(
                    TextBlock(
                        page=page_idx,
                        bbox=(x0, y0, x1, y1),
                        text=text,
                        kind=kind,
                    )
                )
    blocks.sort(key=lambda b: (b.page, b.bbox[1], b.bbox[0]))
    return blocks


# === 2. Таблицы ===
def extract_tables(pdf_path: Path) -> List[TextBlock]:
    blocks: List[TextBlock] = []
    try:
        tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
    except Exception as e:
        print(f"⚠️  Camelot failed: {e}")
        return blocks

    for i, table in enumerate(tables):
        text = "\n".join(["\t".join(row) for row in table.df.values.tolist()])
        blocks.append(
            TextBlock(
                page=table.page,
                bbox=(0, 0, 0, 0),  # Camelot не даёт bbox напрямую
                text=text,
                kind="table",
            )
        )
    return blocks


# === 3. Изображения ===
def extract_images(pdf_path: Path) -> List[TextBlock]:
    blocks: List[TextBlock] = []
    pdf_path = Path(pdf_path)
    out_dir = pdf_path.parent / "figures"
    out_dir.mkdir(exist_ok=True)

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc, start=1):
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                fname = f"{pdf_path.stem}_p{page_idx}_img{img_idx+1}.png"
                fpath = out_dir / fname
                pix.save(fpath)
                blocks.append(
                    TextBlock(
                        page=page_idx,
                        bbox=(0, 0, 0, 0),
                        text=str(fpath),
                        kind="figure",
                    )
                )
                pix = None
    return blocks


# === 4. Главная функция ===
def extract_layout(pdf_path: str | Path) -> List[TextBlock]:
    pdf_path = Path(pdf_path)
    text_blocks = extract_text_and_captions(pdf_path)
    table_blocks = extract_tables(pdf_path)
    image_blocks = extract_images(pdf_path)
    return text_blocks + table_blocks + image_blocks
