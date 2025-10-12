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
    total_pages = 0

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        print(f"📘 Parsing PDF: {pdf_path.name}")
        for page_idx, page in enumerate(doc, start=1):
            raw_blocks = page.get_text("blocks")
            if not raw_blocks:
                print(f"⚠️  Page {page_idx}: no text blocks found")
                continue

            xs = [b[0] for b in raw_blocks]
            median_x = sorted(xs)[len(xs)//2]
            x_min, x_max = min(xs), max(xs)
            col_gap = x_max - x_min
            two_columns = col_gap > 350

            if two_columns:
                left_blocks = [b for b in raw_blocks if b[0] < median_x]
                right_blocks = [b for b in raw_blocks if b[0] >= median_x]
                print(f"🧭 Page {page_idx:>2}: two-column layout detected "
                      f"(median x={median_x:.1f}, left={len(left_blocks)}, right={len(right_blocks)})")
                column_groups = (left_blocks, right_blocks)
            else:
                print(f"📄 Page {page_idx:>2}: single-column layout (total {len(raw_blocks)} blocks)")
                column_groups = (raw_blocks,)

            # Проходим по всем блокам
            for col_blocks in column_groups:
                for b in sorted(col_blocks, key=lambda x: x[1]):
                    x0, y0, x1, y1 = b[:4]
                    text = b[4].strip()
                    if not text:
                        continue

                    # 🔧 Исправленная логика классификации подписи
                    clean_text = text.lstrip().replace("\xa0", " ").strip()
                    kind = "body"
                    if re.match(r"^(fig|figure|table)\b", clean_text, re.IGNORECASE):
                        kind = "caption"
                    elif re.match(r"^(eq|equation)\b", clean_text, re.IGNORECASE):
                        kind = "equation"

                    # лог для отладки
                    if kind != "body":
                        print(f"🧩 Page {page_idx:>2} block classified as {kind}: {clean_text[:60]}...")

                    blocks.append(TextBlock(
                        page=page_idx,
                        bbox=(x0, y0, x1, y1),
                        text=text,
                        kind=kind
                    ))

    print(f"\n✅ Extracted {len(blocks)} total text blocks from {total_pages} pages.\n")
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
                bbox=(0, 0, 0, 0),
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
