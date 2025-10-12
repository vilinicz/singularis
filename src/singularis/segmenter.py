from __future__ import annotations
import json
from pathlib import Path
import spacy
from tqdm import tqdm
from .layout_extractor import extract_layout
from .models import TextBlock


def segment_pdf(pdf_path: str | Path):
    """Выполняет сегментацию PDF на предложения и сохраняет результат в outputs/."""
    pdf_path = Path(pdf_path)
    base_dir = pdf_path.parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    layout_path = out_dir / "layout.json"
    segments_path = out_dir / "segments.json"

    # === 1. Загружаем layout или создаём заново ===
    if layout_path.exists():
        print(f"📄 Using existing layout: {layout_path}")
        with open(layout_path, "r") as f:
            data = json.load(f)
        blocks = [TextBlock(**b) for b in data]
    else:
        print(f"⚙️ Layout not found, extracting from {pdf_path.name}...")
        blocks = extract_layout(pdf_path)
        json.dump([b.model_dump() for b in blocks], open(layout_path, "w"), indent=2)
        print(f"✅ Layout saved to {layout_path}")

    # === 2. Инициализация spaCy ===
    print("⚙️ Loading spaCy model (en_core_web_md)...")
    nlp = spacy.load("en_core_web_md")

    # === 3. Разделение на предложения ===
    print("✂️ Splitting text into sentences...")
    sentences = []
    sid = 0
    for block in tqdm(blocks, desc="Processing blocks"):
        if not block.text.strip():
            continue
        doc = nlp(block.text)
        for sent in doc.sents:
            sentences.append({
                "id": f"s_{sid}",
                "page": block.page,
                "text": sent.text.strip(),
                "source_bbox": block.bbox,
                "kind": block.kind
            })
            sid += 1

    # === 4. Сохраняем результат ===
    json.dump(sentences, open(segments_path, "w"), indent=2)
    print(f"\n✅ Segmented {len(sentences)} sentences → {segments_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m singularis.segmenter <path_to_pdf>")
        sys.exit(1)
    segment_pdf(sys.argv[1])
