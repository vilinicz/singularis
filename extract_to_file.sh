#!/usr/bin/env bash
# Извлекает layout из PDF и записывает все блоки в JSON файл

set -e  # остановить при ошибке

# 👇 Укажи путь к PDF
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDF_PATH="${SCRIPT_DIR}/article.pdf"

# 👇 Имя выходного файла
OUT_FILE="layout.json"

if [ ! -f "$PDF_PATH" ]; then
  echo "❌ File not found: $PDF_PATH"
  exit 1
fi

echo "📄 Extracting layout from: $PDF_PATH"
echo "💾 Saving to: $OUT_FILE"

poetry run python <<PYCODE
from singularis.layout_extractor import extract_layout
import json, pathlib

pdf_path = pathlib.Path("${PDF_PATH}")
blocks = extract_layout(pdf_path)

data = [b.model_dump() for b in blocks]

out_path = pathlib.Path("${OUT_FILE}")
out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
print(f"✅ Wrote {len(blocks)} blocks to {out_path}")
PYCODE
