#!/usr/bin/env bash
set -e
# Извлекает layout из PDF и сохраняет его в JSON

# Определяем путь до корня проекта
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PDF_PATH="$PROJECT_DIR/article2.pdf"
OUT_PATH="$PROJECT_DIR/outputs/layout.json"

echo "📘 Extracting layout from: $PDF_PATH"

poetry run python - <<PYCODE
from singularis.layout_extractor import extract_layout
from pathlib import Path
import json

pdf = Path("$PDF_PATH")
out_path = Path("$OUT_PATH")
blocks = extract_layout(pdf)
out_path.parent.mkdir(exist_ok=True)
json.dump([b.model_dump() for b in blocks], open(out_path, "w"), indent=2)
print(f"✅ Layout extracted → {out_path}")
PYCODE
