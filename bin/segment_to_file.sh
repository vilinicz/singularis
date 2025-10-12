#!/usr/bin/env bash
set -e

# === Путь до проекта и PDF ===
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PDF_PATH="$BASE_DIR/article.pdf"
OUT_DIR="$BASE_DIR/outputs"

mkdir -p "$OUT_DIR"

echo "🧩 Starting segmentation for: $PDF_PATH"

LAYOUT_JSON="$OUT_DIR/layout.json"
SEGMENTS_JSON="$OUT_DIR/segments.json"

# === Проверяем, есть ли layout ===
if [ ! -f "$LAYOUT_JSON" ]; then
  echo "⚙️  Layout not found, running extractor first..."
  poetry run python -m singularis.pdf_extractor "$PDF_PATH" > "$LAYOUT_JSON"
else
  echo "📄 Using existing layout: $LAYOUT_JSON"
fi

# === Запускаем сегментацию ===
echo "✂️  Running segmenter..."
poetry run python -m singularis.segmenter "$PDF_PATH"

echo "✅ Done! Segments saved to: $SEGMENTS_JSON"
