#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDF_PATH="${SCRIPT_DIR}/article.pdf"

if [ ! -f "$PDF_PATH" ]; then
  echo "❌ File not found: $PDF_PATH"
  exit 1
fi

poetry run python <<PYCODE
from singularis.layout_extractor import extract_layout
import json, pathlib
pdf = pathlib.Path("${PDF_PATH}")
blocks = extract_layout(pdf)
print(f"Total blocks: {len(blocks)}")
for b in blocks[:20]:
    print(f"{b.page:>2} | {b.kind:<7} | {b.text[:80]!r}")
PYCODE

# This is to test pdf_extractor
#poetry run python <<PYCODE
#from singularis.pdf_extractor import extract_text_blocks
#import json, sys, pathlib
#
#pdf_path = pathlib.Path("${PDF_PATH}")
#print(f"📄 Extracting from: {pdf_path}")
#
#blocks = extract_text_blocks(pdf_path)
#print(f"Total blocks: {len(blocks)}")
#
#for b in blocks[:10]:
#    print(f"Page {b.page} | y={b.bbox[1]:.1f} -> {b.text[:80]!r}")
#PYCODE
