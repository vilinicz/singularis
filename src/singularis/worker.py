#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Worker/orchestrator (in-memory) that ALWAYS writes artifacts into outdir:
  - outdir/s0.json
  - outdir/s1.json  (JSONL)
  - outdir/<pdf_stem>.tei.xml
"""
import argparse
from pathlib import Path
import json

from .parse_pdf_grobid import parse_pdf_to_items
from .label_with_spacy import label_items

PKG_DIR = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser(description="IMRaD pipeline worker (always writes outputs)")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--server", default="http://localhost:8070")
    ap.add_argument("--patterns", default=str(PKG_DIR / "rules.json"))
    ap.add_argument("--outdir", default=str(PKG_DIR.parent.parent / "outputs"))
    ap.add_argument("--model", default="en_core_web_sm")
    ap.add_argument("--citation-soft", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pdf_stem = Path(args.pdf).stem
    s0_path  = outdir / "s0.json"
    s1_path  = outdir / "s1.json"
    tei_path = outdir / f"{pdf_stem}.tei.xml"

    # Step 1: parse (in-memory) + ALWAYS dump TEI + s0.json
    items, tei = parse_pdf_to_items(args.server, args.pdf)
    tei_path.write_text(tei, encoding="utf-8")
    s0_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    # Step 2: label (in-memory) + ALWAYS dump s1.json (JSONL)
    records = label_items(items, args.patterns, model=args.model, citation_soft=args.citation_soft)
    s1_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    print(f"[worker] items={len(items)} labeled={len(records)}")
    print(f"  tei → {tei_path}")
    print(f"  s0  → {s0_path}")
    print(f"  s1  → {s1_path}")

if __name__ == "__main__":
    main()
