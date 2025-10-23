#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 (library + CLI): GROBID → TEI (with sentences + coords) → items (Python list of dicts)
Library:
  items, tei = parse_pdf_to_items(server, pdf_path)
CLI:
  poetry run python -m singularis.parse_pdf_grobid --pdf ... --server ... --out s0.json
"""
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Union
import requests
from lxml import etree

COORD_ITEM_RE = re.compile(
    r"^\s*(?:p)?(?P<page>\d+)\s*[: ,]\s*(?P<x>[-\d.]+)\s*,\s*(?P<y>[-\d.]+)\s*,\s*(?P<w>[-\d.]+)\s*,\s*(?P<h>[-\d.]+)"
)
_WORD = r"(?:^|[^a-z])"; _EOW = r"(?:$|[^a-z])"

# ---------- low-level helpers ----------
def grobid_fulltext_tei(server: str, pdf_path: Union[str, Path], timeout: int = 120) -> str:
    url = server.rstrip("/") + "/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        files = {"input": f}
        data = [
            ("segmentSentences", "1"),
            ("teiCoordinates", "s"),
            ("teiCoordinates", "p"),
            ("teiCoordinates", "head"),
            ("teiCoordinates", "figure"),
            ("teiCoordinates", "table"),
            ("teiCoordinates", "biblStruct"),
        ]
        r = requests.post(url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.text

def parse_coords_attr(coords: str) -> List[Dict]:
    if not coords:
        return []
    boxes = []
    for chunk in coords.split(";"):
        m = COORD_ITEM_RE.match(chunk)
        if not m:
            continue
        page = int(m.group("page"))
        x = float(m.group("x")); y = float(m.group("y"))
        w = float(m.group("w")); h = float(m.group("h"))
        boxes.append({"page": page, "x": x, "y": y, "w": w, "h": h})
    return boxes

def union_bbox(boxes: List[Dict]) -> Tuple[int, List[float]]:
    if not boxes:
        return 1, [0,0,0,0]
    page = boxes[0]["page"]
    xs0 = [b["x"] for b in boxes if b["page"] == page]
    ys0 = [b["y"] for b in boxes if b["page"] == page]
    xs1 = [b["x"] + b["w"] for b in boxes if b["page"] == page]
    ys1 = [b["y"] + b["h"] for b in boxes if b["page"] == page]
    return page, [min(xs0), min(ys0), max(xs1), max(ys1)]

def _clean_head_text(txt: str) -> str:
    t = (txt or "").strip()
    t = re.sub(r"^\s*(?:\d+|[IVXLCM]+)[\.)]?\s+", "", t, flags=re.I)
    return t.replace("&", "and").lower()

def map_head_to_hint(head_text: str) -> str:
    t = _clean_head_text(head_text)
    if not t: return "OTHER"
    import re as _re
    if _re.search(rf"{_WORD}(abstract|introduction|background|aims and scope){_EOW}", t): return "INTRO"
    if (_re.search(rf"{_WORD}(materials? and methods?){_EOW}", t) or
        _re.search(rf"{_WORD}(methods?|methodology){_EOW}", t) or
        _re.search(rf"{_WORD}(experimental(?: section)?){_EOW}", t) or
        _re.search(rf"{_WORD}(patients? and methods?|subjects? and methods?){_EOW}", t) or
        _re.search(rf"{_WORD}(study design){_EOW}", t) or
        _re.search(rf"{_WORD}(statistical (analysis|methods?)){_EOW}", t)):
        return "METHODS"
    if (_re.search(rf"{_WORD}(results? and discussion){_EOW}", t) or
        _re.search(rf"{_WORD}(general discussion|discussion|conclusions?|concluding remarks|implications|limitations){_EOW}", t)):
        return "DISCUSSION"
    if _re.search(rf"{_WORD}(results?|findings|outcomes){_EOW}", t): return "RESULTS"
    if _re.search(rf"{_WORD}(references|bibliography|works cited){_EOW}", t): return "REFERENCES"
    return "OTHER"

def _page0_from_boxes(boxes):
    page, _bbox = union_bbox(boxes)
    return (page - 1) if page is not None else None, _bbox

def has_struct_citation(sent_el: etree._Element) -> bool:
    for el in sent_el.iter():
        tag = etree.QName(el).localname.lower()
        if tag in {"ref","ptr"}:
            typ = (el.get("type") or "").lower()
            tgt = (el.get("target") or el.get("{http://www.w3.org/1999/xlink}href") or "")
            if typ in {"bibr","bibl","citation"}: return True
            if tgt.startswith("#b") or "bibl" in tgt.lower(): return True
        if tag == "bibl": return True
    return False

def is_in_abstract(node: etree._Element) -> bool:
    """
    Возвращает True, если узел находится внутри <abstract> или <div type="abstract|summary">.
    Работает для структур GROBID: <abstract> в teiHeader/front или <text>/<front>/<div type="abstract">.
    """
    el = node
    while el is not None:
        tag = etree.QName(el).localname.lower()
        if tag == "abstract":
            return True
        if tag == "div" and (el.get("type") or "").lower() in {"abstract", "summary"}:
            return True
        el = el.getparent()
    return False

def tei_iter_sentences(tei_xml: str) -> Iterable[Dict]:
    root = etree.fromstring(tei_xml.encode("utf-8")) if isinstance(tei_xml, str) else tei_xml
    current_imrad_section = "OTHER"
    for el in root.iter():
        tag = etree.QName(el).localname
        if tag == "head":
            head_txt = "".join(el.itertext()).strip()
            label = map_head_to_hint(head_txt)
            if label != "OTHER":
                current_imrad_section = label
        elif tag == "s":
            text = "".join(el.itertext()).strip()
            if not text: continue
            boxes = parse_coords_attr(el.get("coords") or "")
            page0, bbox = _page0_from_boxes(boxes)
            section_hint = "ABSTRACT" if is_in_abstract(el) else current_imrad_section

            yield {"text": text, "page": page0, "bbox": bbox, "section_hint": section_hint,
                   "is_caption": False, "caption_type": "", "has_citation": has_struct_citation(el)}
        elif tag == "figDesc":
            text = "".join(el.itertext()).strip()
            if not text: continue
            boxes = parse_coords_attr(el.get("coords") or "")
            if not boxes:
                fig = el.getparent() if el.getparent() is not None and etree.QName(el.getparent()).localname == "figure" else None
                if fig is not None:
                    boxes = parse_coords_attr(fig.get("coords") or "")
            page0, bbox = _page0_from_boxes(boxes)
            yield {"text": text, "page": page0, "bbox": bbox, "section_hint": current_imrad_section,
                   "is_caption": True, "caption_type": "Figure", "has_citation": has_struct_citation(el)}
        elif tag == "table":
            thead = el.find("./{http://www.tei-c.org/ns/1.0}head")
            if thead is None: continue
            text = "".join(thead.itertext()).strip()
            if not text: continue
            boxes = parse_coords_attr(thead.get("coords") or "") or parse_coords_attr(el.get("coords") or "")
            page0, bbox = _page0_from_boxes(boxes)
            yield {"text": text, "page": page0, "bbox": bbox, "section_hint": current_imrad_section,
                   "is_caption": True, "caption_type": "Table", "has_citation": has_struct_citation(el)}

# ---------- public API ----------
def parse_pdf_to_items(server: str, pdf_path: Union[str, Path]):
    """Return (items_list, tei_str) without writing files."""
    tei = grobid_fulltext_tei(server, pdf_path)
    items = list(tei_iter_sentences(tei))
    return items, tei

# ---------- CLI wrapper (optional dump) ----------
def main():
    ap = argparse.ArgumentParser(description="Step 1: GROBID parse → s0.json (optional)")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--server", default="http://localhost:8070")
    ap.add_argument("--out", default="")
    ap.add_argument("--tei-out", default="")  # optional explicit tei path
    args = ap.parse_args()

    items, tei = parse_pdf_to_items(args.server, args.pdf)

    if args.tei_out:
        Path(args.tei_out).write_text(tei, encoding="utf-8")
    else:
        Path(args.pdf).with_suffix(".tei.xml").write_text(tei, encoding="utf-8")

    if args.out:
        Path(args.out).write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[step1] items={len(items)} out={args.out or '-'} tei={(args.tei_out or Path(args.pdf).with_suffix('.tei.xml'))}")

if __name__ == "__main__":
    main()
