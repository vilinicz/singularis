#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF → (sentences with labels, page, bbox), captions included.
- Two-column aware reading order (heuristic).
- Text normalization (join hard line breaks, dehyphenation).
- Captions for Figure/Table are detected as special items and analyzed too.
- spaCy Matcher/DependencyMatcher labels (8 roles): Input Fact, Hypothesis, Experiment,
  Technique, Result, Dataset, Analysis, Conclusion.
- Consecutive sentences with the same label are merged; bbox is the union.

Deps:
  poetry add spacy pymupdf pdfminer.six pypdf  # (pdfminer/pypdf optional fallback)
  poetry run python -m spacy download en_core_web_sm

Run:
  poetry run python imrad_spacy_layout.py --pdf article.pdf --out out.jsonl --md out.md
"""

import sys, re, json, argparse, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# -------------------- utils --------------------

def dehyphenate(s: str) -> str:
    # join hyphenated line breaks: "exam-\nple" -> "example"
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)
    return s

def normalize_spaces(s: str) -> str:
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

def is_caption_text(line: str) -> Tuple[bool, Optional[str]]:
    # Detect Figure/Table captions at line start
    m = re.match(r"^\s*(Figure|Fig\.?|Table)\s+([0-9IVX]+)[\.:]\s*(.*)", line, re.I)
    if m:
        return True, m.group(1).capitalize()
    return False, None

# -------------------- PDF extraction with PyMuPDF (fitz) --------------------

def extract_blocks_with_bbox(pdf_path: str, max_pages: Optional[int]=None) -> List[Dict]:
    """
    Returns list of blocks:
      {page, x0,y0,x1,y1, text, is_caption, caption_type}
    Reading order is adjusted for two columns via a simple split by dominant vertical gap.
    """
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    n_pages = len(doc) if max_pages is None else min(len(doc), max_pages)

    all_blocks = []
    for pno in range(n_pages):
        page = doc[pno]
        # Use "blocks" (layout blocks: text boxes). Each block: [x0,y0,x1,y1, "text", block_no, block_type]
        blocks = page.get_text("blocks")  # list of tuples
        # Keep only text blocks
        tb = []
        for b in blocks:
            x0,y0,x1,y1,txt,_,bt = b[0],b[1],b[2],b[3],b[4],b[5],b[6] if len(b) > 6 else 0
            if not isinstance(txt, str):
                continue
            if bt != 0:
                continue  # 0 = text
            # normalize block text minimally (do line-level normalization later)
            tb.append((x0,y0,x1,y1,txt))

        if not tb:
            continue

        # Two-column heuristic:
        # compute median x center, and try splitting blocks into left/right columns by a vertical midline
        centers = [(x0+x1)/2 for x0,y0,x1,y1,_ in tb]
        minx = min(x0 for x0,_,_,_,_ in tb)
        maxx = max(x1 for _,_,x1,_,_ in tb)
        mid = (minx + maxx) / 2.0

        left_blocks = [b for b in tb if (b[0]+b[2])/2 <= mid]
        right_blocks = [b for b in tb if (b[0]+b[2])/2 >  mid]

        # Sort reading order: top-to-bottom within each column, left column first
        left_blocks.sort(key=lambda t: (round(t[1],1), round(t[0],1)))
        right_blocks.sort(key=lambda t: (round(t[1],1), round(t[0],1)))

        ordered = left_blocks + right_blocks

        for (x0,y0,x1,y1,txt) in ordered:
            # split by lines; keep bbox for whole block (coarse bbox for lines inside)
            lines = [ln for ln in txt.split("\n") if ln.strip()]
            for ln in lines:
                is_cap, cap_type = is_caption_text(ln)
                all_blocks.append({
                    "page": pno,
                    "x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1),
                    "text": ln,
                    "is_caption": bool(is_cap),
                    "caption_type": cap_type
                })
    doc.close()
    return all_blocks

# -------------------- Sentence splitting with provenance --------------------

def spaCy_pipeline(model: str="en_core_web_sm"):
    import spacy
    try:
        nlp = spacy.load(model)
    except Exception as e:
        raise SystemExit(
            "[error] spaCy model missing. In Poetry:\n"
            "  poetry add spacy\n"
            "  poetry run python -m spacy download en_core_web_sm\n"
        ) from e
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

def make_sentences_from_blocks(nlp, blocks: List[Dict]) -> List[Dict]:
    """
    Turn block-level lines into sentences with (page, bbox) provenance.
    Strategy:
      - Per block line: normalize line text (dehyphenate + strip newlines).
      - Run spaCy on the line and produce sentence spans.
      - Sentence bbox = block bbox (coarse but reliable).
      - Captions are flagged and included in the stream.
    """
    sents = []
    for b in blocks:
        line = dehyphenate(b["text"])
        line = normalize_spaces(line).replace("\n"," ").strip()
        if not line:
            continue
        doc = nlp(line)
        for s in doc.sents:
            st = s.text.strip()
            if not st:
                continue
            sents.append({
                "text": st,
                "page": b["page"],
                "bbox": [b["x0"], b["y0"], b["x1"], b["y1"]],
                "is_caption": b["is_caption"],
                "caption_type": b["caption_type"] or "",
            })
    return sents

# -------------------- IMRAD headings (priors) --------------------

SEC_PATTERNS = [
    (r"^\s*(abstract)\s*$", "INTRO"),
    (r"^\s*(introduction|background)\s*$", "INTRO"),
    (r"^\s*(materials?\s+and\s+methods|methods?|methodology|experimental)\s*$", "METHODS"),
    (r"^\s*(results?|findings|outcomes|evaluation)\s*$", "RESULTS"),
    (r"^\s*(discussion|conclusions?|concluding\s+remarks|implications|limitations)\s*$", "DISCUSSION"),
    (r"^\s*(references|bibliography|works\s+cited)\s*$", "REFERENCES"),
]
SEC_RE = [(re.compile(p, re.I), lab) for p,lab in SEC_PATTERNS]

def assign_section_hints(sentences: List[Dict]) -> None:
    """
    Use block-text lines to detect section-like headings and attach section hints
    to the following sentences until next heading appears.
    Since we already split per line/block, we can cheaply check heading-like lines.
    """
    cur = "OTHER"
    for s in sentences:
        # if a line itself looks like a heading (short, uppercase-ish), switch
        t = s["text"]
        for rx, lab in SEC_RE:
            if rx.match(t):
                cur = lab
                # mark the heading sentence as OTHER to avoid mislabeling
                s["section_hint"] = "OTHER"
                break
        else:
            s["section_hint"] = cur

# -------------------- Matchers (8 labels) --------------------

def build_matchers(nlp):
    from spacy.matcher import Matcher, DependencyMatcher
    m = Matcher(nlp.vocab)
    d = DependencyMatcher(nlp.vocab)

    # RESULT
    m.add("RES_VERB_CUES", [
        [{"LOWER":{"IN":["our","these"]}}, {"LOWER":{"IN":["results","findings"]}},
         {"LEMMA":{"IN":["show","indicate","demonstrate","reveal","suggest"]}}],
        [{"LOWER":"we"}, {"LEMMA":{"IN":["show","find","observe","demonstrate","indicate","reveal","report"]}}],
    ])
    m.add("RES_STATS", [
        [{"LOWER":{"IN":["p","p-value"]}}, {"IS_PUNCT":True,"OP":"?"}, {"LOWER":{"IN":["<","≤","<="]}}, {"LIKE_NUM":True}],
        [{"LIKE_NUM":True}, {"TEXT":{"REGEX":"%"}}, {"LOWER":{"IN":["increase","decrease","improvement","reduction"]}}],
        [{"LIKE_NUM":True}, {"TEXT":{"REGEX":"%"}}, {"LOWER":"ci"}],
        [{"LOWER":{"IN":["odds","hazard"]}}, {"LOWER":"ratio"}],
        [{"LOWER":{"IN":["rmse","auc","auroc","accuracy","precision","recall","sensitivity","specificity"]}}],
        [{"LOWER":"compared"}, {"LOWER":"to"}], [{"LOWER":"achieved"}], [{"LOWER":"yielded"}]
    ])
    d.add("RES_WE_VERB", [[
        {"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["show","find","observe","demonstrate","indicate","reveal","suggest","report"]}}},
        {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"subj","RIGHT_ATTRS":{"DEP":{"IN":["nsubj","nsubjpass"]},"LOWER":"we"}}
    ]])

    # EXPERIMENT
    m.add("EXP_SURFACE", [
        [{"LOWER":"we"}, {"LEMMA":{"IN":["conduct","perform","run","carry","implement"]}}],
        [{"LOWER":"we"}, {"LEMMA":{"IN":["measure","collect","recruit","randomize","enroll","administer"]}}],
        [{"LOWER":{"IN":["trial","experiment","study"]}}],
        [{"LOWER":{"IN":["placebo","control","controlled","double-blind","randomized"]}}],
        [{"TEXT":{"REGEX":"^n\\s*=\\s*\\d+"}}]
    ])
    d.add("EXP_DOBJ", [[
        {"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["conduct","perform","run","carry","measure","use","utilize","apply","calibrate","assemble","administer"]}}},
        {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obj","RIGHT_ATTRS":{"DEP":{"IN":["dobj","obj"]},
         "LOWER":{"IN":["experiment","experiments","study","trial","measurement","setup","apparatus","assay"]}}}
    ]])

    # TECHNIQUE
    m.add("TEC_SURFACE", [
        [{"LOWER":{"IN":["using","with","via","through","by"]}},
         {"POS":{"IN":["DET","ADJ"]},"OP":"*"},
         {"LEMMA":{"IN":["method","technique","protocol","assay","algorithm","pipeline","architecture","classifier","model"]}}],
        [{"LOWER":{"IN":["pcr","rt-pcr","western","elisa","mass","spectrometry","mrna","rna-seq","immunohistochemistry","random-forest","svm","cox","kaplan-meier"]}}],
        [{"LOWER":{"IN":["assay","assays","protocol","protocols"]}}]
    ])
    d.add("TEC_USING", [[
        {"RIGHT_ID":"v","RIGHT_ATTRS":{"POS":"VERB"}},
        {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obl","RIGHT_ATTRS":{"DEP":{"IN":["prep","agent"]}, "LOWER":{"IN":["using","with","via","by","through"]}}}
    ]])

    # DATASET
    m.add("DATA_SURFACE", [
        [{"LOWER":{"IN":["dataset","data","registry","cohort","biobank","database"]}}],
        [{"LOWER":{"IN":["mimic","mimic-iii","mimic-iv","uk","biobank","eicu","clinicaltrials.gov","tcga","physionet"]}}],
        [{"LOWER":{"IN":["patients","participants","subjects"]}}],
        [{"LOWER":"n"}, {"IS_PUNCT":True,"OP":"?"}, {"LOWER":"="}, {"LIKE_NUM":True}],
        [{"LOWER":"nct"}, {"IS_DIGIT":True,"OP":"+"}]
    ])
    d.add("DATA_SOURCE", [[
        {"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["collect","use","utilize","obtain","source","recruit","enroll","include"]}}},
        {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obj","RIGHT_ATTRS":{"DEP":{"IN":["obj","dobj"]},
         "LOWER":{"IN":["data","dataset","datasets","patients","participants","subjects","records","cohort"]}}}
    ]])

    # ANALYSIS
    m.add("ANA_SURFACE", [
        [{"LOWER":"we"}, {"LEMMA":{"IN":["analyze","analyse","assess","evaluate","model","fit","estimate","adjust","normalize","standardize"]}}],
        [{"LOWER":{"IN":["regression","logistic","linear","cox","anova","ancova","mixed-effects","multivariate","univariate"]}}],
        [{"LOWER":{"IN":["kaplan-meier","survival","hazard","odds"]}}],
        [{"LOWER":{"IN":["significance","multiple","testing","bonferroni","fdr"]}}],
    ])
    d.add("ANA_DEP", [[
        {"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["analyze","analyse","evaluate","assess","model","fit","estimate","adjust"]}}},
        {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obj","RIGHT_ATTRS":{"DEP":{"IN":["obj","dobj","obl"]}}}
    ]])

    # HYPOTHESIS
    m.add("HYP_SURFACE", [
        [{"LOWER":{"IN":["we","our"]}}, {"LEMMA":{"IN":["hypothesize","hypothesise","postulate","posit","predict","propose"]}}],
        [{"LOWER":{"IN":["we","our"]}}, {"LOWER":{"IN":["hypothesis","hypotheses"]}}],
        [{"LOWER":{"IN":["we","this","the"]}}, {"LOWER":"study","OP":"?"}, {"LEMMA":{"IN":["aim","seek"]}}, {"LOWER":"to"}],
        [{"LOWER":{"IN":["we"]}}, {"LEMMA":{"IN":["expect"]}}, {"LOWER":"that"}]
    ])
    d.add("HYP_THAT", [[
        {"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["hypothesize","postulate","posit","predict","propose","expect"]}}},
        {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"comp","RIGHT_ATTRS":{"DEP":{"IN":["ccomp","xcomp","advcl"]}}}
    ]])

    # INPUT FACT
    m.add("INF_SURFACE", [
        [{"LOWER":{"IN":["according","given","based"]}}, {"LOWER":"on","OP":"?"}],
        [{"LOWER":{"IN":["it","this"]}}, {"LEMMA":{"IN":["be"]}}, {"LOWER":{"IN":["known","established","well-known"]}}],
        [{"LOWER":{"IN":["prior","previous","existing"]}}, {"LOWER":{"IN":["work","evidence","studies","literature"]}}],
        [{"LOWER":{"IN":["guidelines","consensus","recommendations"]}}],
        [{"LOWER":{"IN":["baseline","assumption","assumptions","inclusion","exclusion","criteria"]}}]
    ])
    d.add("INF_CITATION", [[
        {"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["report","show","demonstrate"]}}},
        {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obl","RIGHT_ATTRS":{"LOWER":{"IN":["previously","earlier"]}}}
    ]])

    # CONCLUSION
    m.add("CONC_SURFACE", [
        [{"LOWER":{"IN":["in","overall"]}}, {"LOWER":"conclusion","OP":"?"}],
        [{"LOWER":{"IN":["in","overall"]}}, {"LOWER":"summary"}],
        [{"LOWER":{"IN":["we"]}}, {"LEMMA":{"IN":["conclude","confirm"]}}],
        [{"LOWER":{"IN":["these","our","the"]}}, {"LOWER":{"IN":["findings","results","data"]}},
         {"LEMMA":{"IN":["support","suggest","highlight","underscore"]}}],
        [{"LOWER":{"IN":["implications","clinical","practice","translation","future"]}}]
    ])
    d.add("CONC_DEP", [[
        {"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["conclude","suggest","support","confirm","highlight","underscore"]}}},
        {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"subj","RIGHT_ATTRS":{"DEP":{"IN":["nsubj","nsubjpass"]}}}
    ]])

    return m, d

LABELS = ["Input Fact","Hypothesis","Experiment","Technique","Result","Dataset","Analysis","Conclusion"]
SECTION_PRIORS = {
    "INTRO":      {"Input Fact":1, "Hypothesis":2},
    "METHODS":    {"Experiment":2, "Technique":2, "Dataset":1, "Analysis":1},
    "RESULTS":    {"Result":2, "Analysis":1},
    "DISCUSSION": {"Conclusion":2, "Result":1, "Hypothesis":1},
    "REFERENCES": {},
    "OTHER":      {}
}
PREFIX2LABEL = {
    "INF_":"Input Fact","HYP_":"Hypothesis","EXP_":"Experiment","TEC_":"Technique",
    "RES_":"Result","DATA_":"Dataset","ANA_":"Analysis","CONC_":"Conclusion"
}

def score_sentence(span, section, matcher, depmatcher):
    scores = {lab:0 for lab in LABELS}
    hits = defaultdict(int)
    for mid, s, e in matcher(span):
        name = span.vocab.strings[mid]; hits[name]+=1
        for pref, lab in PREFIX2LABEL.items():
            if name.startswith(pref):
                scores[lab]+=1
    for mid, toks in depmatcher(span):
        name = span.vocab.strings[mid]; hits[name]+=1
        for pref, lab in PREFIX2LABEL.items():
            if name.startswith(pref):
                scores[lab]+=2
    for lab,w in SECTION_PRIORS.get(section, {}).items():
        scores[lab]+=w
    return scores, hits

def looks_like_reference(text: str) -> bool:
    _J = [r"\bvol\.?\b", r"\bno\.?\b", r"\bpp\.?\b", r"\bdoi:?\b", r"\bissn\b", r"\bet al\.?\b"]
    if re.search("|".join(_J), text, re.I):
        return True
    if re.search(r"\b(19|20)\d{2}\b", text):  # years
        if re.search(r"\b\d{1,4}\s*([–-]\s*\d{1,4})?\b", text):  # pages
            return True
    return False

def decide_label(scores: Dict[str,int], section: str, had_matches: bool) -> str:
    if section=="REFERENCES" or not had_matches:
        return "OTHER" if section != "RESULTS" else "Result"
    m = max(scores.values())
    if m <= 0: return "OTHER"
    cands = [lab for lab,v in scores.items() if v==m]
    order = ["Result","Experiment","Technique","Analysis","Dataset","Hypothesis","Conclusion","Input Fact"]
    for lab in order:
        if lab in cands: return lab
    return cands[0]

# -------------------- Merge consecutive same-label sentences --------------------

def merge_adjacent(records: List[Dict]) -> List[Dict]:
    if not records: return records
    merged = []
    cur = dict(records[0])
    for r in records[1:]:
        # merge if same label and same page (simple heuristic)
        if r["label"] == cur["label"] and r["page"] == cur["page"]:
            cur["text"] += " " + r["text"]
            # union bbox
            x0 = min(cur["bbox"][0], r["bbox"][0]); y0 = min(cur["bbox"][1], r["bbox"][1])
            x1 = max(cur["bbox"][2], r["bbox"][2]); y1 = max(cur["bbox"][3], r["bbox"][3])
            cur["bbox"] = [x0,y0,x1,y1]
        else:
            merged.append(cur); cur = dict(r)
    merged.append(cur)
    return merged

# -------------------- Main --------------------

def main():
    ap = argparse.ArgumentParser(description="PDF → sentences (8 roles) with page/bbox, captions included")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", type=int, default=0, help="max pages to read (0=all)")
    ap.add_argument("--out", default="out.jsonl")
    ap.add_argument("--md", default="")
    ap.add_argument("--model", default="en_core_web_sm")
    args = ap.parse_args()

    # 1) PDF → block lines with bbox (two-column aware)
    blocks = extract_blocks_with_bbox(args.pdf, max_pages=(args.pages or None))
    # quick normalization at line-level already done later

    # 2) spaCy pipeline
    nlp = spaCy_pipeline(args.model)
    matcher, depmatcher = build_matchers(nlp)

    # 3) Sentences with bbox (coarse: block bbox)
    sents = make_sentences_from_blocks(nlp, blocks)
    # 4) Section hints
    assign_section_hints(sents)

    # 5) Score & label
    records = []
    for i, s in enumerate(sents):
        text = s["text"]
        sec = s["section_hint"]
        doc = nlp.make_doc(text)  # no need to re-split
        # DependencyMatcher needs parsed doc; but for a single sentence we can run nlp(text)
        doc = nlp(text)
        scores, hits = score_sentence(doc, sec, matcher, depmatcher)
        had_matches = bool(hits)
        if looks_like_reference(text):  # filter bib tails
            label = "OTHER"
        else:
            label = decide_label(scores, sec, had_matches)
        records.append({
            "idx": i,
            "page": s["page"],
            "bbox": s["bbox"],
            "section_hint": sec,
            "is_caption": s["is_caption"],
            "caption_type": s["caption_type"],
            "label": label,
            "scores": scores,
            "matches": hits,
            "text": text
        })

    # 6) Merge consecutive same-label sentences
    records = merge_adjacent(records)

    # 7) Write
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.md:
        groups = defaultdict(list)
        for r in records: groups[r["label"]].append(r)
        lines = ["# Labels (merged spans)\n"]
        for lab in ["Result","Experiment","Technique","Analysis","Dataset","Hypothesis","Conclusion","Input Fact","OTHER"]:
            if not groups[lab]: continue
            lines.append(f"## {lab}  \n(count: {len(groups[lab])})")
            for r in groups[lab][:200]:
                p = r["page"]+1
                lines.append(f"- p.{p} {r['text']}")
            lines.append("")
        Path(args.md).write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] items={len(records)} jsonl={args.out}" + (f" md={args.md}" if args.md else ""))

if __name__ == "__main__":
    main()
