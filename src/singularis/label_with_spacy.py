#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 (library + CLI): spaCy rule-based labeling
Library:
  records = label_items(items, patterns_or_path, model="en_core_web_sm", citation_soft=True)
CLI:
  poetry run python -m singularis.label_with_spacy --s0 s0.json --patterns rules.json --out s1.json
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List, Union
from collections import defaultdict
import re

import spacy
from spacy.matcher import Matcher, DependencyMatcher

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
CIT_STRUCT_BONUS = 4
CIT_RULE_BONUS = 3
CIT_RULE_PREFIX = "INF_CIT_"

def looks_like_reference(text: str) -> bool:
    if re.search(r"\b(vol\.?|no\.?|pp\.?|doi:?|issn|et al\.)\b", text, re.I): return True
    if re.search(r"\b(19|20)\d{2}\b", text) and re.search(r"\b\d{1,4}\s*[–-]\s*\d{1,4}\b", text): return True
    return False

def build_nlp(model="en_core_web_sm"):
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

def _load_patterns(patterns_or_path: Union[str, Path, Dict]) -> Dict:
    if isinstance(patterns_or_path, (str, Path)):
        return json.loads(Path(patterns_or_path).read_text(encoding="utf-8"))
    return patterns_or_path  # already a dict

def _build_matchers(nlp, spec: Dict):
    matcher = Matcher(nlp.vocab)
    depmatcher = DependencyMatcher(nlp.vocab)
    for name, pat_list in spec.get("matcher", {}).items():
        matcher.add(name, pat_list)
    for name, dep_list in spec.get("depmatcher", {}).items():
        depmatcher.add(name, dep_list)
    return matcher, depmatcher

def _score_sentence(doc_sent, section, matcher, depmatcher):
    scores = {lab: 0 for lab in LABELS}
    hits = defaultdict(int)
    for mid, s, e in matcher(doc_sent):
        name = doc_sent.vocab.strings[mid]; hits[name] += 1
        for pref, lab in PREFIX2LABEL.items():
            if name.startswith(pref):
                scores[lab] += 1
        if name.startswith(CIT_RULE_PREFIX):
            scores["Input Fact"] += CIT_RULE_BONUS
    for mid, toks in depmatcher(doc_sent):
        name = doc_sent.vocab.strings[mid]; hits[name] += 1
        for pref, lab in PREFIX2LABEL.items():
            if name.startswith(pref):
                scores[lab] += 2
    for lab, w in SECTION_PRIORS.get(section, {}).items():
        scores[lab] += w
    return scores, hits

def _decide_label(scores: Dict[str,int], section: str, had_matches: bool) -> str:
    if section=="REFERENCES" or not had_matches:
        return "OTHER" if section != "RESULTS" else "Result"
    m = max(scores.values())
    if m <= 0: return "OTHER"
    cands = [lab for lab,v in scores.items() if v==m]
    order = ["Result","Experiment","Technique","Analysis","Dataset","Hypothesis","Conclusion","Input Fact"]
    for lab in order:
        if lab in cands: return lab
    return cands[0]

# ---------- public API ----------
def label_items(items: List[Dict], patterns_or_path: Union[str, Path, Dict],
                *, model="en_core_web_sm", citation_soft=True) -> List[Dict]:
    """Return labeled records (list of dicts) without writing files."""
    nlp = build_nlp(model)
    spec = _load_patterns(patterns_or_path)
    matcher, depmatcher = _build_matchers(nlp, spec)

    out = []
    for i, it in enumerate(items):
        doc = nlp(it["text"])
        scores, hits = _score_sentence(doc, it["section_hint"], matcher, depmatcher)

        if it.get("has_citation"):
            scores["Input Fact"] += CIT_STRUCT_BONUS
            hits["INF_CIT_STRUCT"] = hits.get("INF_CIT_STRUCT", 0) + 1

        label = _decide_label(scores, it["section_hint"], had_matches=bool(hits))

        if looks_like_reference(it["text"]):
            label = "OTHER"

        if (not it.get("has_citation")) and citation_soft and label == "Input Fact":
            non_cit_hits = sum(v for k, v in hits.items() if not (k.startswith("INF_CIT_") or k in {"INF_CITATION","INF_CIT_STRUCT"}))
            cit_hits = sum(v for k, v in hits.items() if (k.startswith("INF_CIT_") or k in {"INF_CITATION","INF_CIT_STRUCT"}))
            if cit_hits > 0 and non_cit_hits == 0:
                label = "OTHER"

        out.append({
            "idx": i, "section": it["section_hint"], "label": label, "text": it["text"],
            "page": it["page"], "bbox": it["bbox"], "is_caption": it["is_caption"],
            "caption_type": it["caption_type"], "scores": scores, "matches": dict(hits),
        })
    return out

# ---------- CLI wrapper (optional dump) ----------
def main():
    ap = argparse.ArgumentParser(description="Step 2: label s0.json → s1.json (optional)")
    ap.add_argument("--s0", required=True)
    ap.add_argument("--patterns", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="en_core_web_sm")
    ap.add_argument("--citation-soft", action="store_true")
    args = ap.parse_args()

    items = json.loads(Path(args.s0).read_text(encoding="utf-8"))
    records = label_items(items, args.patterns, model=args.model, citation_soft=args.citation_soft)

    if args.out:
        Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")

    print(f"[step2] items={len(records)} out={args.out or '-'}")

if __name__ == "__main__":
    main()
