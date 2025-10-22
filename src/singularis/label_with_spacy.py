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

# -----------------------
# Labels & section priors
# -----------------------
LABELS = ["Input Fact","Hypothesis","Experiment","Technique","Result","Dataset","Analysis","Conclusion"]

SECTION_PRIORS = {
    "ABSTRACT": {"Hypothesis": 1},
    "INTRO": {"Input Fact": 1, "Hypothesis": 2},
    "METHODS": {"Experiment": 2, "Technique": 2, "Dataset": 1, "Analysis": 1},
    "RESULTS": {"Result": 2, "Analysis": 1},
    "DISCUSSION": {"Conclusion": 2, "Result": 1, "Hypothesis": 1},
    "CONCLUSION": {"Conclusion": 2},
    "REFERENCES": {},
    "OTHER": {},
}

PREFIX2LABEL = {
    "INF_":"Input Fact","HYP_":"Hypothesis","EXP_":"Experiment","TEC_":"Technique",
    "RES_":"Result","DATA_":"Dataset","ANA_":"Analysis","CONC_":"Conclusion"
}

# -----------------------------
# Citation bonuses (your logic)
# -----------------------------
CIT_STRUCT_BONUS = 4
CIT_RULE_BONUS = 3
CIT_RULE_PREFIX = "INF_CIT_"

# =========================
# [IMRAD-A] Tunable weights
# =========================
WEIGHTS = {
    "boosts": {
        "ANA": 3,           # has_analysis_test
        "RES": 3,           # has_res_summary or has_res_verb_cues or has_res_stats
        "HYP_INTRO": 2,     # hyp cues in INTRO/ABSTRACT
        "EXP": 2,           # experiment ops
        "TEC": 2,           # using/with, scales
        "DATA_NO_SIG": 2,   # dataset cues without significance
        "INF_CIT": 1        # citation in INTRO/ABSTRACT
    },
    # [IMRAD-E] universal tie order when scores tie
    "tie_order": ["Conclusion","Result","Analysis","Technique","Experiment","Dataset","Input Fact","OTHER"]
}

# Секционно-зависимые приоритеты (IMRAD)
SECTION_TIE_ORDER = {
    # Аннотация: мини-вывод и цели
    "ABSTRACT":   ["Conclusion","Result","Hypothesis","Dataset","Technique","Analysis","Experiment","Input Fact"],
    # Введение: цель/фон > интерпретации
    "INTRO":      ["Hypothesis","Input Fact","Conclusion","Result","Analysis","Technique","Experiment","Dataset"],
    # Методы: техника и эксперимент первыми
    "METHODS":    ["Technique","Experiment","Dataset","Analysis","Result","Input Fact","Hypothesis","Conclusion"],
    # Результаты: результаты > анализ > датасет
    "RESULTS":    ["Result","Analysis","Dataset","Technique","Experiment","Conclusion","Input Fact","Hypothesis"],
    # Дискуссия: выводы > результаты > анализ
    "DISCUSSION": ["Conclusion","Result","Analysis","Technique","Experiment","Dataset","Input Fact","Hypothesis"],
    # Заключение: выводы на первом месте
    "CONCLUSION": ["Conclusion","Result","Analysis","Technique","Experiment","Dataset","Input Fact","Hypothesis"],
}

SIG_WORDS = {"significant", "significantly", "significance"}

# ABSTRACT header rule ids (from rules.json)
# RES_ABS_HEAD_RESULTS and CONC_ABS_HEAD come from matcher
ABS_HEAD_RESULTS_KEYS = ["RES_ABS_HEAD_RESULTS"]
ABS_HEAD_CONC_KEYS    = ["CONC_ABS_HEAD"]

# ----------------------
# Small helper functions
# ----------------------
def looks_like_reference(text: str) -> bool:
    if re.search(r"\b(vol\.?|no\.?|pp\.?|doi:?|issn|et al\.)\b", text, re.I):
        return True
    if re.search(r"\b(19|20)\d{2}\b", text) and re.search(r"\b\d{1,4}\s*[–-]\s*\d{1,4}\b", text):
        return True
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

# ======================================================
# scoring by raw rule hits (kept close to your original)
# ======================================================
def _score_sentence(doc_sent, section, matcher, depmatcher):
    scores = {lab: 0 for lab in LABELS}
    hits = defaultdict(int)

    # Surface patterns
    for mid, s, e in matcher(doc_sent):
        name = doc_sent.vocab.strings[mid]
        hits[name] += 1
        for pref, lab in PREFIX2LABEL.items():
            if name.startswith(pref):
                scores[lab] += 1
        if name.startswith(CIT_RULE_PREFIX):
            scores["Input Fact"] += CIT_RULE_BONUS

    # Dependency patterns
    for mid, toks in depmatcher(doc_sent):
        name = doc_sent.vocab.strings[mid]
        hits[name] += 1
        for pref, lab in PREFIX2LABEL.items():
            if name.startswith(pref):
                scores[lab] += 2

    # Your original section priors (section already normalized by upstream)
    for lab, w in SECTION_PRIORS.get(section, {}).items():
        scores[lab] += w

    return scores, hits

# ==========================================
# [IMRAD-C] Flags computed from hits & tokens
# ==========================================
def _any_hit(hits: Dict[str,int], keys) -> bool:
    return any(k in hits and hits[k] > 0 for k in keys)

def _compute_flags(doc, hits: Dict[str,int], section_hint: str, has_citation_field: bool):
    # section comes normalized from parse_pdf_grobid / worker.py
    sec = (section_hint or "").upper()

    # result cues
    has_res_summary    = _any_hit(hits, ["RES_SUMMARY"])
    has_res_verb_cues  = _any_hit(hits, ["RES_VERB_CUES","RES_WE_VERB"])
    has_res_stats      = _any_hit(hits, ["RES_STATS"])

    # analysis cues
    has_analysis_test  = _any_hit(hits, ["ANA_SURFACE","ANA_DEP","ANA_USING"])

    # hyp intro (treat ABSTRACT as intro-like if not merged upstream)
    is_intro_like      = sec in {"INTRO","ABSTRACT"}
    has_hyp_intro      = _any_hit(hits, ["HYP_SURFACE"]) and is_intro_like

    # experiment ops
    has_experiment_ops = _any_hit(hits, ["EXP_SURFACE","EXP_DOBJ"])

    # technique
    has_tech_using     = _any_hit(hits, ["TEC_USING","TEC_SURFACE"])
    has_scale_classif  = has_tech_using  # included in TEC_SURFACE patterns

    # dataset
    has_dataset_cues   = _any_hit(hits, ["DATA_SURFACE"])

    # citation rules or structural field
    has_citation_rule  = _any_hit(hits, [
        "INF_CIT_BRACK_NUM","INF_CIT_PAREN_AUTHOR_YEAR","INF_CIT_PAREN_YEAR_ONLY","INF_CIT_ETAL_YEAR","INF_CIT_DOI"
    ]) or bool(has_citation_field)

    # significance (word or stats)
    has_significance_word = any(tok.lemma_.lower() in SIG_WORDS for tok in doc)
    has_significance = has_res_stats or has_significance_word

    # ABSTRACT-specific cues
    text = doc.text
    has_abs_head_results = _any_hit(hits, ABS_HEAD_RESULTS_KEYS)
    has_abs_head_conc    = _any_hit(hits, ABS_HEAD_CONC_KEYS)
    has_pct_list         = (text.count("%") >= 2)

    return {
        "section": sec,
        "has_res_summary": has_res_summary,
        "has_res_verb_cues": has_res_verb_cues,
        "has_res_stats": has_res_stats,
        "has_analysis_test": has_analysis_test,
        "has_hyp_intro": has_hyp_intro,
        "has_experiment_ops": has_experiment_ops,
        "has_tech_using": has_tech_using,
        "has_scale_classification": has_scale_classif,
        "has_dataset_cues": has_dataset_cues,
        "has_citation_rule": has_citation_rule,
        "has_significance": has_significance,
        # new flags used in boosts/tiebreaks
        "has_abs_head_results": has_abs_head_results,
        "has_abs_head_conc": has_abs_head_conc,
        "has_pct_list": has_pct_list,
        "has_hyp_surface_any": _any_hit(hits, ["HYP_SURFACE"])
    }

# ============================================
# [IMRAD-D] Apply soft boosts (section priors
# уже применены в _score_sentence)
# ============================================
def _apply_boosts(scores: Dict[str,int], flags: Dict[str, bool]):
    if flags["has_analysis_test"]:
        scores["Analysis"] = scores.get("Analysis", 0) + WEIGHTS["boosts"]["ANA"]
    if flags["has_res_summary"] or flags["has_res_verb_cues"] or flags["has_res_stats"]:
        scores["Result"] = scores.get("Result", 0) + WEIGHTS["boosts"]["RES"]
    if flags["has_hyp_intro"]:
        scores["Hypothesis"] = scores.get("Hypothesis", 0) + WEIGHTS["boosts"]["HYP_INTRO"]
    if flags["has_experiment_ops"]:
        scores["Experiment"] = scores.get("Experiment", 0) + WEIGHTS["boosts"]["EXP"]
    if flags["has_tech_using"] or flags["has_scale_classification"]:
        scores["Technique"] = scores.get("Technique", 0) + WEIGHTS["boosts"]["TEC"]
    if flags["has_dataset_cues"] and not flags["has_significance"]:
        scores["Dataset"] = scores.get("Dataset", 0) + WEIGHTS["boosts"]["DATA_NO_SIG"]
    if flags["section"] in {"INTRO","ABSTRACT"} and flags["has_citation_rule"]:
        scores["Input Fact"] = scores.get("Input Fact", 0) + WEIGHTS["boosts"]["INF_CIT"]

    # ABSTRACT: headers + percent lists → Result/Conclusion
    if flags["section"] == "ABSTRACT":
        if flags["has_abs_head_results"]:
            scores["Result"] = scores.get("Result", 0) + 2
        if flags["has_abs_head_conc"]:
            scores["Conclusion"] = scores.get("Conclusion", 0) + 2
        if flags["has_pct_list"]:
            scores["Result"] = scores.get("Result", 0) + 1

    # ABSTRACT/METHODS: when dataset cues present and no strong significance/analysis — help Dataset
    if flags["section"] in {"ABSTRACT","METHODS"} and flags["has_dataset_cues"] and not (flags["has_significance"] or flags["has_analysis_test"]):
        scores["Dataset"] = scores.get("Dataset", 0) + 1

    # INTRO/ABSTRACT: aims/objectives detected — slightly downweight Technique so it doesn't overshadow Hypothesis
    if flags["section"] in {"INTRO","ABSTRACT"} and flags.get("has_hyp_surface_any") and scores.get("Technique", 0) > 0:
        scores["Hypothesis"] += 2
        scores["Technique"] -= 1

    return scores

# =========================================
# [IMRAD-E] Universal tie-breaks & postfixes
# =========================================
def _resolve_label(scores: Dict[str,int], flags: Dict[str,bool], had_matches: bool) -> str:
    if flags["section"] == "REFERENCES" or not had_matches:
        return "OTHER" if flags["section"] != "RESULTS" else "Result"
    if not scores:
        return "OTHER"

    # максимум и список кандидатов по очкам
    max_score = max(scores.values())
    cands = [lab for lab, v in scores.items() if v == max_score]

    # Берём секционный порядок (иначе глобальный)
    section_order = SECTION_TIE_ORDER.get(flags["section"], WEIGHTS["tie_order"])

    # 1) Если ничья — выбираем по секционному порядку
    if len(cands) > 1:
        cands.sort(key=lambda x: section_order.index(x) if x in section_order else len(section_order))
        chosen = cands[0]
    else:
        chosen = cands[0]

    # 2) Почти-ничья (delta ≤ 1): если есть класс, который в секционном порядке предпочтительнее chosen
    #    и отстаёт на <=1 балл, поднимем его.
    delta = 1
    for pref in section_order:
        if pref == chosen:
            break
        if scores.get(pref, -999) >= max_score - delta:
            chosen = pref
            break

    # --- мягкие пост-фиксы ---
    if chosen == "Technique" and flags["has_analysis_test"]:
        chosen = "Analysis"
    if chosen == "Technique" and (flags["has_res_summary"] or flags["has_res_verb_cues"] or flags["has_res_stats"]):
        chosen = "Result" if flags["section"] not in {"DISCUSSION","INTRO","ABSTRACT"} else "Conclusion"
    if chosen == "OTHER" and flags["has_experiment_ops"]:
        chosen = "Experiment"
    if chosen == "Dataset" and flags["has_significance"]:
        chosen = "Result"

    return chosen

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
        section_hint = (it["section_hint"] or "").upper()

        # raw scoring by rules
        scores, hits = _score_sentence(doc, section_hint, matcher, depmatcher)

        # structural citation bonus
        if it.get("has_citation"):
            scores["Input Fact"] += CIT_STRUCT_BONUS
            hits["INF_CIT_STRUCT"] = hits.get("INF_CIT_STRUCT", 0) + 1

        # flags, soft boosts, tie-breaks & postfixes
        flags = _compute_flags(doc, hits, section_hint=section_hint, has_citation_field=bool(it.get("has_citation")))
        scores = _apply_boosts(scores, flags)
        label = _resolve_label(scores, flags, had_matches=bool(hits))

        # safeguards
        if looks_like_reference(it["text"]):
            label = "OTHER"

        if (not it.get("has_citation")) and citation_soft and label == "Input Fact":
            non_cit_hits = sum(v for k, v in hits.items()
                               if not (k.startswith("INF_CIT_") or k in {"INF_CITATION","INF_CIT_STRUCT"}))
            cit_hits = sum(v for k, v in hits.items()
                           if (k.startswith("INF_CIT_") or k in {"INF_CITATION","INF_CIT_STRUCT"}))
            if cit_hits > 0 and non_cit_hits == 0:
                label = "OTHER"

        out.append({
            "idx": i,
            "section": it["section_hint"],
            "label": label,
            "text": it["text"],
            "page": it["page"],
            "bbox": it["bbox"],
            "is_caption": it["is_caption"],
            "caption_type": it["caption_type"],
            "scores": scores,
            "matches": dict(hits),
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
