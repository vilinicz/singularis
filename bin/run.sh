#!/usr/bin/env bash
set -euo pipefail

# --- repo layout (relative to this script) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"
PKG_NAME="singularis"
PKG_DIR="${SRC_DIR}/${PKG_NAME}"
OUT_DIR_DEFAULT="${ROOT_DIR}/outputs"

# --- defaults aligned to your tree ---
PDF_DEFAULT="${PKG_DIR}/article10.pdf"
SERVER_DEFAULT="http://localhost:8070"
PATTERNS_DEFAULT="${PKG_DIR}/rules.json"   # you named it rules.json
MODEL_DEFAULT="en_core_web_sm"
CITATION_SOFT_DEFAULT="1"                  # 1=on, 0=off
USE_POETRY_DEFAULT="1"                     # 1=use poetry run, 0=system python

usage() {
  cat <<EOF
Usage: $(basename "$0") [-p PDF] [-s SERVER] [-t RULES_JSON] [-o OUTDIR] [-m MODEL] [--no-citation-soft] [--no-poetry]

Project-aware launcher for the IMRaD pipeline.

Defaults (based on your tree):
  PDF:        ${PDF_DEFAULT}
  SERVER:     ${SERVER_DEFAULT}
  RULES JSON: ${PATTERNS_DEFAULT}
  OUTDIR:     ${OUT_DIR_DEFAULT}
  MODEL:      ${MODEL_DEFAULT}

Examples:
  $(basename "$0")
  $(basename "$0") -p "${PKG_DIR}/article.pdf" -o "${ROOT_DIR}/outputs"
  $(basename "$0") -p "/abs/path/paper.pdf" --no-citation-soft --no-poetry
EOF
}

# --- args ---
PDF="${PDF_DEFAULT}"
SERVER="${SERVER_DEFAULT}"
PATTERNS="${PATTERNS_DEFAULT}"
OUTDIR="${OUT_DIR_DEFAULT}"
MODEL="${MODEL_DEFAULT}"
CITATION_SOFT="${CITATION_SOFT_DEFAULT}"
USE_POETRY="${USE_POETRY_DEFAULT}"

LONG_OPTS=pdf:,server:,patterns:,outdir:,model:,no-citation-soft,no-poetry,help
PARSED=$(getopt -o "p:s:t:o:m:h" --long "$LONG_OPTS" -n "$(basename "$0")" -- "$@") || { usage; exit 2; }
eval set -- "$PARSED"
while true; do
  case "$1" in
    -p|--pdf) PDF="$2"; shift 2 ;;
    -s|--server) SERVER="$2"; shift 2 ;;
    -t|--patterns) PATTERNS="$2"; shift 2 ;;
    -o|--outdir) OUTDIR="$2"; shift 2 ;;
    -m|--model) MODEL="$2"; shift 2 ;;
    --no-citation-soft) CITATION_SOFT="0"; shift ;;
    --no-poetry) USE_POETRY="0"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done

# --- sanity checks on tree ---
[[ -d "$PKG_DIR" ]] || { echo "[error] package dir not found: $PKG_DIR"; exit 1; }
[[ -f "${PKG_DIR}/worker.py" ]] || { echo "[error] ${PKG_DIR}/worker.py not found"; exit 1; }
[[ -f "${PKG_DIR}/parse_pdf_grobid.py" ]] || { echo "[error] ${PKG_DIR}/parse_pdf_grobid.py not found"; exit 1; }
[[ -f "${PKG_DIR}/label_with_spacy.py" ]] || { echo "[error] ${PKG_DIR}/label_with_spacy.py not found"; exit 1; }
[[ -f "$PATTERNS" ]] || { echo "[error] rules file not found: $PATTERNS"; exit 1; }
[[ -f "$PDF" ]] || { echo "[error] PDF not found: $PDF"; exit 1; }

mkdir -p "$OUTDIR"

# --- python command (poetry-aware) ---
if [[ "$USE_POETRY" == "1" && -f "${ROOT_DIR}/pyproject.toml" && $(command -v poetry) ]]; then
  PY_CMD=(poetry run python)
else
  PY_CMD=(python)
fi

# --- check grobid (best effort) ---
if command -v curl >/dev/null 2>&1; then
  if curl -sf "${SERVER%/}/api/isalive" >/dev/null 2>&1; then
    echo "[ok] GROBID alive at ${SERVER}"
  else
    echo "[warn] Can't confirm GROBID at ${SERVER}; continuing..."
  fi
fi

# --- run worker as module with PYTHONPATH=src ---
ENV_PREFIX=(env PYTHONPATH="${SRC_DIR}")
CMD=("${PY_CMD[@]}" -m "${PKG_NAME}.worker" \
  --pdf "$PDF" \
  --server "$SERVER" \
  --patterns "$PATTERNS" \
  --outdir "$OUTDIR" \
  --model "$MODEL")

if [[ "$CITATION_SOFT" == "1" ]]; then
  CMD+=("--citation-soft")
fi

echo "[root]    $ROOT_DIR"
echo "[src]     $SRC_DIR"
echo "[package] $PKG_DIR"
echo "[pdf]     $PDF"
echo "[rules]   $PATTERNS"
echo "[outdir]  $OUTDIR"
echo "[python]  ${PY_CMD[*]}"
echo "[run]     ${ENV_PREFIX[*]} ${CMD[*]}"

("${ENV_PREFIX[@]}" "${CMD[@]}")

echo
echo "[done] Outputs in ${OUTDIR}:"
echo "  - s0.json"
echo "  - s1.json  (JSONL)"
echo "  - $(basename "${PDF%.*}").tei.xml"
