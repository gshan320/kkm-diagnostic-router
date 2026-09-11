"""Central configuration: filesystem layout, chunking, embeddings, LLM."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# backend/app/config.py -> backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")

# ---------------------------------------------------------------- filesystem
DATA_DIR = BACKEND_DIR / "data"
RAW_PDF_DIR = DATA_DIR / "raw_pdfs"
RAW_JSON_DIR = DATA_DIR / "raw_json"
CHROMA_DIR = DATA_DIR / "chroma_db"
FUKKM_JSON = RAW_JSON_DIR / "fukkm_database.json"

COLLECTION_NAME = "kkm_corpus"

# ---------------------------------------------------------------- embeddings
# MedEmbed-large-v0.1: 1024 dims, max_seq_length 512, fine-tuned for medical
# retrieval. Replaces all-MiniLM-L6-v2, which could not bridge symptom text to
# disease vocabulary - measured, the TB CPG sat at cosine 0.306 for "pulmonary
# tuberculosis sputum smear" but 0.564 for "chronic cough haemoptysis night
# sweats" and fell out of the top 60 entirely. Full HF repo id: the model is
# not under the sentence-transformers/ org.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "abhinand/MedEmbed-large-v0.1")
EMBEDDING_MAX_TOKENS = int(os.getenv("EMBEDDING_MAX_TOKENS", "512"))

# 512 is lossless now that the embedder accepts 512 tokens, so a graded
# recommendation stays whole with its evidence level instead of being split.
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "512"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))

# ------------------------------------------------------ hybrid retrieval
# Dense and BM25 each propose CANDIDATE_K chunks; the two ranked lists are
# fused with Reciprocal Rank Fusion, then a cross-encoder reranks the union
# down to the k the caller asked for. BM25 is the deterministic channel: it
# matches "haemoptysis" as a string, which is both more reliable and more
# explainable than any embedding score.
CANDIDATE_K = int(os.getenv("CANDIDATE_K", "30"))
RRF_K = int(os.getenv("RRF_K", "60"))
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "1") not in ("0", "false", "False")

# Cross-encoder reads query and chunk together, so it can judge interactions a
# bi-encoder structurally cannot ("calf swelling" + "pleuritic chest pain" -> PE).
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") not in ("0", "false", "False")

# ------------------------------------------------------------ generation speed
# Measured on the paediatric dengue case, Qwen3-8B-4bit on an M4 Air:
# prefill 18,780 tok @ 135 tok/s = 139s, decode 1,518 tok @ 9.9 tok/s = 153s.
# Roughly half the wall clock is decode, so context cuts alone cannot fix it.
#
# KV_BITS quantises the KV cache: less memory bandwidth per decode step, which
# is what a long context actually costs on this hardware. Set 0 to disable.
KV_BITS = int(os.getenv("KV_BITS", "8"))
# Speculative decoding: a small model of the SAME family drafts tokens the 8B
# verifies in one pass. Empty string disables it.
DRAFT_MODEL = os.getenv("DRAFT_MODEL", "")

# ------------------------------------------------------ red-flag recall floor
# Layers 1-3 make a miss rare; only this makes it impossible for the listed
# presentations. Rules live in red_flags.py and only ever ADD sources.
RED_FLAG_CHUNKS_PER_RULE = int(os.getenv("RED_FLAG_CHUNKS_PER_RULE", "3"))
RED_FLAG_MAX_CHUNKS = int(os.getenv("RED_FLAG_MAX_CHUNKS", "15"))
RED_FLAGS_ENABLED = os.getenv("RED_FLAGS_ENABLED", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------- LLM
# Local MLX model on Apple Silicon. Weights are pulled from Hugging Face on
# first use and cached under ~/.cache/huggingface; inference is fully offline.
LOCAL_MLX_MODEL = os.getenv(
    "LOCAL_MLX_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit"
)
# 4000 is ample for a full DiagnosticSchema. The old 16000 bought nothing but
# wall-clock: a 3B model does not produce 16k tokens of clinical value.
MAX_TOKENS_TRIAGE = int(os.getenv("MAX_TOKENS_TRIAGE", "4000"))
MAX_TOKENS_INQUIRY = int(os.getenv("MAX_TOKENS_INQUIRY", "32000"))

# ------------------------------------------------------------------ retrieval
# How many chunks each of the three Mode A sub-retrievals pulls.
TRIAGE_K = int(os.getenv("TRIAGE_K", "8"))
# Discriminator cells (one per MTS table cell) pulled in by metadata filter for
# the triage level the vitals justify and its immediate neighbours.
MTS_CELL_K = int(os.getenv("MTS_CELL_K", "8"))
CPG_K = int(os.getenv("CPG_K", "10"))
DRUG_K = int(os.getenv("DRUG_K", "8"))
# Extra pass over the Paediatric Protocols when the patient is a child.
PAEDS_K = int(os.getenv("PAEDS_K", "6"))
PAEDIATRIC_AGE_YEARS = float(os.getenv("PAEDIATRIC_AGE_YEARS", "12"))
# Mode B takes its depth from the request (InquiryRequest.max_sources).

# ------------------------------------------------------------------ doc types
DOC_TYPE_TRIAGE = "TRIAGE_PROTOCOL"
DOC_TYPE_CPG_FULL = "CPG_FULL"
DOC_TYPE_CPG_QR = "CPG_QUICK_REFERENCE"
DOC_TYPE_PAEDS = "PAEDIATRIC_PROTOCOL"
DOC_TYPE_DRUG = "DRUG_FORMULARY"

# Added 2026-09-11 for two MOH documents that are authoritative but are NOT
# clinical guidelines, and must not be treated as one.
#
# HTA_REPORT - a MaHTAS Health Technology Assessment. It appraises whether MOH
#   should adopt a technology: systematic-review prose, AUROCs, odds ratios,
#   cost-effectiveness, staff attitudes. The NEWS report states its scoring
#   chart on ONE page out of 122; the rest is evidence about the score, not
#   instructions for a patient. A sentence like "NEWS >= 5, 30-day mortality
#   OR 11.8" is a study finding, and nothing in it should ever ground a dose or
#   support a drug indication.
#
# PATIENT_FLOW_POLICY - who may be redirected away from an ETD and who may not.
#   It governs disposition, not diagnosis, and the one in the corpus is a STATE
#   policy (Selangor), so it must never be quoted as a national rule.
DOC_TYPE_HTA = "HTA_REPORT"
DOC_TYPE_POLICY = "PATIENT_FLOW_POLICY"

# The types a drug recommendation may be grounded in. Membership here is what
# `_check_drug_indications` and `_dose_sentence` read: a doc_type left out
# cannot support a drug or supply a dose, which is the whole reason the two
# types above exist separately.
CLINICAL_DOC_TYPES = [
    DOC_TYPE_CPG_FULL,
    DOC_TYPE_CPG_QR,
    DOC_TYPE_PAEDS,
]

# Authoritative, retrievable, and deliberately outside CLINICAL_DOC_TYPES.
ADJUNCT_DOC_TYPES = [
    DOC_TYPE_HTA,
    DOC_TYPE_POLICY,
]

# CORS origins for the Next.js dev server
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if o.strip()
]
