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
# A4.1 measured the cross-encoder at ~20 s of every report (14 calls x 30
# pairs x 512 tokens on this machine). It orders passages within documents the
# topic gate has already chosen, so it reranks a smaller pool of shorter inputs.
# 15 x 320 measured ~12 s faster per report and kept every key page (rhabdo
# Heat p10/p9/p6, Hyperkalaemia p17; ACS aspirin, ticagrelor, clopidogrel).
RERANK_POOL = int(os.getenv("RERANK_POOL", "15"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "320"))

# ------------------------------------------------------------ generation speed
# Measured on the paediatric dengue case, Qwen3-8B-4bit on an M4 Air:
# prefill 18,780 tok @ 135 tok/s = 139s, decode 1,518 tok @ 9.9 tok/s = 153s.
# Roughly half the wall clock is decode, so context cuts alone cannot fix it.
#
# KV_BITS quantises the KV cache: less memory bandwidth per decode step, which
# is what a long context actually costs on this hardware. Set 0 to disable.
KV_BITS = int(os.getenv("KV_BITS", "8"))
# Second pass (rag_engine RagEngine._second_stage): when the completeness
# checklist finds at least TWO_STAGE_MIN_GAPS required elements missing, retrieve
# again for the working DIAGNOSIS and ask for the missing items only. Costs one
# more, shorter generation on those cases. TWO_STAGE=0 turns it off.
TWO_STAGE = os.getenv("TWO_STAGE", "1") not in ("0", "false", "False")
TWO_STAGE_MIN_GAPS = int(os.getenv("TWO_STAGE_MIN_GAPS", "2"))
# F5: "quote" fills missing checklist elements with the verbatim source sentence
# (no model call); "llm" is the old second pass. Run 10's LLM pass took 62 s and
# re-wrote only items already in the plan.
STAGE2_MODE = os.getenv("STAGE2_MODE", "quote")
TWO_STAGE_CHUNKS = int(os.getenv("TWO_STAGE_CHUNKS", "6"))
MAX_TOKENS_STAGE2 = int(os.getenv("MAX_TOKENS_STAGE2", "1500"))

# Keep the KV cache of the fixed system prompt between requests (rag_engine
# _generate). Set 0 to prefill the whole prompt every time.
PROMPT_CACHE = os.getenv("PROMPT_CACHE", "1") not in ("0", "false", "False")
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

# EXTERNAL_GUIDELINE - a non-KKM clinical guideline, indexed ONLY when an
#   indexed KKM document cites it (source_policy.py; the citation's title, page
#   and words are recorded in data/source_qualifications.json and checked by
#   the tests). Its filename carries an "EXT " prefix, so every citation, dose
#   note and governing-guideline line shows it is not KKM. The first one,
#   CHAMP 2025 (US DoD, exertional rhabdomyolysis), was removed on 2026-09-30:
#   no KKM document cites it.
DOC_TYPE_EXTERNAL = "EXTERNAL_GUIDELINE"

# The types a drug recommendation may be grounded in. Membership here is what
# `_check_drug_indications` and `_dose_sentence` read: a doc_type left out
# cannot support a drug or supply a dose, which is the whole reason the two
# types above exist separately.
CLINICAL_DOC_TYPES = [
    DOC_TYPE_CPG_FULL,
    DOC_TYPE_CPG_QR,
    DOC_TYPE_PAEDS,
    DOC_TYPE_EXTERNAL,
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

# ------------------------------------------------------- audit + GPU queue
# Every report is appended to a local SQLite store (see audit.py). The rows hold
# the intake as typed - patient data in real use - so the file lives under
# data/audit/, which is git-ignored. AUDIT_ENABLED=0 turns recording off.
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "1") not in ("0", "false", "False")
AUDIT_DB = Path(os.getenv("AUDIT_DB", str(DATA_DIR / "audit" / "audit.db")))

# How many runs may wait for the reasoning model before new ones are refused
# with "busy, try again". One run takes ~2-4 minutes here, so 8 waiting is
# already a half-hour line; letting it grow further only hides the problem.
GPU_QUEUE_MAX = int(os.getenv("GPU_QUEUE_MAX", "8"))

# ------------------------------------------------------- attention layer (A1)
# TRIAGE_CONTEXT: "code" puts a two-line code-decided TRIAGE block in the prompt
# instead of MTS table chunks; "chunks" restores the old behaviour.
TRIAGE_CONTEXT = os.getenv("TRIAGE_CONTEXT", "code")
# Formulary rows by drug NAME from the kept guideline text (1), or by embedding
# the treatment sentences (0, the pre-A1 behaviour).
FORMULARY_BY_NAME = os.getenv("FORMULARY_BY_NAME", "1") not in ("0", "false", "False")
# KEY FINDINGS block at the top of the prompt and a reminder before the task.
FOCUS_BLOCK = os.getenv("FOCUS_BLOCK", "1") not in ("0", "false", "False")

# ------------------------------------------------------- attention layer (A2)
# A short model call names the working differential before retrieval; with
# HYPOTHESES=0 only the red-flag rules name it.
HYPOTHESES = os.getenv("HYPOTHESES", "1") not in ("0", "false", "False")
MAX_TOKENS_HYPOTHESES = int(os.getenv("MAX_TOKENS_HYPOTHESES", "160"))
# Per-query depth, total clinical passages, and the cap per document.
HYPOTHESIS_QUERY_K = int(os.getenv("HYPOTHESIS_QUERY_K", "6"))
CLINICAL_TOTAL = int(os.getenv("CLINICAL_TOTAL", "12"))
MAX_PER_DOC = int(os.getenv("MAX_PER_DOC", "6"))
# FUKKM rows put in front of the model (by name, from the kept guideline text).
FORMULARY_ROWS = int(os.getenv("FORMULARY_ROWS", "6"))
# Drop documents about a condition neither in the intake nor among the
# working hypotheses (and not forced by a red-flag rule).
TOPIC_GATE = os.getenv("TOPIC_GATE", "1") not in ("0", "false", "False")

# ------------------------------------------------------- attention layer (A3)
# Keep each clinical passage's most relevant sentences (plus any sentence that
# names a drug with a dose), order passages edge-first, fit a token budget.
DISTILL = os.getenv("DISTILL", "1") not in ("0", "false", "False")
DISTILL_SENTENCES = int(os.getenv("DISTILL_SENTENCES", "5"))
CONTEXT_BUDGET_TOKENS = int(os.getenv("CONTEXT_BUDGET_TOKENS", "3500"))
# Reference-list pages (17.5% of clinical chunks) are skipped at retrieval and
# dropped before the prompt; their paper titles read as treatment text.
DROP_BIBLIOGRAPHY = os.getenv("DROP_BIBLIOGRAPHY", "1") == "1"
# A red-flag rule's secondary document (a cause or complication, not the rule's
# own guideline and not a hypothesis) keeps at most this many passages.
# Formulary-by-name drops drugs weighted under this share of the top drug.
FORMULARY_MIN_SHARE = float(os.getenv("FORMULARY_MIN_SHARE", "0.2"))
# Build models, indexes and the system-prompt cache at server start, so the
# first report is as fast as the rest (see RagEngine.warm_up).
WARMUP = os.getenv("WARMUP", "1") == "1"
# A4.4: the model no longer writes what code derives exactly - vital-sign rows
# (MTS table), prescriber categories (FUKKM record). 0 restores the old template.
OUTPUT_DIET = os.getenv("OUTPUT_DIET", "1") == "1"
# A4.2: share of an item's (rarity-weighted) terms one cited sentence must hold.
CITATION_SUPPORT_MIN = float(os.getenv("CITATION_SUPPORT_MIN", "0.3"))
FORCED_SECONDARY_PASSAGES = int(os.getenv("FORCED_SECONDARY_PASSAGES", "2"))

# ------------------------------------------------------- condition cards
# data/condition_cards.json (cards_build.py): per-condition KKM sentences for
# every condition the corpus gives management for. They drive the completeness
# checklist, complications, source cautions, the quoted disposition floor and
# the "KKM is silent" statements wherever no hand-written rule exists.
CARDS = os.getenv("CARDS", "1") not in ("0", "false", "False")
# Passages from the likeliest hypothesis's card put in front of the model: the
# chunks holding its core first-hours items, so management for the diagnosis is
# in the prompt even when the complaint's wording retrieved something else.
CARD_PASSAGES = int(os.getenv("CARD_PASSAGES", "3"))
# F5: at most this many checklist gaps are filled from source per report.
QUOTED_GAP_MAX = int(os.getenv("QUOTED_GAP_MAX", "3"))
