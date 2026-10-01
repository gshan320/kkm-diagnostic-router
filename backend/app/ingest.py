"""Ingestion engine: KKM PDFs + FUKKM JSON -> ChromaDB.

Usage
-----
    python -m app.ingest              # incremental upsert (deterministic IDs)
    python -m app.ingest --reset      # drop the collection first
    python -m app.ingest --dry-run    # parse + chunk + report, no embedding
    python -m app.ingest --stats      # what is already indexed

Every chunk carries: filename, cpg_title, edition_year, edition, doc_type,
page_number (PDFs) or drug fields (FUKKM), version_date, status, plus chunk_index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pymupdf

from . import config, source_policy

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", stream=sys.stdout
)
log = logging.getLogger("ingest")

BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Boilerplate that pollutes chunks. The MTS 2022 copy in raw_pdfs/ is a
# Studocu re-host, so every page carries its watermark furniture.
# ---------------------------------------------------------------------------
NOISE_PATTERNS = [
    re.compile(r"^lOMoARcPSD\|?\d*$"),
    re.compile(r"^messages\.[a-z_]+$"),
    re.compile(r"^Downloaded by .*$", re.IGNORECASE),
    re.compile(r"^Learning Skills For Open Distance Learners.*$", re.IGNORECASE),
    re.compile(r"^Malaysian Triage Scale New Revised 2022 \d+ \d+$"),
    re.compile(r"^studocu.*$", re.IGNORECASE),
    re.compile(r"^This document is available free of charge on.*$", re.IGNORECASE),
    # Running headers of the MOH AMO Standard Practice Guidelines (2023).
    re.compile(r"^(?:Clinical|Standard Practice) Guidelines For AMO in EMTS MOH$", re.IGNORECASE),
]

ACRONYM_TITLES = {"MTS": "Malaysian Triage Scale"}

# A one-line label prepended to EVERY chunk of a document type that is
# authoritative but is not a clinical guideline. It travels with the chunk into
# the prompt, so the model is told what it is reading at the point it reads it -
# the doc_type in the source header says HTA_REPORT, and nothing tells the model
# what an HTA report is or is not for.
#
# Config keeps these types out of CLINICAL_DOC_TYPES, which already stops them
# grounding a drug or a dose in code. This is the other half: stopping them
# being READ as clinical instruction.
DOC_TYPE_CAUTION = {
    config.DOC_TYPE_HTA: (
        "SOURCE TYPE - HEALTH TECHNOLOGY ASSESSMENT. An MOH evidence appraisal "
        "OF a tool, not a clinical guideline. Numbers in this document (AUROC, "
        "odds ratios, mortality and admission rates) are findings from published "
        "studies about the tool's performance, NOT instructions for this "
        "patient. It cannot set a triage level, support a drug or supply a dose."
    ),
    config.DOC_TYPE_POLICY: (
        "SOURCE TYPE - STATE PATIENT-FLOW POLICY, not a national one. It governs "
        "disposition and redirection out of the department, never diagnosis, "
        "treatment or triage level. Say which state it binds if you rely on it."
    ),
}

# Document-wide cautions, keyed by a filename substring. Same idea as
# DOC_TYPE_CAUTION but for a fact about ONE document rather than a whole type.
#
# Pain Management in ETD 2020 grades pain differently from MTS 2022, and the
# disagreement sits at a single score:
#
#     score      Pain Management in ETD 2020 (p10)   MTS 2022 (p7)
#     1 - 3      MILD, green zone                    Level 4
#     4 - 6      MODERATE, yellow zone               Level 3
#     7          SEVERE, red zone                    Level 3  <-- disagree
#     8 - 10     SEVERE, red zone                    Level 2
#
# Both are current MOH documents and BOTH ARE RIGHT, because they answer
# different questions: MTS grades how long a patient may wait, the pain
# guideline grades which analgesia they get. The danger is that the pain
# guideline states its bands in ZONE COLOURS - "GREEN ZONE", "YELLOW ZONE",
# "RED ZONE" - and MTS levels also carry colours, so "SEVERE PAIN, Pain Score
# 7-10, RED ZONE" reads exactly like an instruction to triage a pain score of 7
# as MTS red. It is not; it is an instruction to give morphine.
#
# Worth recording why this matters more than a normal conflict: the hand-typed
# prompt block replaced on 2026-09-11 said "Severe Acute Pain (Pain Score 7-10)"
# at Level 2 and "Moderate Pain (Pain Score 4-6)" at Level 3. Those are THIS
# document's bands, not the MTS grid's. The drift was not random - it was
# another MOH document's numbers, which is exactly how this kind of mistake
# arrives and why it survived so long.
DOC_CAUTIONS: dict[str, str] = {
    "Pain Management in Emergency and Trauma Department": (
        "CAUTION ON PAIN BANDS - this document grades pain for ANALGESIA, not "
        "for triage, and its bands differ from the Malaysian Triage Scale at a "
        "score of 7: it calls 7-10 SEVERE, while MTS 2022 page 7 prints "
        "'Severe Pain (8 - 10)' at Level 2 and 'Pain Score 4 - 7' at Level 3. "
        "Its GREEN / YELLOW / RED ZONE labels are ANALGESIA zones and are NOT "
        "MTS triage levels or colours. Use this document for which analgesia to "
        "give; use MTS for the triage level. Never take a triage level from it."
    ),
}

# Page-specific cautions, keyed by (filename substring, page number).
#
# Same principle as IMPLICIT_LEVEL_COLUMNS: a fact about one document that the
# extractor cannot infer and that changes how the page must be read.
#
# The NEWS HTA report prints the NEWS (2012) and NEWS2 (2017) parameter charts
# SIDE BY SIDE on one page, and the text layer flattens them into a single run
# with the two version labels emitted AFTER both tables. The result is one chunk
# offering two different threshold sets for the same six vital signs - a
# respiratory rate of 22 scores 0 on the first chart and 2 on the second - with
# nothing in the text saying which row belongs to which version.
PAGE_CAUTIONS: dict[tuple[str, int], str] = {
    ("National Early Warning Score", 23): (
        "CAUTION - THIS PAGE PRINTS TWO DIFFERENT VERSIONS OF THE SCORE SIDE BY "
        "SIDE: NEWS (2012) on the left and NEWS2 (2017) on the right. The text "
        "below interleaves them and the version labels are separated from their "
        "tables, so the thresholds in it CANNOT be attributed to one version or "
        "the other. Do not compute or quote a NEWS score from this page."
    ),
}

WORD_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
VERSION_DATE_RE = re.compile(r"v(20\d{2})(\d{2})(\d{2})")
NUM_EDITION_RE = re.compile(r"(\d+)\s*(?:st|nd|rd|th)\s*Edition", re.IGNORECASE)
# A year that sits next to the word "Edition" is the edition year; a bare year
# elsewhere in the body is usually a cited reference ("National Antibiotic
# Guidelines 2019") or a dose ("2000 mg OD").
EDITION_YEAR_RE = re.compile(
    r"(?:(?P<before>19\d{2}|20\d{2})\s*[(\-\u2013,]?\s*(?:\w+\s+){0,2}Edition"
    r"|Edition\)?[\s,\-\u2013:]{0,4}(?P<after>19\d{2}|20\d{2}))",
    re.IGNORECASE,
)
WORD_EDITION_RE = re.compile(
    r"\b(" + "|".join(WORD_ORDINALS) + r")\s+Edition", re.IGNORECASE
)
# FUKKM prescriber-category codes: A, A*, B, C, C+, A/KK, and comma combos.
CATEGORY_CODE_RE = re.compile(
    r"^\s*(?:A\*?|B|C\+?|KK|A/KK|A\*/KK)(?:\s*,\s*(?:A\*?|B|C\+?|KK|A/KK|A\*/KK))*\s*$"
)
ATC_RE = re.compile(r"^([A-Z]\d{2}[A-Z]{2}\d{2})")


# ===========================================================================
# Data model
# ===========================================================================


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


@dataclass
class DocMeta:
    filename: str
    cpg_title: str
    edition_year: str
    edition: str
    doc_type: str
    version_date: str = ""


@dataclass
class IngestReport:
    pdf_files: int = 0
    pdf_pages: int = 0
    pdf_chunks: int = 0
    drug_records: int = 0
    drug_chunks: int = 0
    skipped_pages: int = 0
    warnings: list[str] = field(default_factory=list)
    per_document: dict = field(default_factory=dict)


# ===========================================================================
# Text cleaning + tokenised chunking
# ===========================================================================


def clean_text(raw: str) -> str:
    """Drop watermark furniture, collapse the ligatures PyMuPDF leaves behind."""
    # A soft hyphen (U+00AD) marks where a word MAY break; the CHAMP 2025 PDF
    # carries 438 of them, each splitting a word ("Mo\xadnitor") for BM25,
    # the embedder and every citation check.
    raw = re.sub(r"\xad\s*", "", raw or "")
    out: list[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(p.match(s) for p in NOISE_PATTERNS):
            continue
        out.append(s)
    text = "\n".join(out)
    # Private Use Area glyphs are Wingdings/Symbol bullets embedded by the PDF
    # producer. They survive extraction and render as tofu boxes in the report.
    text = re.sub(r"[\uE000-\uF8FF\uFFF0-\uFFFF]", " ", text)
    text = (
        text.replace("ﬁ", "fi").replace("ﬂ", "fl")
        .replace("‘", "'").replace("’", "'")
        .replace("“", '"').replace("”", '"')
        .replace("•", "- ").replace("‣", "- ").replace("▪", "- ")
    )
    return re.sub(r"[ \t]{2,}", " ", text).strip()


_WORDS: set[str] | None = None


def _known_words() -> set[str]:
    """Whole-word tokens of the embedding model's vocabulary - what counts as
    a word on its own when deciding whether a line break split one."""
    global _WORDS
    if _WORDS is None:
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
            vocab = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL).get_vocab()
            _WORDS = {w for w in vocab if w.isalpha() and not w.startswith("##")}
        except Exception as exc:  # noqa: BLE001
            log.warning("No tokenizer vocabulary for word re-joining (%s); skipping it.", exc)
            _WORDS = set()
    return _WORDS


_BROKEN3 = re.compile(r"\b([A-Za-z]{2,})\n([a-z]{2,})\n([a-z]{2,})\b")
_BROKEN2 = re.compile(r"\b([A-Za-z]{2,})\n([a-z]{2,})\b")


def rejoin_broken_words(pages: list[str]) -> tuple[list[str], int]:
    """Re-join words a PDF layout broke across lines without a hyphen.

    The CHAMP 2025 algorithm pages extract as "diagno\nstic", "phos\npho\nrus".
    Two (or three) line-break fragments are joined only when BOTH hold:
      - the joined word appears whole elsewhere in the same document, and
      - at least one fragment is not a word on its own (so "in\nformation"
        and "heat\nstroke" stay two words).
    Returns (pages, number of joins)."""
    known = _known_words()
    if not known:
        return pages, 0
    whole: dict[str, int] = {}
    for t in pages:
        for w in re.findall(r"[a-z]{4,}", t.lower()):
            whole[w] = whole.get(w, 0) + 1
    joins = 0

    def ok(parts: tuple[str, ...]) -> bool:
        word = "".join(parts).lower()
        return whole.get(word, 0) >= 1 and any(p.lower() not in known for p in parts)

    def fix3(m: re.Match) -> str:
        nonlocal joins
        if ok(m.groups()):
            joins += 1
            return "".join(m.groups())
        return m.group(0)

    def fix2(m: re.Match) -> str:
        nonlocal joins
        if ok(m.groups()):
            joins += 1
            return "".join(m.groups())
        return m.group(0)

    out = [_BROKEN2.sub(fix2, _BROKEN3.sub(fix3, t)) for t in pages]
    return out, joins


class Tokenizer:
    """Real MiniLM tokenizer when available; whitespace estimate otherwise."""

    def __init__(self) -> None:
        self._hf = None
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415

            self._hf = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
            log.info("Chunking with the %s tokenizer.", config.EMBEDDING_MODEL)
        except Exception as exc:  # pragma: no cover - offline / no transformers
            log.warning(
                "Falling back to whitespace token estimation (%s: %s)",
                type(exc).__name__,
                exc,
            )

    def split(self, text: str, size: int, overlap: int) -> list[str]:
        if not text:
            return []
        if self._hf is not None:
            ids = self._hf.encode(text, add_special_tokens=False)
            if len(ids) <= size:
                return [text]
            step = max(1, size - overlap)
            return [
                self._hf.decode(ids[i : i + size], skip_special_tokens=True).strip()
                for i in range(0, len(ids), step)
                if ids[i : i + size]
            ]
        # ~1.3 wordpiece tokens per whitespace word for clinical English
        words = text.split()
        wsize = max(1, int(size / 1.3))
        woverlap = max(0, int(overlap / 1.3))
        if len(words) <= wsize:
            return [text]
        step = max(1, wsize - woverlap)
        return [
            " ".join(words[i : i + wsize]).strip()
            for i in range(0, len(words), step)
            if words[i : i + wsize]
        ]


# ===========================================================================
# PDF metadata parsing
# ===========================================================================


# TRIAGE_PROTOCOL is a privileged type, not a topic label: it is the only type
# whose tables are mined into discriminator cells, and the only one
# discriminator_cells() will serve as the triage grid. `\bMTS\b` replaces a bare
# `"mts" in filename` substring test for the reason recorded against the
# `\bpregnan` trap in the triage modifiers - an unanchored needle eventually
# matches a word nobody thought of.
TRIAGE_FILENAME_RE = re.compile(r"\bMTS\b|\btriage\b", re.IGNORECASE)
HTA_FILENAME_RE = re.compile(r"\bMaHTAS\b|health technology assessment", re.IGNORECASE)
POLICY_FILENAME_RE = re.compile(r"redirection polic|patient flow", re.IGNORECASE)


def classify_doc(filename: str) -> str:
    low = filename.lower()
    # The "CPG " and "QR " prefixes are the corpus convention and the strongest
    # signal there is, so they are tested FIRST. They have to be: nearly every
    # MOH clinical guideline is published BY MaHTAS and says so on its cover,
    # and "CPG Early Management of Head Injury ... MaHTAS.pdf" is a filename a
    # future download could plausibly arrive with. Without this order the
    # HTA test below would type a clinical guideline as an evidence appraisal
    # and quietly drop it out of CLINICAL_DOC_TYPES - it could then no longer
    # ground a drug or a dose, and nothing would report that it had stopped.
    # "EXT " first: a non-KKM guideline must never be typed as a KKM CPG, and
    # its title may well contain "Clinical Practice Guideline".
    if low.startswith("ext "):
        return config.DOC_TYPE_EXTERNAL
    if low.startswith("qr ") or "quick reference" in low:
        return config.DOC_TYPE_CPG_QR
    if low.startswith("cpg "):
        return config.DOC_TYPE_CPG_FULL
    # Tested BEFORE the triage rule on purpose. Both are documents ABOUT triage
    # decisions - a redirection policy is triage-away criteria, and the NEWS HTA
    # report says "triage" on most of its pages - so either would otherwise be
    # typed TRIAGE_PROTOCOL and have its tables mined as MTS levels. Only
    # MTS 2022 may carry that type.
    if HTA_FILENAME_RE.search(filename):
        return config.DOC_TYPE_HTA
    if POLICY_FILENAME_RE.search(filename):
        return config.DOC_TYPE_POLICY
    if TRIAGE_FILENAME_RE.search(filename):
        return config.DOC_TYPE_TRIAGE
    if "paediatric-protocol" in low or "paediatric protocol" in low:
        return config.DOC_TYPE_PAEDS
    return config.DOC_TYPE_CPG_FULL


def parse_edition(text: str) -> str:
    if m := NUM_EDITION_RE.search(text):
        return f"{int(m.group(1))}"
    if m := WORD_EDITION_RE.search(text):
        return f"{WORD_ORDINALS[m.group(1).lower()]}"
    return ""


def parse_edition_year(filename: str, head_text: str, full_text: str = "") -> str:
    """Explicit year in the filename wins, then a year on the cover/title pages,
    then a year adjacent to the word "Edition" anywhere in the body, then the
    `vYYYYMMDD` publication-version stamp."""
    stem = Path(filename).stem
    if m := YEAR_RE.search(VERSION_DATE_RE.sub("", stem)):
        return m.group(1)
    for y in YEAR_RE.findall(head_text):
        if 1990 <= int(y) <= 2035:
            return y  # cover / title-page year
    for m in EDITION_YEAR_RE.finditer(full_text):
        year = m.group("before") or m.group("after")
        if year and 1990 <= int(year) <= 2035:
            return year
    if m := VERSION_DATE_RE.search(stem):
        return m.group(1)
    return ""


def parse_version_date(filename: str) -> str:
    """Extracts the version date stamp if present (e.g. v20240131 -> 2024-01-31)."""
    if m := VERSION_DATE_RE.search(filename):
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return ""


def parse_title(filename: str) -> str:
    stem = Path(filename).stem
    stem = VERSION_DATE_RE.sub("", stem)
    stem = re.sub(r"^(?:QR|EXT)\s+", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"^CPG\s+", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\((?:[^()]*edition[^()]*)\)", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\b\d+\s*(?:st|nd|rd|th)\s+Edition\b", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\bPDF\s*Final\b", "", stem, flags=re.IGNORECASE)
    stem = stem.replace("-", " ").replace("_", " ")
    stem = re.sub(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", "", stem)
    stem = re.sub(r"\bv?\d+\.\d+\b", "", stem)
    stem = re.sub(r"\s{2,}", " ", stem).strip(" .,-")
    # Trailing file-copy counters ("...-Hospital-1") are not part of the title,
    # but "Type 1"/"Type 2"/"Stage 3" are.
    stem = re.sub(
        r"(?<!\b[Tt]ype)(?<!\b[Ss]tage)(?<!\b[Ll]evel)(?<!\b[Cc]lass)\s+\d{1,2}$",
        "",
        stem,
    )
    stem = stem.strip()
    # Short filename acronyms, expanded to the document's own title text.
    return ACRONYM_TITLES.get(stem.upper(), stem) or Path(filename).stem


def describe_pdf(path: Path, head_text: str, full_text: str = "") -> DocMeta:
    filename = path.name
    return DocMeta(
        filename=filename,
        cpg_title=parse_title(filename),
        edition_year=parse_edition_year(filename, head_text, full_text),
        # Cover pages only. Scanning full_text scrapes the bibliography instead
        # ("ATLS Student Course Manual (9th Edition)") and stamps the document
        # with an edition it does not have.
        edition=parse_edition(filename) or parse_edition(head_text),
        doc_type=classify_doc(filename),
        version_date=parse_version_date(filename),
    )


# ===========================================================================
# PDF -> chunks
# ===========================================================================


def _doc_key(filename: str) -> str:
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------- MTS discriminator tables
# The triage document is a GRID, and the ordinary text path destroys it. Page 7
# comes out as four level-columns joined by the same " - " that separates the
# bullets inside one column:
#
#   severe pain ( 8 - 10 ) - gcs < 13 or drop > 2 - temp 37. 5 - 39 c - pain
#   score 4 - 7 - appears unwell - history of fever - no documented fever -
#   no fever - no pain
#
# Nothing recoverable there says "No Pain" belongs to Level 5 and "Pain Score
# 4 - 7" to Level 3. find_tables() recovers 18 tables across the 19 pages with
# their structure intact, so a triage document additionally yields ONE CHUNK PER
# DISCRIMINATOR CELL, carrying mts_level / cohort / row-label metadata. That
# turns "hope the right column ranks" into a metadata filter.
#
# These chunks are ADDED, never substituted: the prose pass still runs over the
# same pages, because the policy paragraphs that matter most - the triage-away
# pathway on page 6, the paediatric rules on page 13 - sit outside the tables.

LEVEL_COL_RE = re.compile(r"LEVEL\s*([1-5])", re.IGNORECASE)

# COMPLAINTS LIST (ADULT), pages 8-11, prints its level columns with NO header
# row - the caption spans the table and the columns are left implicit. Skipping
# it costs the single largest source of Level 5 criteria in the document, so the
# mapping is stated here instead, and every cell it produces is marked
# header_inferred=True.
#
# Evidence that column 1 is Level 2 and column 4 is Level 5, not Levels 1-4:
#   * the ADULT table on page 7 is laid out L2 / L3 / L4 / L5, and this table
#     has the same five-column shape;
#   * column 1 of the ABDOMINAL PAIN row reads "Pain Score 8 - 10", which page 7
#     places at Level 2, not Level 1;
#   * column 4 reads "All other presentations" (EAR / ENT) and "Comfortable /
#     Local Swelling / Itchiness" (ALLERGY) - Routine, not Early Care.
# The paediatric complaints list on pages 15-18 DOES print its header, and runs
# L1 to L4, so it needs no entry here.
IMPLICIT_LEVEL_COLUMNS: dict[str, dict[int, int]] = {
    "COMPLAINTS LIST (ADULT)": {1: 2, 2: 3, 3: 4, 4: 5},
}
# Pages 4-5 print CRITICAL FIRST LOOK and RAPID ASSESSMENT with FOUR output
# columns: LEVEL 1 / LEVEL 2 / LEVEL 3 / SECONDARY TRIAGE. The last one is not
# a triage level - it is the clearance criteria that let a patient leave Primary
# Triage with no escalation ("Walking", "SpO2 > 94%", "Alert, sit upright").
#
# Until 2026-09-11 every non-level column was swept into key_cols, so those
# criteria were concatenated onto the ROW LABEL of the Level 1/2/3 cells beside
# them. A resuscitation cell read:
#
#   Discriminator: SHOCK STATE - Peripheries - Pulses - AVPU
#                  / - Warm, pink, pulses normal - Alert, walking
#   MTS Level 1 (RESUSCITATION) criteria: - Pale, cyanosed, cold peripheries ...
#
# These cells are FORCE-FETCHED by discriminator_cells() on every Level 1-2
# case, so the model was handed a self-contradicting chunk. The row label now
# comes from the columns that LEAD the table, and a trailing non-level column
# becomes its own chunk at MTS_LEVEL_CLEARANCE.
#
# Level 0 is not a level the document defines. It is used here precisely
# because discriminator_cells() filters on range(level-1, level+2), which is
# bounded at 1: a clearance cell can never be served as a triage criterion.
MTS_LEVEL_CLEARANCE = 0
CLEARANCE_LABEL = "CLEARANCE"

COHORT_PAGE_RE = re.compile(r"\bPAEDIATRIC|\bPAEDS\b", re.IGNORECASE)
COHORTS = (("PAEDIATRIC", "PAEDIATRIC"), ("ADULT", "ADULT"))
MTS_LEVEL_NAMES = {
    1: "RESUSCITATION", 2: "EMERGENCY", 3: "URGENT", 4: "EARLY_CARE", 5: "ROUTINE",
}


# "DEHYDRA-\nTION" and "GENITO-\nURINARY" are row labels wrapped mid-word by the
# PDF. The lookbehind/lookahead keep genuine ranges intact: "37.5 -\n39" has a
# space before the hyphen and is left alone.
_WRAPPED_WORD_RE = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")


def _cell(value) -> str:
    return clean_text(_WRAPPED_WORD_RE.sub("", str(value or ""))).strip()


def _split_columns(width: int, levels: dict[int, int]) -> tuple[list[int], list[int]]:
    """Split the non-level columns by position relative to the first level column.

    Row labels LEAD the table (one column on page 7, two on page 14 where an age
    band qualifies every row). A non-level column that sits AFTER the levels is
    an outcome column, not part of the label - see MTS_LEVEL_CLEARANCE.
    """
    first_level = min(levels)
    key = [i for i in range(width) if i < first_level and i not in levels]
    outcome = [i for i in range(width) if i > first_level and i not in levels]
    return key, outcome


def _table_cohort(caption: str, fallback: str) -> str:
    upper = caption.upper()
    for needle, name in COHORTS:
        if needle in upper:
            return name
    return fallback


def extract_table_cells(doc, meta: DocMeta, key: str) -> Iterator[Chunk]:
    """One chunk per (row label, triage level) cell of every level-column table."""
    cohort = "GENERAL"
    for page_index in range(doc.page_count):
        page_number = page_index + 1
        try:
            found = doc[page_index].find_tables()
        except Exception as exc:  # a malformed page must not stop the ingest
            log.warning("     table scan failed on page %d: %s", page_number, exc)
            continue

        page_text = doc[page_index].get_text()
        for table_index, table in enumerate(found.tables):
            try:
                rows = table.extract()
            except Exception:
                continue
            if not rows:
                continue

            # The header is the row that names the level columns. Anything above
            # it is the caption.
            header_at = next(
                (i for i, r in enumerate(rows)
                 if sum(1 for c in r if c and LEVEL_COL_RE.search(str(c))) >= 2),
                None,
            )
            inferred = False
            if header_at is not None:
                header = rows[header_at]
                levels = {
                    i: int(m.group(1))
                    for i, c in enumerate(header)
                    if c and (m := LEVEL_COL_RE.search(str(c)))
                }
                key_cols, outcome_cols = _split_columns(len(header), levels)
                caption = " ".join(
                    _cell(c) for r in rows[:header_at] for c in r if _cell(c)
                ) or _cell(header[key_cols[0]] if key_cols else "")
            else:
                header = rows[0]
                caption = " ".join(_cell(c) for c in header if _cell(c))
                mapping = IMPLICIT_LEVEL_COLUMNS.get(caption.upper())
                if not mapping:
                    continue
                levels = dict(mapping)
                key_cols, outcome_cols = _split_columns(len(header), levels)
                header_at, inferred = 0, True

            cohort = _table_cohort(
                caption,
                "PAEDIATRIC" if COHORT_PAGE_RE.search(page_text) else cohort,
            )

            carried: dict[int, str] = {}
            for row_index, row in enumerate(rows[header_at + 1:], start=header_at + 1):
                # Continuation rows leave the row label blank (the paediatric
                # table spans four age bands under one "HEART RATE" label).
                for i in key_cols:
                    if i < len(row) and _cell(row[i]):
                        carried[i] = _cell(row[i])
                row_label = " / ".join(
                    carried[i].replace("\n", " ") for i in key_cols if carried.get(i)
                )
                # The trailing SECONDARY TRIAGE column: not a level, but the
                # only printed statement of what CLEARS Primary Triage.
                for col in outcome_cols:
                    if col >= len(row):
                        continue
                    body = _cell(row[col])
                    if len(body) < 3:
                        continue
                    outcome = _cell(header[col]) if col < len(header) else ""
                    # "SECONDARY\nTRIAGE" is one header wrapped by the PDF.
                    outcome = " ".join(outcome.split()) or "SECONDARY TRIAGE"
                    yield Chunk(
                        id=f"pdf::{key}::p{page_number:04d}::t{table_index:02d}"
                           f"::r{row_index:02d}::O{col:02d}",
                        text=(
                            f"[{meta.cpg_title}"
                            + (f" \u2014 {meta.edition_year}" if meta.edition_year else "")
                            + f" | page {page_number} | {caption or 'discriminator table'}"
                            + f" | {outcome}]\n"
                            f"Discriminator: {row_label or caption}\n"
                            f"{outcome} - NOT a triage level. These are the criteria "
                            f"under which the patient CLEARS Primary Triage with no "
                            f"Level 1-3 escalation and proceeds to Secondary Triage:\n"
                            f"{body}"
                        ),
                        metadata={
                            "source_type": "pdf",
                            "filename": meta.filename,
                            "cpg_title": meta.cpg_title,
                            "edition_year": meta.edition_year,
                            "edition": meta.edition,
                            "doc_type": meta.doc_type,
                            "page_number": page_number,
                            "chunk_index": row_index,
                            "version_date": meta.version_date,
                            "status": "active",
                            # Level 0 keeps this out of every discriminator_cells
                            # fetch, which is bounded at level 1. See the comment
                            # on MTS_LEVEL_CLEARANCE.
                            "mts_level": MTS_LEVEL_CLEARANCE,
                            "mts_label": CLEARANCE_LABEL,
                            "cohort": cohort,
                            "table_caption": caption[:200],
                            "row_label": row_label[:200],
                            "outcome_column": outcome[:100],
                            "is_table_cell": True,
                            "header_inferred": inferred,
                        },
                    )
                for col, level in levels.items():
                    if col >= len(row):
                        continue
                    body = _cell(row[col])
                    if len(body) < 3:
                        continue
                    label = MTS_LEVEL_NAMES[level]
                    header_line = (
                        f"[{meta.cpg_title}"
                        + (f" \u2014 {meta.edition_year}" if meta.edition_year else "")
                        + f" | page {page_number} | {caption or 'discriminator table'}"
                        + f" | LEVEL {level} - {label}]"
                    )
                    yield Chunk(
                        id=f"pdf::{key}::p{page_number:04d}::t{table_index:02d}"
                           f"::r{row_index:02d}::L{level}",
                        text=(
                            f"{header_line}\n"
                            f"Discriminator: {row_label or caption}\n"
                            f"MTS Level {level} ({label.replace('_', ' ')}) criteria:\n"
                            f"{body}"
                        ),
                        metadata={
                            "source_type": "pdf",
                            "filename": meta.filename,
                            "cpg_title": meta.cpg_title,
                            "edition_year": meta.edition_year,
                            "edition": meta.edition,
                            "doc_type": meta.doc_type,
                            "page_number": page_number,
                            "chunk_index": row_index,
                            "version_date": meta.version_date,
                            "status": "active",
                            # The keys the retriever can filter on.
                            "mts_level": level,
                            "mts_label": label,
                            "cohort": cohort,
                            "table_caption": caption[:200],
                            "row_label": row_label[:200],
                            "is_table_cell": True,
                            # True where the level columns are not printed and
                            # come from IMPLICIT_LEVEL_COLUMNS instead.
                            "header_inferred": inferred,
                        },
                    )


# ------------------------------------------------------- MTS action tables
# Pages 5-6 carry two tables with no LEVEL columns at all, so the discriminator
# extractor above skips them outright and they survive only as flattened prose:
#
#   INFECTIOUS DISEASES / HAZMAT         DISEASES/EXPOSURE | TO BE PLACED AT |
#                                        INITIAL ACTIONS          (13 rows)
#   AGGRESSIVE / POTENTIALLY VIOLENT     CONDITIONS | POTENTIAL ACTIONS
#   PERSONS                                                        (3 rows)
#
# Their output column is a PLACEMENT ("Isolation (Negative Pressure)",
# "Decontamination"), not a triage level, and the intake already collects the
# inputs for them - `exposure_risk` and `behavioural_risk` in schemas.py. Until
# now those enum values had no indexed source behind them: the placement facts
# existed only as hand-written English in the intake description dict.
#
# These rows are indexed with is_table_cell=False so they can never reach
# discriminator_cells(), which is the one consumer of that key.
NOTE_ROW_RE = re.compile(r"^note\s*[:\-]", re.IGNORECASE)


def extract_action_rows(doc, meta: DocMeta, key: str) -> Iterator[Chunk]:
    """One chunk per row of every headed table that has NO level columns."""
    for page_index in range(doc.page_count):
        page_number = page_index + 1
        try:
            found = doc[page_index].find_tables()
        except Exception as exc:
            log.warning("     action-table scan failed on page %d: %s", page_number, exc)
            continue

        for table_index, table in enumerate(found.tables):
            try:
                rows = table.extract()
            except Exception:
                continue
            if len(rows) < 3:
                continue
            # Anything with level columns belongs to extract_table_cells().
            if any(
                sum(1 for c in r if c and LEVEL_COL_RE.search(str(c))) >= 2 for r in rows
            ):
                continue

            # Row 0 is a caption spanning the table (one filled cell); row 1
            # names the columns. Without that shape there is no table to read.
            filled0 = [i for i, c in enumerate(rows[0]) if _cell(c)]
            if filled0 != [0]:
                continue
            caption = _cell(rows[0][0])
            # COMPLAINTS LIST (ADULT) prints no level header either, and would
            # match every shape test above. It is a level table whose columns
            # are supplied by IMPLICIT_LEVEL_COLUMNS, and indexing it here as
            # well would duplicate 130 discriminator cells with their level
            # stripped off - the exact meaning the cell path exists to keep.
            if caption.upper() in IMPLICIT_LEVEL_COLUMNS:
                continue
            header = [_cell(c) for c in rows[1]]
            if sum(1 for h in header if h) < 2:
                continue
            out_cols = [i for i in range(1, len(header)) if header[i]]

            for row_index, row in enumerate(rows[2:], start=2):
                label = " ".join(_cell(row[0]).split()) if row else ""
                if not label:
                    continue
                body = {
                    i: _cell(row[i])
                    for i in out_cols
                    if i < len(row) and _cell(row[i])
                }
                # A trailing "Note:" row applies to every row above it and has
                # no output columns of its own.
                if not body:
                    if not NOTE_ROW_RE.match(label):
                        continue
                    text = (
                        f"[{meta.cpg_title}"
                        + (f" \u2014 {meta.edition_year}" if meta.edition_year else "")
                        + f" | page {page_number} | {caption}]\n"
                        f"Note applying to every row of the {caption} table:\n{label}"
                    )
                    row_kind = "note"
                else:
                    lines = "\n".join(
                        f"{header[i]}:\n"
                        + "\n".join(f"- {ln.strip()}" for ln in body[i].splitlines() if ln.strip())
                        for i in sorted(body)
                    )
                    text = (
                        f"[{meta.cpg_title}"
                        + (f" \u2014 {meta.edition_year}" if meta.edition_year else "")
                        + f" | page {page_number} | {caption}]\n"
                        f"{header[0] or 'Condition'}: {label}\n"
                        f"NOT a triage level - this table assigns placement and "
                        f"initial actions, which apply in addition to whatever "
                        f"triage level the patient is given.\n{lines}"
                    )
                    row_kind = "action"

                yield Chunk(
                    id=f"pdf::{key}::p{page_number:04d}::t{table_index:02d}"
                       f"::a{row_index:02d}",
                    text=text,
                    metadata={
                        "source_type": "pdf",
                        "filename": meta.filename,
                        "cpg_title": meta.cpg_title,
                        "edition_year": meta.edition_year,
                        "edition": meta.edition,
                        "doc_type": meta.doc_type,
                        "page_number": page_number,
                        "chunk_index": row_index,
                        "version_date": meta.version_date,
                        "status": "active",
                        # NOT a discriminator cell. discriminator_cells() filters
                        # on is_table_cell, so False is what keeps a placement
                        # rule out of the triage-level evidence.
                        "is_table_cell": False,
                        "is_action_row": True,
                        "row_kind": row_kind,
                        "table_caption": caption[:200],
                        "row_label": label[:200],
                        "cohort": "GENERAL",
                    },
                )


# Documents whose content lives in tables and figures: extracted by
# structured_pdf.py rather than as a text stream.
STRUCTURED_TABLE_RE = re.compile(r"Standard Practice Guidelines for Assistant Medical Officer", re.I)
_VOCAB: set[str] | None = None


def _vocabulary() -> set[str]:
    """Words a figure's OCR may use: the embedder's vocabulary plus every word
    already in the index (medical terms - SABA, TBSA, FEV - that no general
    vocabulary holds)."""
    global _VOCAB
    if _VOCAB is None:
        words = set(_known_words())
        try:
            for t in get_collection().get(include=["documents"])["documents"]:
                words.update(re.findall(r"[a-z]{3,}", (t or "").lower()))
        except Exception as exc:  # noqa: BLE001 - an empty index still has the embedder's words
            log.warning("Index vocabulary unavailable for OCR checks (%s).", exc)
        _VOCAB = words
    return _VOCAB


def _ocr_sidecar(path: Path) -> dict:
    side = path.with_name(path.name + ".ocr.json")
    if not side.exists():
        return {}
    try:
        return json.loads(side.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        log.error("Unreadable OCR sidecar %s: %s", side.name, exc)
        return {}


def _ocr_pages(path: Path) -> list[str]:
    """OCR text for a scanned PDF, from `<file>.pdf.ocr.json` written by
    tools/ocr_pdf.swift (Apple Vision). Used only for pages whose own text
    layer is empty - two MOH CPGs (Sore Throat 2003, Non-Variceal UGIB 2003)
    are page images with no text layer at all."""
    side = path.with_name(path.name + ".ocr.json")
    if not side.exists():
        return []
    try:
        return list(json.loads(side.read_text(encoding="utf-8")).get("pages", []))
    except Exception as exc:
        log.error("Unreadable OCR sidecar %s: %s", side.name, exc)
        return []


def load_pdf_chunks(
    tokenizer: Tokenizer, report: IngestReport, only: set[str] | None = None
) -> Iterator[Chunk]:
    """`only`: filenames to process (the rest are skipped) - `--new-only`."""
    pdfs = sorted(config.RAW_PDF_DIR.glob("*.pdf"))
    if only is not None:
        pdfs = [p for p in pdfs if p.name in only]
    if not pdfs and only is None:
        report.warnings.append(f"No PDFs found in {config.RAW_PDF_DIR}")
        return

    for path in pdfs:
        # KKM, Malaysian, or cited by a KKM document - see source_policy.py.
        if not source_policy.allowed(path.name):
            report.warnings.append(
                f"Skipped {path.name}: not a KKM or Malaysian document and no KKM citation is recorded "
                f"for it in {source_policy.QUALIFICATIONS.name}")
            log.warning("Skipped %s (source policy)", path.name)
            continue
        try:
            doc = pymupdf.open(path)
        except Exception as exc:
            report.warnings.append(f"Could not open {path.name}: {exc}")
            log.error("Could not open %s: %s", path.name, exc)
            continue

        pages = [clean_text(page.get_text()) for page in doc]
        if STRUCTURED_TABLE_RE.search(path.name):
            # Table rows, chapter headings and the text of figure images - see
            # structured_pdf.py. Everything below (OCR fill, re-joining) still runs.
            from . import structured_pdf  # noqa: PLC0415
            side = _ocr_sidecar(path)
            pages = [clean_text(t) for t in structured_pdf.pages(
                doc, side.get("pages", []), _vocabulary(), side.get("lines", []))]
            log.info("     structured extraction: table rows, headings and figure text (%d pages)", len(pages))
        ocr = _ocr_pages(path)
        if ocr:
            filled = 0
            for i, text in enumerate(pages):
                if len(text.strip()) < 30 and i < len(ocr) and len(ocr[i].strip()) >= 30:
                    pages[i] = clean_text(ocr[i])
                    filled += 1
            log.info("     OCR sidecar: text for %d image-only page(s)", filled)
        pages, joined = rejoin_broken_words(pages)
        if joined:
            log.info("     re-joined %d word(s) broken across lines", joined)
        head = "\n".join(pages[: min(4, len(pages))])
        meta = describe_pdf(path, head, "\n".join(pages))
        key = _doc_key(meta.filename)
        report.pdf_files += 1
        n_chunks = 0
        table_chunks: list[Chunk] = []
        action_chunks: list[Chunk] = []
        empty_pages = 0

        # A triage document is a grid before it is prose. Extract the cells
        # while the document is still open; the prose pass below still runs.
        if meta.doc_type == config.DOC_TYPE_TRIAGE:
            try:
                table_chunks = list(extract_table_cells(doc, meta, key))
            except Exception as exc:
                report.warnings.append(
                    f"{meta.filename}: discriminator-table extraction failed "
                    f"({type(exc).__name__}: {exc}). The triage grid is indexed as "
                    "prose only, which does not preserve which level a criterion "
                    "belongs to."
                )
                log.error("Table extraction failed for %s: %s", meta.filename, exc)
            try:
                action_chunks = list(extract_action_rows(doc, meta, key))
            except Exception as exc:
                report.warnings.append(
                    f"{meta.filename}: action-table extraction failed "
                    f"({type(exc).__name__}: {exc}). The placement rules on pages "
                    "5-6 - isolation, decontamination, Code GREY - are indexed as "
                    "prose only."
                )
                log.error("Action extraction failed for %s: %s", meta.filename, exc)
        doc.close()

        log.info(
            "PDF  %-62s | %-20s | year=%-4s ed=%-2s | %d pages",
            meta.filename[:62],
            meta.doc_type,
            meta.edition_year or "?",
            meta.edition or "?",
            len(pages),
        )

        for page_index, text in enumerate(pages):
            report.pdf_pages += 1
            if len(text) < 40:
                report.skipped_pages += 1
                empty_pages += 1
                continue
            page_number = page_index + 1
            caution = DOC_TYPE_CAUTION.get(meta.doc_type, "")
            doc_caution = next(
                (note for needle, note in DOC_CAUTIONS.items()
                 if needle in meta.filename),
                "",
            )
            page_caution = next(
                (
                    note for (needle, pg), note in PAGE_CAUTIONS.items()
                    if pg == page_number and needle in meta.filename
                ),
                "",
            )
            preamble = "".join(
                f"{c}\n" for c in (caution, doc_caution, page_caution) if c
            )
            for ci, piece in enumerate(
                tokenizer.split(text, config.CHUNK_TOKENS, config.CHUNK_OVERLAP_TOKENS)
            ):
                if len(piece) < 40:
                    continue
                n_chunks += 1
                yield Chunk(
                    id=f"pdf::{key}::p{page_number:04d}::c{ci:02d}",
                    text=(
                        f"[{meta.cpg_title}"
                        + (f" — {meta.edition_year}" if meta.edition_year else "")
                        + f" | page {page_number}]\n{preamble}{piece}"
                    ),
                    metadata={
                        "source_type": "pdf",
                        "filename": meta.filename,
                        "cpg_title": meta.cpg_title,
                        "edition_year": meta.edition_year,
                        "edition": meta.edition,
                        "doc_type": meta.doc_type,
                        "page_number": page_number,
                        "chunk_index": ci,
                        "version_date": meta.version_date,
                        "status": "active",
                    },
                )
        for chunk in table_chunks:
            n_chunks += 1
            yield chunk
        for chunk in action_chunks:
            n_chunks += 1
            yield chunk
        if action_chunks:
            log.info(
                "     %d action rows (placement / initial actions, no triage level)",
                len(action_chunks),
            )
        if table_chunks:
            levels = sorted(
                {c.metadata["mts_level"] for c in table_chunks}
                - {MTS_LEVEL_CLEARANCE}
            )
            n_clear = sum(
                1 for c in table_chunks
                if c.metadata["mts_level"] == MTS_LEVEL_CLEARANCE
            )
            log.info(
                "     %d discriminator cells across levels %s (+%d clearance)",
                len(table_chunks) - n_clear, ", ".join(str(x) for x in levels), n_clear,
            )
            if 5 not in levels:
                report.warnings.append(
                    f"{meta.filename}: no LEVEL 5 discriminator cells were "
                    "extracted. Level 5 retrieval cannot be filtered."
                )

        report.pdf_chunks += n_chunks
        report.per_document[meta.filename] = {
            "doc_type": meta.doc_type,
            "cpg_title": meta.cpg_title,
            "edition_year": meta.edition_year,
            "edition": meta.edition,
            "chunks": n_chunks,
            "table_cells": len(table_chunks),
            "action_rows": len(action_chunks),
        }
        if empty_pages:
            log.info("     %d page(s) held no extractable text (images/covers).", empty_pages)
        if not meta.edition_year:
            report.warnings.append(
                f"{meta.filename}: edition year not stated in the filename or the "
                "document text. Comparative answers cannot date this source — add "
                "the year to the filename (e.g. '... (6th Edition) 2020.pdf') and re-ingest."
            )


# ===========================================================================
# FUKKM JSON -> chunks
# ===========================================================================


def detect_fukkm_shape(records: Sequence[dict]) -> str:
    """`current` = drug_name holds a real name. `column_shifted` = the scraper
    wrote the row number into drug_name and shifted every later column."""
    if not records:
        return "empty"
    sample = records[: min(200, len(records))]
    numeric_names = sum(1 for r in sample if str(r.get("drug_name", "")).strip().isdigit())
    cat_in_indication = sum(
        1 for r in sample if CATEGORY_CODE_RE.match(str(r.get("indication", "")))
    )
    if numeric_names > len(sample) * 0.8 and cat_in_indication > len(sample) * 0.8:
        return "column_shifted"
    return "current"


def normalise_fukkm(record: dict, shape: str) -> dict:
    """Return a canonical drug record regardless of which scrape produced it."""
    if shape == "column_shifted":
        norm = {
            "fukkm_no": str(record.get("drug_name", "")).strip(),
            "drug_name": str(record.get("drug_name_recovered", "")).strip(),
            "mdc_code": str(record.get("prescriber_category", "")).strip(),
            "prescriber_category": str(record.get("indication", "")).strip(),
            "indication": str(record.get("dosage", "")).strip(),
            "dosage": "",
        }
    else:
        norm = {
            "fukkm_no": str(record.get("fukkm_no", "")).strip(),
            "drug_name": str(record.get("drug_name", "")).strip(),
            "mdc_code": str(record.get("mdc_code", "")).strip(),
            "prescriber_category": str(record.get("prescriber_category", "")).strip(),
            "indication": str(record.get("indication", "")).strip(),
            "dosage": str(record.get("dosage", "")).strip(),
        }
    m = ATC_RE.match(norm["mdc_code"])
    norm["atc_code"] = m.group(1) if m else ""
    return norm


def fukkm_text_block(norm: dict) -> str:
    name = norm["drug_name"] or "(generic name not captured by the scraper)"
    lines = [
        "FUKKM (MOH Medicines Formulary / Blue Book) entry",
        f"Generic name: {name}",
    ]
    if norm["fukkm_no"]:
        lines.append(f"FUKKM listing number: {norm['fukkm_no']}")
    if norm["mdc_code"]:
        lines.append(f"MDC code: {norm['mdc_code']}")
    if norm["atc_code"]:
        lines.append(f"WHO ATC code: {norm['atc_code']}")
    if norm["prescriber_category"]:
        lines.append(f"Prescriber category: {norm['prescriber_category']}")
    if norm["indication"]:
        lines.append(f"Indication: {norm['indication']}")
    if norm["dosage"]:
        lines.append(f"Dosage: {norm['dosage']}")
    return "\n".join(lines)


def load_fukkm_chunks(tokenizer: Tokenizer, report: IngestReport) -> Iterator[Chunk]:
    for path in sorted(config.RAW_JSON_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            report.warnings.append(f"Could not parse {path.name}: {exc}")
            log.error("Could not parse %s: %s", path.name, exc)
            continue
        records = payload if isinstance(payload, list) else payload.get("drugs", [])
        if not isinstance(records, list) or not records:
            report.warnings.append(f"{path.name} contained no drug records.")
            continue

        shape = detect_fukkm_shape(records)
        log.info("JSON %-62s | %d records | shape=%s", path.name, len(records), shape)
        if shape == "column_shifted":
            msg = (
                f"{path.name}: scraped with mis-aligned table columns — generic drug "
                "names were never captured (drug_name holds the row number). "
                "Prescriber category, MDC/ATC code and indication text ARE usable. "
                "Re-run `python -m app.scrape_fukkm` to rebuild with drug names."
            )
            report.warnings.append(msg)
            log.warning(msg)

        n = 0
        for record in records:
            norm = normalise_fukkm(record, shape)
            if not any(
                norm[k] for k in ("drug_name", "indication", "dosage", "mdc_code")
            ):
                continue
            report.drug_records += 1
            block = fukkm_text_block(norm)
            row_key = norm["fukkm_no"] or hashlib.sha1(
                block.encode("utf-8")
            ).hexdigest()[:10]
            for ci, piece in enumerate(
                tokenizer.split(block, config.CHUNK_TOKENS, config.CHUNK_OVERLAP_TOKENS)
            ):
                n += 1
                yield Chunk(
                    id=f"fukkm::{row_key}::c{ci:02d}",
                    text=piece,
                    metadata={
                        "source_type": "json",
                        "filename": path.name,
                        "cpg_title": "FUKKM — MOH Medicines Formulary (Blue Book)",
                        "edition_year": "",
                        "edition": "",
                        "doc_type": config.DOC_TYPE_DRUG,
                        "drug_name": norm["drug_name"],
                        "fukkm_no": norm["fukkm_no"],
                        "mdc_code": norm["mdc_code"],
                        "atc_code": norm["atc_code"],
                        "prescriber_category": norm["prescriber_category"],
                        "chunk_index": ci,
                        "version_date": "",
                        "status": "active",
                    },
                )
        report.drug_chunks += n
        report.per_document[path.name] = {
            "doc_type": config.DOC_TYPE_DRUG,
            "cpg_title": "FUKKM — MOH Medicines Formulary (Blue Book)",
            "records": len(records),
            "chunks": n,
            "shape": shape,
        }


# ===========================================================================
# ChromaDB
# ===========================================================================


# ===========================================================================
# Web-only guidelines (scrape_nag.py) -> chunks
# ===========================================================================
# Population is carried in the TITLE, because population.py reads it from
# there: an adult-section page must never ground a child, and the reverse.
NAG_TITLES = {
    "adult": "National Antimicrobial Guideline 2024 (Adults)",
    "paediatric": "National Antimicrobial Guideline 2024 (Paediatrics)",
    "all": "National Antimicrobial Guideline 2024 (Primary Care Pathways)",
}


def load_web_chunks(tokenizer: Tokenizer, report: IngestReport) -> Iterator[Chunk]:
    """The NAG 2024 pages saved by `python -m app.scrape_nag`, if present.

    Each chunk starts with its chapter title (a split page is otherwise a
    table fragment with no subject) and carries `url` in place of a page
    number, so a citation opens the live page."""
    path = config.RAW_JSON_DIR / "nag_pages.json"
    if not path.exists():
        return
    pages = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for page in pages:
        title = NAG_TITLES.get(page.get("population", "all"), NAG_TITLES["all"])
        chapter = page["title"].replace("National Antimicrobial Guideline 2024 - ", "")
        for ci, part in enumerate(tokenizer.split(clean_text(page["text"]),
                                                  config.CHUNK_TOKENS - 32,
                                                  config.CHUNK_OVERLAP_TOKENS)):
            n += 1
            yield Chunk(
                id=f"{page['id']}-{ci}",
                text=f"[NAG 2024 - {chapter}]\n{part}",
                metadata={
                    "filename": f"{page['id']}.html",
                    "cpg_title": title,
                    "edition_year": "2024",
                    "edition": "4",
                    "doc_type": config.DOC_TYPE_CPG_FULL,
                    "chunk_index": ci,
                    "url": page["url"],
                    "section": chapter,
                    "version_date": page.get("fetched", ""),
                    "status": "active",
                },
            )
    report.per_document["National Antimicrobial Guideline 2024 (web)"] = {
        "doc_type": config.DOC_TYPE_CPG_FULL, "chunks": n, "pages": len(pages)}
    log.info("WEB  National Antimicrobial Guideline 2024 | %d pages -> %d chunks", len(pages), n)


def build_embedding_function():
    """all-MiniLM-L6-v2 via sentence-transformers; the ONNX build of the same
    model is the fallback so a torch problem does not block ingestion."""
    from chromadb.utils import embedding_functions as ef

    try:
        fn = ef.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL, normalize_embeddings=True
        )
        log.info("Embeddings: %s", config.EMBEDDING_MODEL)
        return fn
    except Exception as exc:
        # The old fallback was ChromaDB's ONNX MiniLM. That is a *different*
        # model with 384 dims; silently swapping it under a 1024-dim index
        # produces a corrupt collection, not a degraded one. Fail loudly.
        raise RuntimeError(
            f"Could not load the embedding model {config.EMBEDDING_MODEL!r} "
            f"({type(exc).__name__}: {exc}). Refusing to fall back to a "
            "different model - the index dimensionality would not match."
        ) from exc


def get_client():
    import chromadb

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(reset: bool = False):
    client = get_client()
    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
            log.info("Dropped existing collection %r.", config.COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=build_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _batched(items: Iterable[Chunk], size: int) -> Iterator[list[Chunk]]:
    batch: list[Chunk] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _indexed_filenames() -> set[str]:
    try:
        coll = get_collection()
        total = coll.count()
        if not total:
            return set()
        got = coll.get(include=["metadatas"], limit=total)
        return {str(m.get("filename", "")) for m in got["metadatas"] if m}
    except Exception:
        return set()


def build_index(reset: bool = False, dry_run: bool = False, web_only: bool = False,
                new_only: bool = False) -> IngestReport:
    report = IngestReport()
    tokenizer = Tokenizer()

    if config.CHUNK_TOKENS > config.EMBEDDING_MAX_TOKENS:
        report.warnings.append(
            f"CHUNK_TOKENS={config.CHUNK_TOKENS} exceeds the {config.EMBEDDING_MAX_TOKENS}-token "
            f"input limit of {config.EMBEDDING_MODEL}; tokens past {config.EMBEDDING_MAX_TOKENS} "
            "in each chunk are truncated when embedded (they are still returned to "
            "the LLM in full). Set CHUNK_TOKENS={config.EMBEDDING_MAX_TOKENS} for lossless embeddings."
        )

    if web_only:
        # Adds (or refreshes) only the web-only guidelines - no hour-long rebuild.
        chunks = list(load_web_chunks(tokenizer, report))
    elif new_only:
        # Only PDFs not yet in the index: minutes instead of an hour. Chunk ids
        # are derived from the file and page, so an upsert never duplicates.
        have = _indexed_filenames()
        fresh = {p.name for p in config.RAW_PDF_DIR.glob("*.pdf")} - have
        log.info("--new-only: %d PDF(s) not yet indexed: %s", len(fresh), ", ".join(sorted(fresh)) or "none")
        chunks = list(load_pdf_chunks(tokenizer, report, only=fresh))
    else:
        chunks = (list(load_pdf_chunks(tokenizer, report))
                  + list(load_fukkm_chunks(tokenizer, report))
                  + list(load_web_chunks(tokenizer, report)))
    total = len(chunks)
    log.info(
        "Parsed %d chunks (%d from %d PDFs, %d from %d drug records).",
        total, report.pdf_chunks, report.pdf_files, report.drug_chunks, report.drug_records,
    )

    if dry_run:
        log.info("--dry-run: nothing embedded, nothing written.")
        for w in report.warnings:
            log.warning(w)
        return report

    if not total:
        log.error("Nothing to index.")
        return report

    collection = get_collection(reset=reset)
    done = 0
    for batch in _batched(chunks, BATCH_SIZE):
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[c.metadata for c in batch],
        )
        done += len(batch)
        log.info("Indexed %d/%d chunks (%.0f%%)", done, total, 100 * done / total)

    log.info("Collection %r now holds %d chunks.", config.COLLECTION_NAME, collection.count())
    for w in report.warnings:
        log.warning(w)
    return report


def print_stats() -> None:
    collection = get_collection()
    total = collection.count()
    print(f"collection : {config.COLLECTION_NAME}\nchunks     : {total}")
    if not total:
        return
    got = collection.get(include=["metadatas"], limit=total)
    tally: dict[tuple[str, str, str], int] = {}
    for m in got["metadatas"]:
        key = (m.get("filename", "?"), m.get("doc_type", "?"), str(m.get("edition_year", "")))
        tally[key] = tally.get(key, 0) + 1
    print(f"{'chunks':>7}  {'doc_type':<22} {'year':<5} filename")
    for (fn, dt, yr), n in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"{n:>7}  {dt:<22} {yr:<5} {fn}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Index KKM PDFs and FUKKM JSON into ChromaDB.")
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    ap.add_argument("--dry-run", action="store_true", help="parse and report only")
    ap.add_argument("--stats", action="store_true", help="show what is already indexed")
    ap.add_argument("--web-only", action="store_true",
                    help="index only the web-only guidelines (scrape_nag.py), without a rebuild")
    ap.add_argument("--new-only", action="store_true",
                    help="index only PDFs not already in the index, without a rebuild")
    args = ap.parse_args()

    if args.stats:
        print_stats()
        return
    if (args.web_only or args.new_only) and args.reset:
        ap.error("--web-only / --new-only add to the existing index; they cannot be combined with --reset")
    build_index(reset=args.reset, dry_run=args.dry_run, web_only=args.web_only, new_only=args.new_only)


if __name__ == "__main__":
    main()