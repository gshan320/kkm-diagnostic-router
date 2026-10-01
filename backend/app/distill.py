"""Attention layer A3: send the model the sentences that matter, in the order it reads best.

Why this exists
---------------
After A2 the model reads only documents it has a reason to read, but it still
reads them as whole 512-token chunks - 8-10k prompt tokens for a chest pain,
~60 s of prefill on this machine before a word is written. Most of each chunk is
not about this patient. Two findings from the literature say what to do:

  - Keep the useful sentences, drop the rest. RECOMP (Xu et al., ICLR 2024)
    reaches ~6% of the original length with little loss by selecting
    sentences; CRAG (Yan et al., 2024) splits passages into "knowledge strips",
    scores them, and recomposes the kept ones.
  - Put the most relevant material at the start and the end. "Lost in the
    Middle" (Liu et al., TACL 2024) and LongLLMLingua (ACL 2024): models use
    the edges of the context best and the middle worst.

What this module does, per clinical chunk:

  1. split into sentences (PDF line breaks joined first, bullets kept apart);
  2. score each against the attention queries - the key findings, the working
     hypotheses, and the treatment question - with the medical embedder
     (MedEmbed). The scorer only ranks sentences WITHIN documents the A2 topic
     gate already accepted; it is never asked which document is relevant,
     which is the job it was measured to fail at;
  3. keep the best few, with a bonus for sentences that carry a dose or a
     threshold (a number with a unit) - those are what a recommendation needs;
  4. recompose them in their ORIGINAL order, "..." marking each gap, under the
     chunk's own header line, so [S#], document and page stay exact.

Then the chunks are ordered edge-first (best at the top, second-best at the
bottom, weakest in the middle) and cut to a token budget.

FUKKM rows are trimmed to the lines a prescriber reads (name, category,
indication, dosage); the MDC and ATC codes go. The MTS table is never here -
A1 replaced it with the code-decided TRIAGE block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np

# A sentence carrying a number with a unit - a dose, a rate, a threshold.
_QUANTITY = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|mL|l|L|units?|iu|IU|mmol|mEq|%|kg|/kg|/min|/h|hours?|h\b|"
    r"minutes?|mins?|days?|mmHg|°C|x\b|times)", re.I)
_BULLET = re.compile(r"^\s*(?:[-•●▪–*]|\(?[a-z0-9]{1,3}[.)])\s+", re.I)
_HEADER = re.compile(r"^\[[^\]]+\|\s*page[^\]]*\]\s*$", re.I)


def sentences(text: str) -> tuple[str, list[str]]:
    """(header line or "", sentences). Joins PDF line wraps, keeps bullets apart."""
    lines = (text or "").splitlines()
    header = ""
    if lines and _HEADER.match(lines[0].strip()):
        header, lines = lines[0].strip(), lines[1:]
    units: list[str] = []
    buf = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            if buf:
                units.append(buf)
                buf = ""
            continue
        if _BULLET.match(line) and buf:
            units.append(buf)
            buf = line
            continue
        buf = f"{buf} {line}".strip() if buf else line
    if buf:
        units.append(buf)
    out: list[str] = []
    for u in units:
        # Case-blind: part of the index is stored lowercased, and a splitter
        # that waited for a capital letter kept a whole 500-token page as one
        # "sentence". A full stop inside a number ("0.9%") is not followed by
        # a space, so it is not a boundary; "e.g. x" splitting is harmless.
        for s in re.split(r"(?<=[.;])\s+(?=[a-z(\[]|[A-Z])", u):
            s = re.sub(r"\s+", " ", s).strip()
            if len(s) >= 12 and not re.fullmatch(r"[\d\s.,-]+", s):
                out.extend(_windows(s))
    return header, out


MAX_SENTENCE = 360


def _windows(s: str, size: int = 300) -> list[str]:
    """A flattened table has no full stops: ACS CPG p175 (CKD dose adjustments)
    came out as one 2,000-character "sentence" that was kept whole because it
    carried doses. Long runs are cut at word boundaries so each piece is scored
    on its own."""
    if len(s) <= MAX_SENTENCE:
        return [s]
    out, cur = [], ""
    for w in s.split(" "):
        if cur and len(cur) + 1 + len(w) > size:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return out


@dataclass
class Scored:
    chunk: object          # rag_engine.Retrieved
    header: str
    sentences: list[str]
    scores: list[float]
    keep: list[int]

    @property
    def relevance(self) -> float:
        return max((self.scores[i] for i in self.keep), default=0.0)

    def text(self) -> str:
        parts, last = [], -2
        for i in sorted(self.keep):
            if last >= 0 and i != last + 1:
                parts.append("...")
            parts.append(self.sentences[i])
            last = i
        body = "\n".join(parts)
        return f"{self.header}\n{body}" if self.header else body


def distill(
    chunks: Sequence,
    queries: list[str],
    embed: Callable[[list[str]], list[list[float]]],
    *,
    per_chunk: int = 5,
    quantity_bonus: float = 0.05,
    always_keep: Callable[[str], bool] | None = None,
    max_extra: int = 2,
) -> list[Scored]:
    """Score and select sentences for every chunk (see module docstring)."""
    split = [(c, *sentences(c.text)) for c in chunks]
    flat = [s for _, _, ss in split for s in ss]
    if not flat or not queries:
        return [Scored(c, h, ss, [0.0] * len(ss), list(range(len(ss)))) for c, h, ss in split]
    vecs = np.asarray(embed(queries + flat), dtype=np.float32)
    q, sv = vecs[: len(queries)], vecs[len(queries):]
    best_all = (sv @ q.T).max(axis=1)  # embeddings are normalised: dot = cosine
    out: list[Scored] = []
    pos = 0
    for c, header, ss in split:
        scores = []
        for j in range(len(ss)):
            best = float(best_all[pos + j])
            if _QUANTITY.search(ss[j]):
                best += quantity_bonus
            scores.append(best)
        pos += len(ss)
        cited = {j for j, t in enumerate(ss) if is_citation(t)}
        order = [j for j in sorted(range(len(ss)), key=lambda j: -scores[j]) if j not in cited]
        keep = set(order[:per_chunk])
        if always_keep:
            # The best `max_extra` dose sentences beyond the top few - not all
            # of them: a dosing table is nothing but dose sentences.
            extra = [j for j in order if j not in keep and always_keep(ss[j])]
            keep |= set(extra[:max_extra])
        out.append(Scored(c, header, ss, scores, sorted(keep)))
    return out


def edge_order(items: list[Scored]) -> list[Scored]:
    """Best first, second-best last, the rest folded toward the middle."""
    ranked = sorted(items, key=lambda x: -x.relevance)
    front, back = [], []
    for i, it in enumerate(ranked):
        (front if i % 2 == 0 else back).append(it)
    return front + back[::-1]


def trim_formulary(text: str) -> str:
    """The lines a prescriber reads from a FUKKM row; codes dropped."""
    keep = []
    for line in (text or "").splitlines():
        low = line.lower()
        if low.startswith(("mdc code", "who atc code", "fukkm (moh medicines")):
            continue
        keep.append(line)
    return "\n".join(keep).strip()


def with_text(chunk, text: str):
    """A copy of a Retrieved chunk carrying the distilled text."""
    return replace(chunk, text=text)


# ---------------------------------------------------------------------------
# Reference lists. Measured 2026-09-30: the rhabdomyolysis prompt carried the
# bibliography pages of the Dyslipidaemia and Hyperkalaemia CPGs (their paper
# titles say "statin-induced myopathy", "losartan"), and formulary-by-name then
# pulled losartan, ezetimibe and simvastatin out of them. A reference list is
# evidence of nothing for this patient.
#
# Three marks only a bibliography carries - a run of three "surname initials,"
# authors, a journal citation (year ; volume ( issue ) : pages), and doi / pmid
# / "accessed" / "et al. <title>". Inline evidence citations ("Fedele et al,
# 1991; level 5") carry none of them. Tuned on the whole corpus: 1,389 chunks
# have 8+ marks, but some are dosing pages with a reference tail (NAG, Abdominal
# Trauma p30, ~4.3 marks per 1,000 characters), so a WHOLE chunk goes only when
# it is dense (5+ per 1,000); otherwise the citation sentences go one by one.
# ---------------------------------------------------------------------------
_AUTHOR = r"(?!et\b|al\b)[a-z][a-z'-]{2,}(?:\s*-\s*[a-z]+)?\s+[a-z]{1,3}\.?"
_AUTHOR_RUN = re.compile(rf"\b{_AUTHOR}\s*,\s*{_AUTHOR}\s*,\s*{_AUTHOR}\s*[,.;]", re.I)
_JOURNAL = re.compile(
    r"\b(?:19|20)\d\d\s*(?:[a-z]{3,9}\.?\s*(?:\d{1,2}\s*)?)?[;:]\s*\d+\s*"
    r"(?:\(\s*[\d\s-]+\)\s*)?(?::\s*[a-z]?\d+)?", re.I)
_BIB_MISC = re.compile(
    r"\bdoi\b|\bpmid\b|\bpmcid\b|\baccessed\b|\bavailable (?:from|at)\s*:?\s*(?:http|www)|"
    r"\bet al\s*\.\s+[a-z]", re.I)
# APA style (CHAMP 2025): "mingels, a., jacobs, l., ... ( 2009 ). title. clinical
# chemistry, 55 ( 1 ), 101 - 108." - the Vancouver patterns above miss all three.
_APA_AUTHORS = re.compile(r"(?:\b[a-z][a-z'-]{2,}\s*,\s*(?:[a-z]\.\s*){1,3},\s*){2,}", re.I)
_APA_YEAR = re.compile(r"\(\s*(?:19|20)\d\d[a-z]?\s*\)\s*\.")
_APA_VOLUME = re.compile(r"\b\d+\s*\(\s*\d+\s*\)\s*,\s*(?:article\s+)?[a-z]?\d+(?:\s*[\u2013-]\s*\d+)?", re.I)
_PATTERNS = (_AUTHOR_RUN, _JOURNAL, _BIB_MISC, _APA_AUTHORS, _APA_YEAR, _APA_VOLUME)


def citation_marks(text: str) -> int:
    return sum(len(p.findall(text or "")) for p in _PATTERNS)


def is_citation(sentence: str) -> bool:
    return any(p.search(sentence or "") for p in _PATTERNS)


# A dose instruction - never in a reference list. A NAG chunk ("amoxicillin
# 500mg po q8h plus metronidazole 400mg po q8h" + a reference tail) was caught by
# the APA patterns on 2026-09-30; a chunk that doses is guidance.
_DOSE_INSTRUCTION = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|units?|ml)\s*(?:/\s*kg\s*)?(?:/\s*day\s*)?\s*"
    r"(?:po|iv|im|sc|od|bd|tds|qid|q\d+h|stat|daily)\b", re.I)


def is_bibliography(text: str, min_marks: int = 5, per_1000: float = 5.0) -> bool:
    n = citation_marks(text)
    if n < min_marks or n * 1000 / max(len(text or ""), 1) < per_1000:
        return False
    return len(_DOSE_INSTRUCTION.findall(text or "")) < 2
