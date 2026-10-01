"""Attention layer A1: what in this intake deserves the model's attention.

Why this exists
---------------
Measured 2026-09-30 on the rhabdomyolysis case: the model was handed 27
passages / 8,165 tokens, ~44% of it noise - ten MTS table cells for a level the
code already decides, four formulary rows for antivenoms and clotting factors,
snakebite and haemophilia pages. Its answers followed its attention: it cited
MTS page 7 for IV fluids, named the snakebite guideline as governing, proposed
antivenom. Transformer attention is not something this system can retrain, but
WHAT the model attends to is entirely in our hands. Findings from the long-
context literature point one way (Liu et al., TACL 2024, "Lost in the Middle";
LongLLMLingua, ACL 2024; RECOMP, ICLR 2024): fewer, denser, question-focused
inputs, with the key facts at the start and end of the prompt.

This module does the first, cheapest part - deciding what the key facts ARE,
in code, from the intake:

  findings    each clause of the complaint and history, with the red-flag rule
              it fires (weight 3), abnormal vital signs from the MTS table
              (weight 3 / 2), other complaint clauses (2), history (1.5).
  negatives   clauses the intake NEGATES ("no fever", "denies chest pain").
              A ConText/NegEx-style rule set (Chapman et al.; medspaCy's
              implementation), kept small and in-house: a negated finding is
              shown as negative, never searched for, and never fires a red flag.
  missing     core vital signs, ECG and glucose not recorded.

`render()` turns this into the KEY FINDINGS block at the top of the prompt, and
`reminder()` into a one-line repeat just before the task - the two positions
the literature finds models attend to most.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import mts_table, red_flags

# NegEx-style pre-negation triggers. Scope runs to the end of the clause.
_PRE_NEG = re.compile(
    r"\b(?:no|not|denies|denied|denying|without|never|negative for|absence of|"
    r"free of|nil|non[- ]?)\b", re.IGNORECASE)
# Pseudo-negations: phrases containing a trigger that do NOT negate a finding.
_PSEUDO = re.compile(
    r"\b(?:no (?:change|improvement|better|relief|response|longer)|not (?:only|improving|resolved|relieved)|"
    r"without (?:relief|improvement))\b", re.IGNORECASE)
# Clause boundaries: a negation does not reach past these.
_CLAUSE = re.compile(r"[.;,]|\bbut\b|\bhowever\b|\balthough\b|\bexcept\b", re.IGNORECASE)


def split_clauses(text: str) -> list[str]:
    parts = re.split(r"[.;]|,(?![^()]*\))|\b(?:and|with|plus|also)\b", text or "", flags=re.IGNORECASE)
    return [re.sub(r"\s+", " ", p).strip(" -:") for p in parts if len(p.strip()) > 2]


def is_negated(clause: str) -> bool:
    if _PSEUDO.search(clause or ""):
        return False
    return bool(_PRE_NEG.search(clause or ""))


def strip_negated(text: str) -> str:
    """Remove negated spans (trigger to clause end) so pattern-based rules - the
    red-flag recall floor above all - do not fire on "no chest pain"."""
    out, pos = [], 0
    s = text or ""
    for m in _PRE_NEG.finditer(s):
        if _PSEUDO.match(s, m.start()):
            continue
        if m.start() < pos:
            continue
        end = _CLAUSE.search(s, m.end())
        stop = end.start() if end else len(s)
        out.append(s[pos:m.start()])
        out.append(" ")
        pos = stop
    out.append(s[pos:])
    return "".join(out)


@dataclass
class Finding:
    text: str
    kind: str            # complaint | history | vital
    weight: float
    red_flag: str = ""   # the red-flag rule this finding fires, as words


@dataclass
class Focus:
    findings: list[Finding] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def ranked(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: -f.weight)

    def queries(self, limit: int = 4) -> list[str]:
        """The highest-weight findings, as separate search queries (A2)."""
        return [f.text for f in self.ranked() if f.kind != "vital"][:limit]


_CORE = (("systolic_bp", "BP"), ("heart_rate", "HR"), ("respiratory_rate", "RR"),
         ("spo2", "SpO2"), ("temperature", "temperature"))


def extract(req) -> Focus:
    focus = Focus()
    # Rules are matched on the WHOLE (negation-stripped) intake, because some
    # combine clauses - "muscle weakness" fires exertional_muscle_injury only
    # with "marathons" from the history. A clause is tagged with every rule
    # whose triggering phrase lies inside it.
    fired = red_flags.match(strip_negated(f"{req.complaint}. {req.history}"))
    for source, kind, base in ((req.complaint, "complaint", 2.0), (req.history, "history", 1.5)):
        for clause in split_clauses(source):
            if is_negated(clause):
                focus.negatives.append(clause)
                continue
            low = clause.lower()
            labels = [flag.label for flag, hit in fired if hit and hit.lower() in low]
            focus.findings.append(Finding(clause, kind, 3.0 if labels else base, ", ".join(labels)))
    v = req.vitals
    for f in mts_table.evaluate(req.age, v, f"{req.complaint} {req.history}", req.observations()):
        if f.level <= 3:
            focus.findings.append(Finding(f"{f.observed} (MTS L{f.level})", "vital",
                                          3.0 if f.level <= 2 else 2.0))
    focus.missing = [label for fld, label in _CORE if getattr(v, fld) is None]
    if not req.ecg_findings:
        focus.missing.append("ECG")
    if v.capillary_blood_glucose is None:
        focus.missing.append("capillary glucose")
    return focus


def render(focus: Focus, triage_line: str) -> str:
    lines = ["KEY FINDINGS (extracted by code from the intake - weigh these first):"]
    for f in focus.ranked():
        # The rule name goes AFTER the finding: placed first, the model copied
        # "exertional muscle injury" into red_flags as though it were a finding.
        prefix = "[history] " if f.kind == "history" else ""
        suffix = f"  (red-flag rule: {f.red_flag})" if f.red_flag else ""
        lines.append(f"- {prefix}{f.text}{suffix}")
    if focus.negatives:
        lines.append("NEGATIVE (stated absent - do not treat as present): " + "; ".join(focus.negatives))
    if focus.missing:
        lines.append("NOT MEASURED: " + ", ".join(focus.missing))
    lines.append(f"TRIAGE (decided by code from MTS 2022 - do not re-derive, do not cite it for treatment): "
                 f"{triage_line}")
    # A4.3: "describe THIS patient's finding" produced red flags that restated
    # the complaint in all seven rhabdomyolysis runs; a red flag is a danger.
    lines.append("A red flag is a DANGER this presentation carries - a complication or a can't-miss "
                 "cause to act on - not a restatement of the complaint or the diagnosis, and never "
                 "the name of a rule.")
    return "\n".join(lines)


def reminder(focus: Focus) -> str:
    top = [f.text for f in focus.ranked()[:4]]
    return ("REMEMBER the key findings: " + "; ".join(top)
            + ". Base the diagnosis and every recommendation on THESE, citing the [S#] that supports each.")
