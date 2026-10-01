"""Does anything say THIS drug treats THIS patient's working diagnosis?

Why this exists
---------------
Measured 2026-09-30: a 55-year-old with exertional rhabdomyolysis was
recommended heparin, with a "DOSE VERIFIED" badge. Three separate things let it
through, and this module closes the first two:

  1. The old check asked whether the drug's NAME appeared in a retrieved
     guideline. Heparin did - in the VTE CPG, retrieved because "bilateral lower
     extremity pain" reads like a DVT. Nothing asked whether heparin treats
     rhabdomyolysis.
  2. A failed check only added a warning; the drug stayed on the card.
  3. The dose check never knew the indication had failed, so it verified 5000
     units against FUKKM and printed VERIFIED. (`rag_engine` now gates dosing
     behind this module: a withheld drug is never dose-checked.)

The question asked here is narrower and answerable in code: is there a source
sentence that links this drug to the WORKING diagnosis? Three sources, tried in
order of authority, and each verdict carries the sentence it rests on:

  SUPPORTED     the FUKKM listing's own `indication` field, or a sentence in a
                clinical guideline chunk retrieved for this patient, names the
                working diagnosis. Heparin 739 lists "venous thrombosis and
                pulmonary embolism ... myocardial infarction and arterial
                embolism" - so it passes for a DVT and an ACS, and fails for
                rhabdomyolysis.
  SYMPTOMATIC   the FUKKM indication is a symptom or a supportive purpose
                ("mild to moderate pain and pyrexia", "for replenishing fluid")
                that THIS intake or report actually has. Analgesia and fluids
                treat what the patient has, not what the patient has been
                diagnosed with, and must not be withheld for lack of a disease
                name.
  CONDITIONAL   the only link is to one of the report's DIFFERENTIALS, not the
                working diagnosis. Shown, but marked "only if <differential> is
                confirmed" - otherwise the model could license any drug by
                listing its indication as a differential.
  WITHHELD      no source links the drug to this patient. Removed from the drug
                list and named, with the reason, in the safety checks.

This module can only REMOVE or DOWNGRADE a recommendation. It never adds one,
so it cannot introduce a new error of commission.

Matching is by concept phrase, not by single words. "Heart failure" must not
match heparin's "heart surgery", and "cerebral infarction" must not match
"myocardial infarction": generic words (heart, infarction, failure, acute...)
only count as part of a two-word phrase, while specific words (rhabdomyolysis,
thrombosis, pneumonia, dengue...) count on their own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SUPPORTED = "SUPPORTED"
SYMPTOMATIC = "SYMPTOMATIC"
CONDITIONAL = "CONDITIONAL"
WITHHELD = "WITHHELD"

# Words that carry no diagnostic meaning on their own.
_STOP = frozenset("""
a an and are as at be by for from in into is it its of on or the to with without
acute chronic severe mild moderate probable possible suspected presumed likely
unspecified other related associated secondary primary disease diseases disorder
disorders syndrome syndromes condition conditions infection infections
treatment treat therapy prophylaxis prevention patients patient adults adult
children child including indicated used use management episode episodes type
due cause caused onset early late new known history exacerbation complicated
uncomplicated state stage grade phase
""".split())

# Specific only in a phrase: body parts and generic pathology words. "heart"
# alone would let heparin's "heart surgery" license it for heart failure.
_GENERIC = frozenset("""
heart cardiac kidney renal liver hepatic lung lungs pulmonary blood vascular
arterial artery venous vein cerebral brain skin bone bones muscle muscles
gastric abdominal chest urinary tract respiratory upper lower left right
infarction failure injury pain fever lesion lesions insufficiency dysfunction
inflammation bleeding haemorrhage hemorrhage obstruction ulcer shock
deep high low total partial major minor simple open closed systemic local
general multiple single recurrent persistent bilateral unilateral central
peripheral traumatic exertional induced drug decompensated compensated
worsening progressive
""".split())

# Clinical abbreviations and variant names, expanded on the diagnosis side so
# "ACS" meets FUKKM's "myocardial infarction". Vocabulary, not clinical rules:
# every entry says "these name the same thing", never "this drug treats that".
_SYNONYMS: dict[str, tuple[str, ...]] = {
    r"\bacs\b|acute coronary": ("myocardial infarction", "angina", "coronary"),
    r"\bstemi\b|\bnstemi\b|\bmi\b": ("myocardial infarction", "coronary"),
    r"\bdvt\b|deep vein": ("venous thrombosis", "thrombosis", "thromboembolism", "vte"),
    r"\bpe\b|pulmonary embol": ("pulmonary embolism", "embolism", "thromboembolism", "vte"),
    r"\bvte\b|thromboembol": ("thromboembolism", "venous thrombosis"),
    r"\baf\b|atrial fib": ("atrial fibrillation",),
    r"\bcopd\b|chronic obstructive": ("chronic obstructive pulmonary", "bronchitis", "bronchospasm"),
    r"asthma": ("asthma", "bronchospasm"),
    r"\bhf\b|heart failure|cardiac failure": ("heart failure", "cardiac failure", "oedema", "edema"),
    r"\bcap\b|pneumonia": ("pneumonia", "respiratory tract infection"),
    r"\buti\b|urinary tract inf|pyelonephritis|cystitis": ("urinary tract infection", "urinary tract"),
    r"\bdka\b|ketoacidosis": ("diabetic ketoacidosis", "diabetes mellitus", "hyperglycaemia"),
    r"\bhhs\b|hyperosmolar": ("diabetes mellitus", "hyperglycaemia"),
    r"stroke|\bcva\b|cerebrovascular": ("cerebral infarction", "stroke", "cerebrovascular"),
    r"hypertensi": ("hypertension", "hypertensive"),
    r"anaphyla": ("anaphylaxis", "anaphylactic", "allergic"),
    r"allergic rhinitis": ("allergic rhinitis", "rhinitis", "allergic"),
    r"seizure|epilep|convuls|status epilepticus": ("seizure", "seizures", "epilepsy", "convulsions", "status epilepticus"),
    r"snake|envenom|viper|cobra|krait": ("snake", "snakebite", "envenomation", "envenoming", "antivenom"),
    # Added 2026-09-30 with the MOH obstetric and psychiatric documents: their
    # titles name the disorder family, not the presentation.
    r"eclampsi|\bhellp\b|gestational hypertension|pregnancy[- ]induced hypertension":
        ("eclampsia", "hypertensive disorders", "hypertensive disorders in pregnancy"),
    r"post-?partum ha?emorrhag|\bpph\b|uterine atony": ("postpartum haemorrhage", "haemorrhage"),
    r"psychos[ie]s|psychotic|schizophren|agitat": ("schizophrenia", "psychosis", "psychoses", "agitation"),
    r"\bmani(?:a|c)\b|bipolar|lithium toxicity": ("bipolar disorder", "bipolar", "mania"),
    r"avuls\w* (?:tooth|teeth|incisor)|dental avuls|tooth avuls|knocked[- ]out tooth":
        ("avulsed", "avulsed permanent anterior teeth", "dental trauma"),
    r"condyl\w* fracture|mandib\w* fracture|jaw fracture": ("mandibular condyle fractures", "mandibular"),
    r"priapism": ("priapism", "erectile dysfunction"),
    r"overdos|poison|intoxicat|organophosph|paraquat": ("poisoning", "antidote", "antidotes", "overdose"),
    r"peripartum cardiomyopathy|heart disease in pregnancy|cardiac disease in pregnancy":
        ("heart disease in pregnancy", "cardiomyopathy"),
    r"dengue": ("dengue",),
    r"heat stroke|heatstroke|heat illness|hyperthermia": ("heat stroke", "hyperthermia", "heat"),
    r"hypoglycaem|hypoglycem": ("hypoglycaemia", "hypoglycemia"),
    r"hyperkalaem|hyperkalem": ("hyperkalaemia", "hyperkalemia"),
    r"sepsis|septic": ("sepsis", "septicaemia", "septicemia", "bacterial infection"),
    r"cellulitis": ("cellulitis", "skin and soft tissue", "soft tissue infection"),
    r"gastritis|dyspepsia|peptic": ("peptic ulcer", "dyspepsia", "gastritis", "reflux"),
}

# FUKKM indication phrases that are symptoms or supportive purposes, each with
# the evidence in THIS patient that would justify it.
_SYMPTOM_INDICATIONS: list[tuple[str, str, str]] = [
    # (indication regex, what the patient must show, label)
    (r"\bpain\b|analges", "pain", "pain"),
    (r"pyrexia|fever|antipyretic", "fever", "fever"),
    (r"nausea|vomiting|emesis", "vomiting", "nausea / vomiting"),
    (r"replenish\w* fluid|fluid (replacement|replenishment|resuscitation)|dehydration|"
     r"hypovolaem|hypovolem|restor\w*/?maintain\w* the concentration of sodium",
     "fluid", "fluid replacement"),
    (r"hypoxa?emia|hypoxia|oxygen", "hypoxia", "hypoxia"),
]


def _norm(text: str) -> str:
    t = (text or "").lower()
    t = t.replace("haem", "hem").replace("oedema", "edema").replace("aem", "em")
    return re.sub(r"[^a-z0-9 ]+", " ", t)


def _words(text: str) -> list[str]:
    return [w for w in _norm(text).split() if w not in _STOP and len(w) > 2]


def concepts(text: str, expand: bool = True) -> set[str]:
    """Matchable concept phrases for a diagnosis: specific single words, every
    adjacent pair of meaningful words, and (unless `expand` is False) synonym
    expansions."""
    words = _words(text)
    out = {w for w in words if w not in _GENERIC and len(w) > 3}
    out |= {f"{a} {b}" for a, b in zip(words, words[1:])}
    if not expand:
        return {c for c in out if c}
    return {c for c in out | expansions(text) if c}


def expansions(text: str, parents: bool = False) -> set[str]:
    """What `text` names under another word: the _SYNONYMS table, then the
    condition cards' own names and aliases (cards.synonyms) - and, with
    `parents`, the broader condition a card belongs to."""
    low = (text or "").lower()
    out: set[str] = set()
    for pattern, names in _SYNONYMS.items():
        if re.search(pattern, low):
            for e in names:
                ew = _words(e)
                if len(ew) == 1 and ew[0] in _GENERIC:
                    continue
                out.add(" ".join(ew))
    from . import cards  # noqa: PLC0415 - cards imports this module
    out |= cards.synonyms(text, parents=parents)
    return {o for o in out if o}


def mentions(concept_set: set[str], text: str, symmetric: bool = False) -> str | None:
    """The first concept found in `text` as a whole-word phrase, else None.

    `symmetric` expands `text` too. The diagnosis side was always expanded
    ("ACS" -> myocardial infarction, coronary), the passage side never was:
    on 2026-09-30 the ACS CPG p31 sentence that verified aspirin 300 mg also
    said "180mg ticagrelor", but it said "STEMI", so ticagrelor was WITHHELD
    for a shocked STEMI. A passage's own abbreviation is expanded the same
    way, and a passage about a subtype (STEMI) speaks for its parent (ACS)."""
    hay = " " + " ".join(_words(text)) + " "
    ordered = sorted(concept_set, key=len, reverse=True)
    for c in ordered:
        if f" {c} " in hay:
            return c
    if symmetric:
        extra = expansions(text, parents=True)
        for c in ordered:
            if c in extra:
                return c
    return None


@dataclass
class Verdict:
    status: str
    basis: str          # the sentence or field the verdict rests on, quoted
    concept: str = ""   # which diagnosis concept it matched


@dataclass
class PatientSignals:
    """What the intake and the report show, for the SYMPTOMATIC path."""
    pain: bool = False
    fever: bool = False
    vomiting: bool = False
    fluid: bool = False
    hypoxia: bool = False

    def has(self, key: str) -> bool:
        return bool(getattr(self, key, False))


def signals(req, diagnostic) -> PatientSignals:
    v = req.vitals
    text = " ".join([
        req.complaint or "", req.history or "",
        " ".join(a.action for a in diagnostic.immediate_actions),
    ]).lower()
    return PatientSignals(
        pain=(v.pain_score or 0) >= 1 or bool(re.search(r"\bpain|ache|colic|tender", text)),
        fever=(v.temperature is not None and v.temperature >= 37.5) or "fever" in text or "febrile" in text,
        vomiting=bool(re.search(r"vomit|nausea|emesis", text)),
        fluid=bool(re.search(r"\bfluid|saline|crystalloid|hartmann|ringer|rehydrat|resuscitat|dehydrat|shock",
                             text)),
        hypoxia=(v.spo2 is not None and v.spo2 < 94) or bool(re.search(r"hypox|oxygen|\bo2\b|desaturat", text)),
    )


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;:])\s+|\n+|•|", text or "")
    return [re.sub(r"\s+", " ", p).strip() for p in parts if len(p.strip()) > 15]


def judge(
    drug_pattern: re.Pattern,
    fukkm_entries: list,
    working: set[str],
    differential: set[str],
    guideline_chunks: list[tuple[str, str]],
    patient: PatientSignals,
    drug_is_generic_fluid: bool = False,
) -> Verdict:
    """Decide one drug.

    drug_pattern       matches the drug's names in prose (rag_engine._alias_pattern)
    fukkm_entries      the FUKKM rows for this drug (formulary Match.entries)
    working            concepts of the working (primary) diagnosis
    differential       concepts of the differentials
    guideline_chunks   (citation, text) of CLINICAL guideline chunks retrieved
                       for this patient - never the formulary, never an HTA report
    """
    # 0. A generic fluid order ("IV fluids", "isotonic crystalloid") names no
    # single FUKKM product, so it can have no FUKKM row - withholding it for
    # that reason removed the one treatment rhabdomyolysis needs (2026-09-30).
    # It is judged as fluid replacement, against the patient.
    if drug_is_generic_fluid and patient.has("fluid"):
        return Verdict(SYMPTOMATIC, "fluid replacement - an action in this report calls for IV fluids", "fluid")
    # 1. The formulary's own indication field.
    for e in fukkm_entries:
        hit = mentions(working, e.indication, symmetric=True)
        if hit:
            return Verdict(SUPPORTED, f"FUKKM {e.fukkm_no} indication: \"{_clip(e.indication, hit)}\"", hit)
    # 2. A guideline sentence naming both the drug and the working diagnosis.
    for cite, text in guideline_chunks:
        for s in _sentences(text):
            if drug_pattern.search(s):
                hit = mentions(working, s, symmetric=True)
                if hit:
                    return Verdict(SUPPORTED, f"{cite}: \"{_clip(s, hit)}\"", hit)
    # 3. A symptom or supportive purpose this patient actually has.
    for e in fukkm_entries:
        for pattern, need, label in _SYMPTOM_INDICATIONS:
            if re.search(pattern, e.indication or "", re.I) and patient.has(need):
                return Verdict(SYMPTOMATIC,
                               f"FUKKM {e.fukkm_no} indication: \"{_clip(e.indication, None)}\" - "
                               f"this patient has {label}", label)
    # 4. Only a differential.
    for e in fukkm_entries:
        hit = mentions(differential, e.indication)
        if hit:
            return Verdict(CONDITIONAL, f"FUKKM {e.fukkm_no} indication: \"{_clip(e.indication, hit)}\"", hit)
    for cite, text in guideline_chunks:
        for s in _sentences(text):
            if drug_pattern.search(s):
                hit = mentions(differential, s, symmetric=True)
                if hit:
                    return Verdict(CONDITIONAL, f"{cite}: \"{_clip(s, hit)}\"", hit)
    listed = "; ".join(_clip(e.indication, None, 160) for e in fukkm_entries[:1] if e.indication)
    return Verdict(WITHHELD, f"FUKKM indications: \"{listed}\"" if listed else
                   "no FUKKM listing and no retrieved guideline sentence")


def _clip(text: str, around: str | None, width: int = 220) -> str:
    t = re.sub(r"\s+", " ", text or "").strip()
    if len(t) <= width:
        return t
    if around:
        i = _norm(t).find(around.split()[0])
        if i > 0:
            start = max(0, i - width // 3)
            return ("..." if start else "") + t[start:start + width].strip() + "..."
    return t[:width].strip() + "..."


GENERIC_FLUID_RE = re.compile(
    r"^\s*(?:iv|intravenous)?\s*(?:isotonic\s+)?(?:fluids?|crystalloids?|fluid resuscitation|"
    r"fluid therapy|rehydration)\s*$", re.I)
