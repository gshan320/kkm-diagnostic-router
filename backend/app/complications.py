"""Attention layer A4.3: red flags are the dangers, anchored to a source sentence.

Why this exists
---------------
In all seven live runs of the rhabdomyolysis case (2026-09-30) the model's
red flags restated the diagnosis - "Rhabdomyolysis", "Exertional muscle
injury", "Dark tea-coloured urine" - and never named hyperkalaemia or
compartment syndrome, the complications that kill. The information was in the
corpus the whole time: the Snakebite guideline, correctly left out of the
prompt as a document, says in general terms "In skeletal muscle breakdown
(rhabdomyolysis), hyperkalemia can also lead to cardiac arrest" (p63).

So, for a working diagnosis with a profile here:

  1. each complication the report's red flags do not already name is looked
     up as a VERBATIM sentence - first in the passages the model was given,
     then anywhere in the index - that is about the condition itself
     (`about`), states the complication (`evidence`), and names no cause the
     patient does not have (a sea-snake sentence never speaks for a runner);
  2. found: added as a red flag whose "why" IS that sentence, cited to its
     page. Not found: nothing is added - silence, never a guess;
  3. a red flag that only restates the diagnosis or the presenting complaint is
     removed once at least one real complication is listed.

Profiles are curated and small on purpose, like every clinical table in this
codebase; each one's quotes are checked against the index by the tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A sentence naming one of these speaks about THAT cause - unless the intake
# names it too.
CAUSE_SPECIFIC = re.compile(
    r"snake|venom|envenom|\bbites?\b|\bsting|dengue|statin|risperidone|hydrophi|malaria|"
    r"leptospir|pregnan|neonat|\bchild", re.I)


@dataclass(frozen=True)
class Complication:
    label: str      # the red flag as shown
    present: str    # regex: the report's red flags already name it
    evidence: str   # regex: a source sentence states it


@dataclass(frozen=True)
class Caution:
    """F2: a "do not" the source states for this condition, quoted verbatim -
    only from a document whose title is about the condition."""
    label: str      # shown, e.g. "Avoid NSAIDs"
    evidence: str   # regex the source sentence must match
    when: str = ""  # regex on the intake: only for this patient if it matches


@dataclass(frozen=True)
class Profile:
    name: str
    diagnosis: str  # regex on the working diagnosis
    about: str      # regex: the sentence is about this condition
    restates: str   # regex on a red flag's name: only the diagnosis or complaint
    complications: tuple[Complication, ...]
    # The condition's abbreviation, trusted ONLY inside a document whose title
    # is about the condition: the HDP manual writes "PE" for pre-eclampsia,
    # everywhere else "PE" is pulmonary embolism.
    abbrev: str = ""
    gloss: str = ""   # shown after a quote that uses the abbreviation
    cautions: tuple[Caution, ...] = ()
    # Titles that are about this condition even though they do not name it
    # (lithium toxicity lives in the Bipolar Disorder CPG).
    titles: str = ""
    # F7: the test that DEFINES the diagnosis - until the intake reports a
    # result, confidence cannot be HIGH. (regex for a result in the intake,
    # label, verbatim definition, source)
    defining_test: tuple[str, str, str, str] | None = None


_RHABDO = r"rh\w{0,3}omyoly\w*|myoglobin\w*|skeletal muscle breakdown"
# ACUTE only: "patients with chronic diseases such as ... renal failure are more
# susceptible" names renal failure as a risk factor, not a complication.
_ACUTE_RENAL = r"acute\s+(?:renal\s+(?:failure|injury|insufficiency)|kidney\s+injury)|\bAKI\b"
# Prose states a relation; a table cell ("Rhabdomyolysis causing renal failure
# & hyperkalaemia Paralysis 11", a sea-snake table) only lists words.
PROSE = re.compile(r"\b(?:can|may|might|lead|leads|cause|causes|include|includes|result|results|is|are|"
                   r"was|were|occur|occurs|develop|develops|should|must|due)\b", re.I)

PROFILES: tuple[Profile, ...] = (
    Profile(
        name="rhabdomyolysis",
        diagnosis=r"rh\w{0,3}omyoly|myoglobinuri",
        about=_RHABDO,
        restates=(r"^\s*(?:exertional\s+)?(?:rhabdomyolysis|myoglobinuria|muscle\s+(?:injury|damage|breakdown|"
                  r"pain|weakness)|exertional\s+muscle\s+injury|dark|tea|cola|generali[sz]ed\s+muscle|"
                  r"severe\s+(?:bilateral\s+)?(?:lower\s+)?(?:extremity|limb|leg)\s+pain)"),
        complications=(
            Complication("Hyperkalaemia - risk of cardiac arrhythmia or arrest",
                         r"hyperkal|potassium|arrhythm|cardiac arrest", r"hyperkal\w*"),
            Complication("Acute kidney injury", r"kidney|renal|\baki\b", _ACUTE_RENAL),
            Complication("Compartment syndrome", r"compartment", r"(?<!abdominal )compartment syndrome"),
        ),
        # No KKM sentence states these for rhabdomyolysis yet; the regex only
        # quotes a KKM document about the condition, so it stays silent until
        # one does. (CHAMP 2025, which did, left the index on 2026-09-30 under
        # the source policy - see source_policy.py.)
        cautions=(
            Caution("Avoid NSAIDs", r"NSAIDs? should be avoided"),
        ),
        # The one KKM definition: statin-associated muscle symptoms, graded by CK.
        defining_test=(
            r"\b(?:CK|CPK|creatine (?:phospho)?kinase)\b\D{0,15}\d[\d,]{2,}",
            "creatine kinase (CK)",
            "this includes myalgia ( ck normal ), myositis ( ck > uln ) and rhabdomyolysis ( ck > 10x of uln )",
            "Management of Dyslipidaemia (6th Edition) 2023 p59",
        ),
    ),
    Profile(
        name="heat stroke",
        diagnosis=r"heat\s*stroke|heat[- ]related|heat\s+(?:illness|injury|exhaustion)|hyperthermi",
        about=r"heat\s*stroke|heat[- ]related|hyperthermi",
        restates=r"^\s*(?:heat\s*stroke|heat[- ]related|heat\s+(?:illness|injury|exhaustion)|hyperthermi\w*|"
                 r"high\s+(?:body\s+)?temperature|fever)",
        complications=(
            Complication("Rhabdomyolysis", _RHABDO, r"rh\w{0,3}omyoly\w*"),
            Complication("Acute kidney injury", r"kidney|renal|\baki\b", _ACUTE_RENAL),
            Complication("Disseminated intravascular coagulation", r"\bdic\b|\bdivc\b|coagul",
                         r"\bDIC\b|\bDIVC\b|disseminated intravascular|coagulopath\w*"),
            Complication("Cardiac arrhythmia", r"arrhythm", r"arrhythmi\w*"),
        ),
    ),
    Profile(
        name="hyperkalaemia",
        diagnosis=r"hyperkal",
        about=r"hyperk\w*|potassium|\bK\+",
        restates=r"^\s*(?:hyperkal\w*|high\s+potassium|raised\s+potassium)",
        complications=(
            Complication("Cardiac arrhythmia or arrest", r"arrhythm|cardiac arrest|ventricular",
                         r"arrhythmi\w*|cardiac arrest"),
        ),
    ),
)


PROFILES = PROFILES + (
    # ---- Added 2026-09-30 with the MOH obstetric and psychiatric documents.
    Profile(
        name="postpartum haemorrhage",
        diagnosis=r"post-?partum ha?emorrhag|\bPPH\b|uterine atony",
        about=r"\bPPH\b|post-?partum|obstetric|uter(?:us|ine)|after (?:delivery|childbirth)|placenta\w*",
        restates=r"^\s*(?:(?:primary |secondary )?post-?partum ha?emorrhag\w*|\bPPH\b|(?:heavy |vaginal )*bleeding|"
                 r"blood loss|uterine atony)",
        complications=(
            # Hypovolaemic, not any shock: "neurogenic shock" from uterine
            # inversion was quoted first (2026-09-30).
            Complication("Hypovolaemic shock", r"shock|hypovol\w*|hypotens\w*",
                         r"(?:hypovol\w*|ha?emorrhagic) shock"),
            Complication("Coagulopathy / DIC", r"\bDIC\b|\bDIVC\b|coagul\w*",
                         r"\bDIC\b|\bDIVC\b|disseminated intravascular|coagulopath\w*"),
        ),
        cautions=(
            Caution("Ergometrine / Syntometrine contraindicated in hypertension or cardiac disease",
                    r"contraindicated in patients with cardiac disease or hypertension"),
        ),
    ),
    Profile(
        name="eclampsia / severe pre-eclampsia",
        diagnosis=r"eclampsi|\bHELLP\b",
        about=r"eclampsi\w*|pre-?eclampsi\w*|\bHELLP\b|hypertensive disorders?",
        restates=r"^\s*(?:(?:severe |imminent )?(?:pre-?)?eclampsi\w*|hypertensi\w*|high (?:blood )?pressure|"
                 r"(?:severe )?headache|blurr\w* vision|seizure|fit)",
        complications=(
            Complication("HELLP syndrome", r"\bHELLP\b|haemolysis|low platelet", r"\bHELLP\b"),
            Complication("Pulmonary oedema", r"pulmonary o?edema", r"pulmonary o?edema"),
            Complication("Intracranial haemorrhage / stroke", r"stroke|intracranial|cerebral ha?emorrhag|\bCVA\b",
                         r"cerebral ha?emorrhag\w*|intracranial ha?emorrhag\w*|stroke|\bCVA\b"),
            Complication("Placental abruption", r"abruption|abruptio", r"abruptio\w*"),
        ),
        cautions=(
            Caution("Give the MgSO4 loading dose slowly", r"rapid injection causes cardiac arrest"),
            Caution("Magnesium toxicity - antidote is calcium gluconate", r"antidote is 10% cal\w* gluconate"),
        ),
        # "PE" is pre-eclampsia in the HDP manual and pulmonary embolism in the VTE CPG.
        abbrev=r"\bpe\b",
        gloss="PE = pre-eclampsia in this guideline",
    ),
    Profile(
        name="acute agitation / psychosis",
        diagnosis=r"agitat|psychos[ie]s|psychotic|schizophren",
        about=r"antipsychotic\w*|neuroleptic|schizophren\w*|psychos[ie]s",
        restates=r"^\s*(?:(?:acute |severe )?(?:agitat\w*|psychos[ie]s|psychotic\w*|aggressi\w*|violen\w*)|"
                 r"hearing voices|hallucinat\w*|schizophren\w*)",
        complications=(
            Complication("Neuroleptic malignant syndrome (with antipsychotics)", r"neuroleptic malignant|\bNMS\b",
                         r"neuroleptic malignant"),
        ),
        abbrev=r"\baps?\b",
        gloss="APs = antipsychotics",
    ),
    Profile(
        name="lithium toxicity",
        diagnosis=r"lithium toxic|lithium (?:overdose|poisoning)",
        about=r"lithium toxicity|toxic lithium|lithium (?:level|concentration)",
        restates=r"^\s*lithium (?:toxicity|overdose|poisoning)",
        titles=r"bipolar",
        complications=(
            Complication("Seizures", r"seizure|convuls|fit", r"seizure\w*"),
        ),
        cautions=(
            Caution("Lithium level rises with dehydration", r"caution use during periods of dehydration"),
        ),
    ),
    Profile(
        name="gout flare",
        diagnosis=r"\bgout|podagra",
        about=r"\bgout\w*",
        restates=r"^\s*(?:acute )?gout\w*",
        complications=(),
        cautions=(
            Caution("Colchicine avoided in severe CKD", r"colchicine should be avoided in severe CKD",
                    when=r"\bCKD\b|chronic kidney|renal (?:impairment|failure)|eGFR|dialysis"),
            Caution("NSAIDs avoided in CKD", r"NSAIDs should also be avoided in CKD",
                    when=r"\bCKD\b|chronic kidney|renal (?:impairment|failure)|eGFR|dialysis"),
        ),
    ),
)


def profile_for(diagnosis: str) -> Profile | None:
    for p in PROFILES:
        if re.search(p.diagnosis, diagnosis or "", re.I):
            return p
    return None


def missing(profile: Profile, red_flag_texts: list[str]) -> list[Complication]:
    joined = " ".join(red_flag_texts)
    return [c for c in profile.complications if not re.search(c.present, joined, re.I)]


def restates(profile: Profile, flag_name: str) -> bool:
    if not re.search(profile.restates, flag_name or "", re.I):
        return False
    # A flag that also names a complication is not a mere restatement.
    return not any(re.search(c.present, flag_name or "", re.I) for c in profile.complications)


def preference(sentence: str, title: str, external: bool = False, on_topic: bool = True) -> tuple:
    """Sort key among qualifying index sentences: a KKM document before a
    non-KKM one (where both speak, KKM wins), then a document ABOUT the
    condition (a PPH shock sentence from the GI-bleeding CPG lost to the PPH
    guide), then prose, then a document not about a specific cause, then a
    readable length, then shorter."""
    return (external, not on_topic, not PROSE.search(sentence), bool(CAUSE_SPECIFIC.search(title or "")),
            not 60 <= len(sentence) <= 260, len(sentence))


def on_topic(profile: Profile, title: str) -> bool:
    t = title or ""
    return bool(re.search(profile.about, t, re.I) or re.search(profile.diagnosis, t, re.I)
                or (profile.titles and re.search(profile.titles, t, re.I)))


def quote_around(text: str, m: re.Match, span: int = 220) -> str:
    """The sentence (or table cell) holding a match, from whitespace-flattened
    text: back to the previous full stop / bullet, forward to the next full
    stop - at most `span` characters each way. Sentence splitting failed on
    table-like pages (the PPH Syntometrine row, the gout CKD sentence)."""
    start = max(0, m.start() - span)
    head = text[start:m.start()]
    cut = max(head.rfind(". "), head.rfind("\u2022"), head.rfind("\u25cf"), head.rfind("; "))
    if cut >= 0:
        begin = start + cut + 2
    else:
        # A table row has no sentence start: keep the cell, not the row.
        begin = max(start, m.start() - 90)
        sp = text.find(" ", begin)
        begin = sp + 1 if 0 <= sp < m.start() else begin
    tail = text[m.end():m.end() + span]
    stop = next((x.start() for x in re.finditer(r"\. |\s[\u2022\u25cf]", tail)
                 if not re.search(r"\b(?:e\.g|i\.e|vs|etc|approx)$", tail[:x.start()], re.I)), -1)
    # No sentence end: a table cell - the matched phrase completes it.
    end = m.end() + (stop + 1 if stop >= 0 else 0)
    return text[begin:end].strip(" .;:\u2022\u25cf") + "."


def gloss_for(profile: Profile, sentence: str) -> str:
    """The abbreviation's meaning, when the quote uses it without the full term."""
    if profile.abbrev and profile.gloss and re.search(profile.abbrev, sentence, re.I) \
            and not re.search(profile.about, sentence, re.I):
        return f" ({profile.gloss})"
    return ""


def qualifies(sentence: str, profile: Profile, comp: Complication, intake: str, title: str = "") -> bool:
    """A sentence may speak for this patient's complication."""
    if not 30 <= len(sentence) <= 320:
        return False
    about = re.search(profile.about, sentence, re.I) or (
        profile.abbrev and re.search(profile.about, title or "", re.I)
        and re.search(profile.abbrev, sentence, re.I))
    if not (about and re.search(comp.evidence, sentence, re.I)):
        return False
    cause = CAUSE_SPECIFIC.search(sentence)
    if cause and not re.search(re.escape(cause.group(0)), intake or "", re.I):
        return False
    return True
