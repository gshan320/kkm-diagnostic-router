"""Condition cards at run time: what the corpus says about THIS diagnosis.

Why this exists
---------------
The completeness checklist, the complication anchor, the source cautions and
the quoted disposition floor were each a hand-written table covering between
one and 21 conditions. A diagnosis outside those tables got none of them. The
cards (built by cards_build.py, every quote verified against the index) carry
the same kinds of knowledge for every condition the corpus gives management
for, so each of those checks now has an answer for any diagnosis the corpus
covers - and, for one it does not, a statement that KKM is silent.

The hand-written tables stay where they exist: each was tuned against a real
failure and is tested against its own page. A card is used where no hand rule
matches, and for the kinds of knowledge the hand rules never had.

Nothing here writes clinical text. Every item returned is a verbatim sentence
with its document and page; the caller decides what to do with it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache

from . import config, indications

CARDS_FILE = config.DATA_DIR / "condition_cards.json"

# Element order for display, and which elements an ED report is judged on.
ELEMENTS = ("definition", "red_flag", "complication", "investigation", "treatment",
            "avoid", "admission", "referral", "discharge", "monitoring")
CHECKLIST_ELEMENTS = ("investigation", "treatment", "monitoring")
ED_SETTINGS = ("ed", "any")
ELEMENT_LABEL = {
    "definition": "diagnostic criteria", "red_flag": "red flags", "complication": "complications",
    "investigation": "investigations", "treatment": "treatment", "avoid": "what to avoid",
    "admission": "admission criteria", "referral": "referral criteria",
    "discharge": "discharge criteria", "monitoring": "monitoring",
}


@dataclass(frozen=True)
class Item:
    element: str
    label: str
    terms: tuple[str, ...]
    when: tuple[str, ...]
    unless: tuple[str, ...]
    core: bool
    setting: str
    population: str
    quote: str
    chunk_id: str
    title: str
    doc_type: str
    page: int | None
    url: str
    external: bool

    @property
    def where(self) -> str:
        if self.page:
            return f"{self.title} p{self.page}"
        return self.title


@dataclass(frozen=True)
class Card:
    id: str
    name: str
    aliases: tuple[str, ...]
    parents: tuple[str, ...]
    population: str
    acuity: str
    documents: tuple[str, ...]
    items: tuple[Item, ...]

    def of(self, element: str) -> list[Item]:
        return [i for i in self.items if i.element == element]


@lru_cache(maxsize=1)
def load() -> tuple[Card, ...]:
    if not CARDS_FILE.exists():
        return ()
    rows = json.loads(CARDS_FILE.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        items = tuple(Item(
            element=i["element"], label=i["label"], terms=tuple(i.get("terms") or ()),
            when=tuple(i.get("when") or ()), unless=tuple(i.get("unless") or ()),
            core=bool(i.get("core")), setting=i.get("setting") or "any",
            population=i.get("population") or "", quote=i["quote"], chunk_id=i["chunk_id"],
            title=i.get("title", ""), doc_type=i.get("doc_type", ""), page=i.get("page"),
            url=i.get("url", ""), external=bool(i.get("external")),
        ) for i in r.get("items") or [])
        out.append(Card(id=r["id"], name=r["name"], aliases=tuple(r.get("aliases") or ()),
                        parents=tuple(r.get("parents") or ()), population=r.get("population") or "all",
                        acuity=r.get("acuity") or "urgent", documents=tuple(r.get("documents") or ()),
                        items=items))
    return tuple(out)


# ------------------------------------------------------------ vocabulary
def _key(text: str) -> str:
    """The matching form of a condition name: indications' word normalisation
    (stop words and punctuation removed, haem/hem unified)."""
    return " ".join(indications._words(text))


@lru_cache(maxsize=1)
def _names() -> dict[str, list[int]]:
    """Normalised name or alias -> indexes of the cards it names."""
    table: dict[str, list[int]] = {}
    for i, c in enumerate(load()):
        for n in (c.name,) + c.aliases:
            k = _key(n)
            # A one-word key must be specific: "shock" or "pain" alone would
            # tie a card to every sentence that mentions them.
            if not k or (" " not in k and (len(k) < 3 or k in indications._GENERIC)):
                continue
            table.setdefault(k, [])
            if i not in table[k]:
                table[k].append(i)
    return table


@lru_cache(maxsize=1)
def _max_words() -> int:
    return max((k.count(" ") + 1 for k in _names()), default=1)


def _phrases(text: str) -> list[tuple[str, int]]:
    """(name, card index) for every card name or alias in `text`, as whole
    word sequences."""
    words = indications._words(text)
    names, n = _names(), _max_words()
    hits = []
    for size in range(min(n, len(words)), 0, -1):
        for start in range(len(words) - size + 1):
            k = " ".join(words[start:start + size])
            for idx in names.get(k, ()):
                hits.append((k, idx))
    return hits


def synonyms(text: str, parents: bool = False) -> set[str]:
    """Card names reached from `text` through the card vocabulary: the name
    of every card whose name or alias appears in it - and, with `parents`,
    each such card's broader condition. Used by indications.mentions to
    expand a PASSAGE the way the diagnosis is expanded, so a sentence about
    "STEMI" can support a drug for "acute coronary syndrome"."""
    cards = load()
    if not cards:
        return set()
    out = set()
    for _, idx in _phrases(text):
        c = cards[idx]
        out.add(_key(c.name))
        if parents:
            out |= {_key(p) for p in c.parents}
    return {o for o in out if o}


# ---------------------------------------------------------------- lookup
def _population_fits(card_pop: str, age: float | None, pregnant: bool) -> bool:
    if age is None:
        return True
    if card_pop == "paediatric":
        return age < 18
    if card_pop == "adult":
        return age >= 12
    if card_pop == "obstetric":
        return pregnant
    return True


def lookup(diagnosis: str, age: float | None = None, pregnant: bool = False) -> Card | None:
    """The card that names this diagnosis. An exact name or alias beats a
    contained one; a longer contained phrase beats a shorter; a card whose
    population fits the patient beats one that does not (which is never
    returned)."""
    cards = load()
    if not cards or not (diagnosis or "").strip():
        return None
    whole = _key(diagnosis)
    best: tuple[tuple, int] | None = None
    for k, idx in _phrases(diagnosis):
        c = cards[idx]
        if not _population_fits(c.population, age, pregnant):
            continue
        score = (k == whole, k.count(" ") + 1, len(k), len(c.items))
        if best is None or score > best[0]:
            best = (score, idx)
    if best is None:
        best = _contained(whole, age, pregnant)
    return cards[best[1]] if best else None


def _contained(whole: str, age: float | None, pregnant: bool) -> tuple | None:
    """The other direction: the diagnosis is a shorter form of a card's name.
    "Rhabdomyolysis" found no card on 2026-10-01 because the card is named
    "Exertional rhabdomyolysis" - so its admission criterion never fired and the
    report said KKM gives no management. The closest name wins (fewest extra
    words), then the fuller card. A generic one-word diagnosis never matches."""
    words = set(whole.split())
    if not words or (len(words) == 1 and (len(whole) < 5 or whole in indications._GENERIC)):
        return None
    cards = load()
    best = None
    for k, idxs in _names().items():
        kw = set(k.split())
        if not words <= kw:
            continue
        for idx in idxs:
            if not _population_fits(cards[idx].population, age, pregnant):
                continue
            score = (-(len(kw) - len(words)), len(cards[idx].items))
            if best is None or score > best[0]:
                best = (score, idx)
    return best


def lineage(card: Card, age: float | None = None, pregnant: bool = False) -> list[Card]:
    """The card and its broader conditions (one level), most specific first:
    a STEMI card's items and then the ACS card's."""
    out = [card]
    for p in card.parents:
        parent = lookup(p, age, pregnant)
        if parent is not None and parent.id not in {c.id for c in out}:
            out.append(parent)
    return out


# ------------------------------------------------------------ patient state
def _stem(word: str) -> str:
    return word[:7] if len(word) >= 7 else word


def _state_words(state: str) -> set[str]:
    return {_stem(w) for w in indications._norm(state).split() if len(w) > 1}


def state_hit(phrases: tuple[str, ...], state: str) -> str | None:
    """The first phrase whose every word occurs in the patient state (stemmed:
    "hypotension" meets "hypotensive"), else None."""
    words = _state_words(state)
    for p in phrases:
        ws = [w for w in indications._norm(p).split() if len(w) > 1]
        if ws and all(_stem(w) in words for w in ws):
            return p
    return None


# G4 (2026-09-30): the shocked-STEMI report quoted a pharmaco-invasive
# sentence that "refers to STABLE patients". A sentence limited to stable,
# low-risk or mild cases never speaks for a patient in one of these states -
# whatever `unless` the extractor wrote.
_STABLE_ONLY = re.compile(
    r"\b(?:haemodynamically |hemodynamically |clinically )?stable patients?\b|\bin (?:haemodynamically )?stable\b|"
    r"\bpatients? who (?:are|is) (?:haemodynamically |hemodynamically )?stable\b|\blow[- ]risk patients?\b|"
    r"\bmild (?:cases?|disease|episodes?)\b|\buncomplicated\b", re.I)
UNSTABLE = ("shock", "hypotension", "unconscious", "reduced consciousness", "hypoxia")


def contradicts_state(sentence: str, state: str) -> bool:
    return bool(_STABLE_ONLY.search(sentence or "")) and state_hit(UNSTABLE, state) is not None


def applies(item: Item, state: str, age: float | None = None, pregnant: bool = False) -> bool:
    """The item's `when` is met (or empty) and no `unless` state is present."""
    if item.population and not _population_fits(item.population, age, pregnant):
        return False
    if contradicts_state(item.quote, state):
        return False
    if item.when and not state_hit(item.when, state):
        return False
    if item.unless and state_hit(item.unless, state):
        return False
    return True


def present(item: Item, answer: str) -> bool:
    """The report already addresses this item: one of its terms occurs in the
    answer as a whole word (a trailing suffix allowed - "hyperkalaemi" meets
    "hyperkalaemia")."""
    # indications._norm on both sides: one spelling (haem/hem, oedema/edema),
    # punctuation gone.
    hay = " " + " ".join(indications._norm(answer).split()) + " "
    for t in item.terms:
        tn = " ".join(indications._norm(t).split())
        if len(tn) < 2:
            continue
        # A short term is an abbreviation and must stand alone: "ck" is not
        # "ckd", "af" is not "after".
        tail = r"(?![a-z0-9])" if len(tn) <= 3 else ""
        if re.search(rf"(?<![a-z0-9]){re.escape(tn)}{tail}", hay):
            return True
    return False


# ----------------------------------------------------------------- queries
def _dedupe(items: list[Item]) -> list[Item]:
    seen, out = set(), []
    for i in items:
        key = (i.element, frozenset(i.terms))
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def _rank(i: Item) -> tuple:
    # KKM before non-KKM, a condition-specific sentence before a general one.
    return (i.external, not i.core, len(i.quote))


# Run of 2026-10-01 (shocked STEMI, HR 124): the ACS card's bradycardia and VF
# algorithm lines - pacing, magnesium, defibrillation, glucagon "if ...
# overdose" - were listed as missing because the extractor marked them core
# with no `when`. A quote that states its own condition is conditional; a
# quote whose surrounding text names a patient state the patient does not
# have belongs to that state.
_CONDITIONAL = re.compile(
    r"\bif\b|\bwhen\b|\bin (?:patients|those|cases|adults|children) (?:with|who)\b|\bunstable\b|"
    r"\boverdose\b|\bpoisoning\b|\bunless\b", re.I)
# A subgroup named by its abbreviation: "In CKD patients presenting with STEMI".
_SUBGROUP = re.compile(r"\b[Ii]n [A-Z]{2,5}s? patients\b|\bpatients with [A-Z]{2,5}s?\b")
STATE_CUES = (
    "bradycardia", "av block", "heart block", "asystole", "cardiac arrest", "pulseless",
    "ventricular fibrillation", "vf", "ventricular tachycardia", "vt", "torsades", "svt",
    "atrial fibrillation", "broad complex", "wide complex", "narrow complex", "overdose", "poisoning",
    "seizure", "hypoglycaemia", "hyperkalaemia", "pregnan", "anaphylaxis",
)


def foreign_states(text: str, state: str) -> list[str]:
    """The patient-state cues in `text` that this patient's state lacks."""
    low = " " + re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()) + " "
    st = " " + re.sub(r"[^a-z0-9 ]+", " ", (state or "").lower()) + " "
    return [c for c in STATE_CUES if f" {c}" in low and f" {c}" not in st]


_INSTRUCTION = re.compile(
    r"\b(?:give|giving|administer\w*|start|initiate|commence|consider|perform|repeat|monitor\w*|use|check|"
    r"send|obtain|refer|admit|treat|insert|apply|should|must|recommended|undergo|titrate|measured?|assess\w*|"
    r"is the \w+ of choice|first[- ]line)\b", re.I)
_DOSE = re.compile(r"\d\s*(?:mg|mcg|ug|µg|g\b|ml|units?|iu|mmol)", re.I)


def contradicted(item: Item, avoid: list[Item]) -> Item | None:
    """An applicable KKM "avoid" item that names what `item` would give -
    "Initiate dopamine infusion" beside "Dopamine should be avoided ... in
    cardiogenic shock" (2026-10-01). The avoid sentence wins."""
    if item.element == "avoid":
        return None
    for a in avoid:
        if any(t for t in a.terms if len(t) >= 4 and (t in item.terms or present(
                Item("x", "", (t,), (), (), False, "any", "", "", "", "", "", None, "", False),
                f"{item.label} {item.quote}"))):
            return a
    return None


def checklist(cards: list[Card], state: str, age: float | None, pregnant: bool = False,
              limit: int = 12, context=None) -> list[Item]:
    """The core first-hours items (investigation, treatment, monitoring) that
    apply to this patient. `context(item)` returns the text around the item's
    quote in its chunk, for the state-cue test."""
    def unconditional(i: Item) -> bool:
        if not i.when and (_CONDITIONAL.search(f"{i.label} {i.quote}") or _SUBGROUP.search(i.quote)):
            return False
        around = context(i) if context else ""
        return not foreign_states(f"{i.label} {i.quote} {around}", state)

    def instructs(i: Item) -> bool:
        # A heading ("TIMI RISK SCORE FOR UA/NSTEMI") is not an instruction; a
        # list of tests may be one.
        return i.element == "investigation" or bool(_INSTRUCTION.search(i.quote) or _DOSE.search(i.quote))

    avoid = [i for c in cards for i in c.of("avoid") if applies(i, state, age, pregnant)]
    items = [i for c in cards for i in c.items
             if i.element in CHECKLIST_ELEMENTS and i.core and i.setting in ED_SETTINGS
             and applies(i, state, age, pregnant) and unconditional(i) and instructs(i)
             and contradicted(i, avoid) is None]
    items.sort(key=_rank)
    return _dedupe(items)[:limit]


def gaps(cards: list[Card], answer: str, state: str, age: float | None, pregnant: bool = False,
         limit: int = 8, context=None) -> list[Item]:
    return [i for i in checklist(cards, state, age, pregnant, context=context)
            if not present(i, answer)][:limit]


_ICU = re.compile(r"\b(?:ICU|intensive care|HDU|high[- ]dependency|critical care|CCU|coronary care)\b", re.I)
_ADMIT_WORD = re.compile(r"\badmi(?:t|ts|tted|ssion)\b|\bhospitali[sz]", re.I)


def admission(cards: list[Card], state: str, age: float | None, pregnant: bool = False
              ) -> tuple[Item, str, str] | None:
    """(item, target disposition, what met it) for the strongest admission
    criterion this patient meets. A conditional criterion needs its `when` in
    the patient state; an unconditional one must be marked core by the
    extractor AND say "admit" in its own words."""
    best = None
    for c in cards:
        for i in c.of("admission"):
            if not _ADMIT_WORD.search(i.quote) and not _ICU.search(i.quote):
                continue
            if i.population and not _population_fits(i.population, age, pregnant):
                continue
            if i.unless and state_hit(i.unless, state):
                continue
            if i.when:
                hit = state_hit(i.when, state)
                if not hit:
                    continue
            elif i.core and _ADMIT_WORD.search(i.quote):
                hit = f"the diagnosis itself ({c.name})"
            else:
                continue
            target = "ADMIT_ICU_HDU" if _ICU.search(i.quote) else "ADMIT_WARD"
            key = (target == "ADMIT_ICU_HDU", not i.external, bool(i.when))
            if best is None or key > best[0]:
                best = (key, (i, target, hit))
    return best[1] if best else None


def complications(cards: list[Card], state: str, age: float | None, pregnant: bool = False) -> list[Item]:
    # "Contraindications to fibrinolytic therapy" was filed as a complication by
    # the extractor (2026-10-01): a contraindication list is not a danger sign.
    items = [i for c in cards for i in c.of("complication") if applies(i, state, age, pregnant)
             and not re.search(r"contraindicat", f"{i.label} {i.quote}", re.I)]
    items.sort(key=_rank)
    return _dedupe(items)


def cautions(cards: list[Card], state: str, age: float | None, pregnant: bool = False) -> list[Item]:
    # First-hours cautions only: "a maintenance aspirin dose of 300-325 mg
    # daily ..." is outpatient advice (2026-10-01).
    items = [i for c in cards for i in c.of("avoid") if applies(i, state, age, pregnant)
             and i.setting in ED_SETTINGS and not foreign_states(f"{i.label} {i.quote}", state)]
    items.sort(key=_rank)
    return _dedupe(items)


# A dose sentence has a PURPOSE. On 2026-10-01 a 1 mg/kg treatment dose of
# enoxaparin was called "the general range" beside "enoxaparin 40mg OD ...
# until the patient is ambulant" - a prophylaxis dose.
_PURPOSE = (
    ("prophylaxis", re.compile(r"prophyla\w*|prevent\w* of (?:dvt|vte|thrombo)|until (?:the patient is )?ambulant", re.I)),
    ("maintenance", re.compile(r"\bmaintenance\b|long[- ]term|daily thereafter", re.I)),
)
_TREATMENT_CUE = re.compile(r"\bloading\b|\bstat\b|\bbolus\b|\binitial\w*|\btreatment of\b|\bacute\b", re.I)


def purpose_mismatch(sentence: str, stated: str) -> bool:
    """The sentence states a dose for a purpose (prophylaxis, maintenance)
    the recommendation does not mention - and names no treatment dose."""
    if _TREATMENT_CUE.search(sentence or ""):
        return False
    return any(rx.search(sentence or "") and not rx.search(stated or "") for _, rx in _PURPOSE)


def has_dose(cards: list[Card], drug_terms: set[str], stated: str = "") -> Item | None:
    """A treatment item that names this drug and carries a dose for the same
    purpose as the recommendation."""
    for c in cards:
        for i in c.of("treatment"):
            if (any(t in drug_terms for t in i.terms) and _DOSE.search(i.quote)
                    and not purpose_mismatch(i.quote, stated)):
                return i
    return None


# ---------------------------------------------------------------- coverage
_REQUIRED = {
    "emergency": ("red_flag", "investigation", "treatment", "admission", "monitoring"),
    "urgent": ("red_flag", "investigation", "treatment", "admission"),
    "routine": ("investigation", "treatment", "referral"),
    "chronic": ("treatment", "referral"),
}


def silent_elements(cards: list[Card]) -> list[str]:
    """Elements this condition needs an answer for that no KKM sentence gives:
    the report states them as gaps instead of letting the model fill them."""
    if not cards:
        return []
    need = _REQUIRED.get(cards[0].acuity, _REQUIRED["urgent"])
    have = {i.element for c in cards for i in c.items if not i.external}
    return [el for el in need if el not in have]
