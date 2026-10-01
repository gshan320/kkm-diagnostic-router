"""Direct name lookup over the FUKKM formulary.

Why this replaces embedding search for the formulary
----------------------------------------------------
FUKKM is a structured database of ~1,700 records, not prose. Every entry is
formatted alike, so the drug name - the only discriminating signal - is a small
fraction of each chunk. Searching it by embedding a treatment sentence returned,
for a shocked STEMI: a pneumococcal vaccine, secukinumab, rituximab and
fluconazole. Clopidogrel ranked 16th; acetylsalicylic acid did not appear in the
top 25.

Meanwhile a plain name lookup over the whole file finds every one of them
immediately. The formulary should be queried like the database it is.

The second failure was vocabulary. FUKKM does not use the words clinicians and
CPGs use:

    model / CPG        FUKKM record
    Aspirin        ->  Acetylsalicylic Acid 100 mg & Glycine 45 mg Tablet
    Nitroglycerin  ->  Glyceryl Trinitrate 0.5mg Sublingual Tablet
    Adrenaline     ->  Epinephrine ...

so both were reported "CATEGORY UNVERIFIED" while sitting in the file.

Matching is scored, not first-hit. A naive substring match for "atropine"
returns "Atropine Sulphate 0.3%, Cocaine HCl 1.7%, Adrenaline ... Mydriatic" -
an eye drop - and would have attached an ophthalmic preparation's category to
an IV resuscitation drug. Exact and whole-word matches outrank substrings, and
single-ingredient products outrank combinations.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass

from . import config

# Strength, pack and dosage-form noise that is not part of the drug's identity.
_STRENGTH = re.compile(
    r"\d+[\d.,/]*\s*(?:mg|mcg|µg|g|ml|l|iu|units?|%|meq|mmol|million|billion)\b",
    re.IGNORECASE,
)
_FORM = re.compile(
    r"\b(?:tablet|tablets|capsule|capsules|injection|injectable|solution|suspension|"
    r"syrup|elixir|infusion|sublingual|oral|vial|ampoule|ampule|pre-?filled|pen|"
    r"sachet|cream|ointment|gel|inhaler|inhalation|nebuliser|nebulizer|drops?|"
    r"suppository|patch|powder|granules|effervescent|sustained release|modified release|"
    r"prolonged release|film[- ]coated|enteric[- ]coated|dispersible|chewable|"
    r"for|in|with|and|adsorbed|sterile|concentrate)\b",
    re.IGNORECASE,
)

# Clinical name -> the term FUKKM actually files it under (and the reverse).
# Curated and auditable on purpose: a fuzzy matcher here would silently attach
# the wrong category to a real prescription.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "aspirin": ("acetylsalicylic acid", "acetylsalicylic"),
    "acetylsalicylic acid": ("aspirin",),
    "nitroglycerin": ("glyceryl trinitrate",),
    "glyceryl trinitrate": ("nitroglycerin", "gtn"),
    "gtn": ("glyceryl trinitrate",),
    "adrenaline": ("epinephrine",),
    "epinephrine": ("adrenaline",),
    "noradrenaline": ("norepinephrine",),
    "norepinephrine": ("noradrenaline",),
    "paracetamol": ("acetaminophen",),
    "acetaminophen": ("paracetamol",),
    "frusemide": ("furosemide",),
    "furosemide": ("frusemide",),
    "salbutamol": ("albuterol",),
    "albuterol": ("salbutamol",),
    "normal saline": ("sodium chloride",),
    "crystalloid": ("sodium chloride",),
    "hartmann": ("sodium lactate", "compound sodium lactate"),
    "vitamin k": ("phytomenadione",),
    "phytomenadione": ("vitamin k",),
    "pethidine": ("meperidine",),
    "lignocaine": ("lidocaine",),
    "lidocaine": ("lignocaine",),
    "thiamine": ("vitamin b1",),
    "isosorbide dinitrate": ("isdn",),
    "soluble insulin": ("insulin human", "regular insulin"),
}


# Route words a clinician writes -> dosage-form words FUKKM files under.
# Prescriber category VARIES BY FORMULATION (glyceryl trinitrate is C
# sublingual, "A, A/KK" as an injection, B as an aerosol), so the route is not
# cosmetic - it selects which category actually applies.
ROUTE_FORMS: dict[str, tuple[str, ...]] = {
    "intravenous": ("injection", "infusion", "iv"),
    "iv": ("injection", "infusion"),
    "intramuscular": ("injection",),
    "im": ("injection",),
    "subcutaneous": ("injection",),
    "sublingual": ("sublingual",),
    # Order is preference, not just membership: an adult "oral" order should
    # resolve to a tablet, not the paediatric syrup that also matches.
    "oral": ("tablet", "capsule", "dispersible", "soluble", "effervescent",
             "chewable", "granules", "suspension", "syrup", "solution"),
    "po": ("tablet", "capsule", "dispersible", "syrup"),
    "rectal": ("suppository", "enema"),
    "nebulised": ("respirator", "nebuli", "inhalation"),
    "nebulized": ("respirator", "nebuli", "inhalation"),
    "inhaled": ("inhaler", "aerosol", "inhalation", "metered"),
    "inhalation": ("inhaler", "aerosol", "inhalation", "metered"),
    "topical": ("cream", "ointment", "gel", "lotion"),
    "ophthalmic": ("eye",),
    "eye": ("eye",),
    "transdermal": ("patch",),
}

# Clinical shorthand that implies a specific strength FUKKM spells out.
STRENGTH_HINTS: dict[str, str] = {
    "normal saline": "0.9",
    "0.9% saline": "0.9",
    "half normal saline": "0.45",
    "dextrose 5": "5%",
}


@dataclass(frozen=True)
class Entry:
    fukkm_no: str
    drug_name: str
    normalised: str
    prescriber_category: str
    indication: str
    dosage: str
    mdc_code: str

    @property
    def citation(self) -> str:
        """A FUKKM listing number is a better provenance handle than an [S#]
        into a retrieved chunk: it identifies the row in the published book."""
        return f"FUKKM {self.fukkm_no}" if self.fukkm_no else "FUKKM (no listing number)"


def normalise(name: str) -> str:
    n = (name or "").lower()
    n = _STRENGTH.sub(" ", n)
    n = _FORM.sub(" ", n)
    n = re.sub(r"[^a-z0-9+/ -]", " ", n)
    return re.sub(r"\s{2,}", " ", n).strip(" ,&-/")


_entries: list[Entry] = []
_loaded = False
_lock = threading.Lock()


def _load() -> list[Entry]:
    global _loaded
    with _lock:
        if _loaded:
            return _entries
        _loaded = True
        try:
            raw = json.loads(config.FUKKM_JSON.read_text(encoding="utf-8"))
        except Exception:
            return _entries
        records = raw if isinstance(raw, list) else raw.get("drugs", [])
        for r in records:
            name = str(r.get("drug_name", "") or "").strip()
            if not name or name.isdigit():
                continue
            _entries.append(Entry(
                fukkm_no=str(r.get("fukkm_no", "") or "").strip(),
                drug_name=name,
                normalised=normalise(name),
                prescriber_category=str(r.get("prescriber_category", "") or "").strip(),
                indication=str(r.get("indication", "") or "").strip(),
                dosage=str(r.get("dosage", "") or "").strip(),
                mdc_code=str(r.get("mdc_code", "") or "").strip(),
            ))
        return _entries


def size() -> int:
    return len(_load())


def _candidates(term: str) -> list[str]:
    t = normalise(term)
    out = [t] if t else []
    for key, syns in SYNONYMS.items():
        if key == t or key in t:
            out.extend(syns)
    return list(dict.fromkeys(out))


def _score(candidate: str, entry: Entry) -> int:
    """Higher is better. Exact and whole-word beat substring; single-ingredient
    products beat combinations, so "atropine" does not resolve to a mydriatic
    eye drop that happens to contain atropine."""
    name = entry.normalised
    if not candidate or not name:
        return 0
    if name == candidate:
        base = 100
    elif name.startswith(candidate + " ") or name.startswith(candidate + "/"):
        base = 85
    elif re.search(rf"(?<![a-z]){re.escape(candidate)}(?![a-z])", name):
        base = 60
    elif candidate in name:
        base = 30
    else:
        return 0
    # A combination product names several ingredients; prefer the plain one.
    ingredients = 1 + name.count(",") + name.count("+") + name.count("&")
    return base - 6 * (ingredients - 1) - min(len(name) // 25, 4)


@dataclass(frozen=True)
class Match:
    """All formulations of one ingredient, plus the route-appropriate pick.

    `categories` is the honest answer when a drug is stocked in several forms
    with different prescriber categories - reporting a single one would attach
    an IV product's category to an oral prescription, or vice versa.
    """
    query: str
    best: Entry
    entries: tuple[Entry, ...]
    route_matched: bool

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({e.prescriber_category for e in self.entries if e.prescriber_category}))

    @property
    def category_display(self) -> str:
        cats = self.categories
        if not cats:
            return ""
        if len(cats) == 1:
            return cats[0]
        if self.route_matched:
            return self.best.prescriber_category
        return " or ".join(cats)

    @property
    def varies(self) -> bool:
        return len(self.categories) > 1 and not self.route_matched

    @property
    def citation(self) -> str:
        return self.best.citation


def _route_bonus(route: str, entry: Entry) -> int:
    """Reward the route, and prefer the earlier form in the route's list.

    Membership alone is not enough: "oral" matches both a tablet and a syrup, and
    an adult oral order resolving to a paediatric syrup is the wrong product even
    when the prescriber category happens to agree."""
    forms = ROUTE_FORMS.get((route or "").strip().lower(), ())
    if not forms:
        return 0
    low = entry.drug_name.lower()
    for i, form in enumerate(forms):
        if form in low:
            return 25 - 2 * i
    return -15


_DOSE_STRENGTH = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mcg|microgram|mg|g|ml|units?|iu)\b", re.IGNORECASE)


def _strength_bonus(dose: str, entry: Entry) -> int:
    """Prefer the formulation whose strength the prescription actually names.

    Measured: "aspirin 300mg" resolved to the 150mg Dispersible Tablet because
    the dose played no part in matching - and FUKKM also stocks a 300mg Soluble
    Tablet, which is the correct ACS loading dose. A dose check built on the
    wrong formulation flags a correct prescription, which is the fastest way to
    train a clinician to ignore dose warnings."""
    if not dose:
        return 0
    name = entry.drug_name.lower()
    for m in _DOSE_STRENGTH.finditer(dose):
        num, unit = m.group(1).replace(",", "."), m.group(2).lower()
        if "." in num:
            num = num.rstrip("0").rstrip(".")
        if re.search(rf"(?<!\d){re.escape(num)}\s*{unit}\b", name):
            return 30
    return 0


def lookup(drug_name: str, route: str = "", dose: str = "", min_score: int = 45) -> Match | None:
    """Best FUKKM record for a clinical drug name, plus every sibling formulation.

    `min_score` deliberately excludes weak substring hits: an unverified category
    is safe, a confidently wrong one is not."""
    entries = _load()
    if not entries or not (drug_name or "").strip():
        return None

    strength = ""
    low = normalise(drug_name)
    for hint, want in STRENGTH_HINTS.items():
        if hint in low or hint in (drug_name or "").lower():
            strength = want
            break

    scored: list[tuple[int, Entry]] = []
    for cand in _candidates(drug_name):
        for entry in entries:
            base = _score(cand, entry)
            if not base:
                continue
            total = base + _route_bonus(route, entry) + _strength_bonus(dose, entry)
            if strength and strength in entry.drug_name.lower():
                total += 20
            scored.append((total, entry))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    if scored[0][0] < min_score:
        return None
    best = scored[0][1]

    # Siblings = every record sharing the winner's normalised ingredient name.
    siblings = tuple(e for e in entries if e.normalised == best.normalised)
    if not siblings:
        siblings = (best,)
    matched = bool(route) and _route_bonus(route, best) > 0
    return Match(query=drug_name, best=best, entries=siblings, route_matched=matched)


# ---------------------------------------------------------------- by-name index
# Attention layer A1 (2026-09-30). The formulary rows put in front of the model
# used to be found by EMBEDDING treatment sentences from the clinical chunks -
# which, for a rhabdomyolysis patient whose context held a snakebite page,
# returned pit-viper and sea-snake antivenom, Factor IX and fondaparinux. A
# formulary is a table: the rows worth showing are the drugs the kept guideline
# text actually NAMES, found by name.

# A generic name is the first word of the normalised product name, or the first
# two when the first alone is a salt or a class word ("sodium chloride").
_TWO_WORD_HEADS = frozenset("""sodium potassium calcium magnesium glyceryl tranexamic folic
mefenamic valproic ferrous human vitamin hydrogen amino isosorbide compound insulin
factor""".split())
# Single words that name lab analytes or everyday substances far more often than
# a prescription - searching for them would pull a row into every context.
_NOT_A_MENTION = frozenset("""sodium potassium calcium magnesium glucose water oxygen iron zinc
phosphate chloride human vitamin normal compound multiple combined sterile
factor protein continuous essential active total standard simple special general balanced
plain natural liquid soft white yellow black green super extra fresh concentrated purified
activated modified recombinant ready single double triple""".split())


def _head(normalised: str) -> str:
    words = normalised.split()
    if not words:
        return ""
    # The second word must be a word: "salbutamol 0 5" (0.5%) is not "salbutamol 0".
    if words[0] in _TWO_WORD_HEADS and len(words) > 1 and words[1].isalpha():
        return f"{words[0]} {words[1]}"
    return words[0]


# Class abbreviation -> the FUKKM generic it means at the bedside. Kept to
# abbreviations with ONE first-line agent; "ICS" or "ACEI" name many drugs.
CLASS_SHORTHAND: dict[str, str] = {
    "saba": "salbutamol",
    # Not classes, but the same kind of bedside shorthand: the HDP manual writes
    # "IV 4g MgSO4", the PPH guide "Syntocinon", and no MgSO4 row was offered
    # for eclampsia (2026-09-30).
    "mgso4": "magnesium sulphate",
    "syntocinon": "oxytocin",
    "sama": "ipratropium",
    "saac": "ipratropium",
    "ocs": "prednisolone",
}

_index: dict[str, list[Entry]] | None = None
_index_rx: re.Pattern | None = None


def _build_index() -> tuple[dict[str, list[Entry]], re.Pattern]:
    global _index, _index_rx
    if _index is None:
        idx: dict[str, list[Entry]] = {}
        for e in _load():
            # Multi-ingredient product lines ("Continuous Ambulatory Peritoneal
            # Dialysis Solution containing ...") are named by what they are
            # for, not by a drug - their first word is ordinary English.
            if len(e.normalised.split()) > 4:
                continue
            h = _head(e.normalised)
            # The stoplist guards single words only: "sodium" alone is a lab
            # value, "sodium chloride" is the fluid.
            if len(h) >= 5 and (" " in h or h not in _NOT_A_MENTION):
                idx.setdefault(h, []).append(e)
        # Clinical synonyms resolve to the head FUKKM files the drug under.
        for key, syns in SYNONYMS.items():
            for sy in syns:
                if sy in idx and key not in idx and len(key) >= 3:
                    idx[key] = idx[sy]
        # Guideline shorthand for the one drug the class stands for in an acute
        # setting. The Asthma CPG 2024 acute pages say "SABA", never
        # "salbutamol", so the reliever was never offered (2026-09-30).
        for abbr, generic in CLASS_SHORTHAND.items():
            if generic in idx and abbr not in idx:
                idx[abbr] = idx[generic]
        _index = idx
        alt = "|".join(re.escape(k) for k in sorted(idx, key=len, reverse=True))
        _index_rx = re.compile(rf"(?<![a-z])(?:{alt})(?![a-z])", re.IGNORECASE)
    return _index, _index_rx


def mentioned(text: str) -> list[str]:
    """Generic names this text mentions, in order of first mention."""
    idx, rx = _build_index()
    seen: dict[str, None] = {}
    for m in rx.finditer(text or ""):
        seen.setdefault(m.group(0).lower(), None)
    return [k for k in seen if k in idx]


def entries_for(generic: str) -> list[Entry]:
    idx, _ = _build_index()
    return idx.get(generic.lower(), [])
