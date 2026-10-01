"""Which patients a source document is written for.

Why this exists
---------------
Measured 2026-09-30: a 55-year-old's rhabdomyolysis report named the Paediatric
Protocols as its governing guideline and cited page 700 - the SEA-SNAKE
envenomation section - for its red flags. Nothing stopped a children's source
reaching an adult, because CLINICAL_DOC_TYPES treats every guideline alike.

A document's population is read from its type and its title:

  paediatric   the Paediatric Protocols, and any guideline whose title is for
               children / neonates / infants ("Dengue Fever in Children",
               "Type 1 Diabetes Mellitus in Children and Adolescents").
  adult        a title that says "in Adults" and names no younger group.
               "Rhinosinusitis in Adolescents AND Adults" is for both.
  all          everything else - MTS 2022 (which has its own paediatric pages),
               FUKKM, and guidelines that do not restrict their population.

And applied by the patient's age:

  under 12     no adult-only document (MOH paediatric protocols stop at 12)
  12 to 17     both - an adolescent may need either
  18 and over  no paediatric-only document
"""

from __future__ import annotations

import re

from . import config

PAEDIATRIC = "paediatric"
ADULT = "adult"
ALL = "all"

# \w* matters: "National Antimicrobial Guideline 2024 (Paediatrics)" reached a
# 45-year-old on 2026-09-30 because "paediatric\b" does not match "Paediatrics".
_CHILD = re.compile(r"\b(child\w*|paediatric\w*|pediatric\w*|neonat\w*|infant\w*|newborn\w*)\b", re.I)
_ADOLESCENT = re.compile(r"\badolescen\w*", re.I)
_ADULT = re.compile(r"\badults?\b", re.I)

CHILD_MAX_AGE = 12.0
ADULT_MIN_AGE = 18.0


def of(meta: dict) -> str:
    if meta.get("doc_type") == config.DOC_TYPE_PAEDS:
        return PAEDIATRIC
    title = f"{meta.get('cpg_title', '')} {meta.get('filename', '')}"
    if _CHILD.search(title) and not _ADULT.search(title):
        return PAEDIATRIC
    if _ADULT.search(title) and not (_CHILD.search(title) or _ADOLESCENT.search(title)):
        return ADULT
    return ALL


def title_population(title: str) -> str:
    return of({"cpg_title": title})


def allowed(meta: dict, age: float) -> bool:
    pop = of(meta)
    if pop == PAEDIATRIC:
        return age < ADULT_MIN_AGE
    if pop == ADULT:
        return age >= CHILD_MAX_AGE
    return True


def label(age: float) -> str:
    if age < CHILD_MAX_AGE:
        return "a child"
    if age < ADULT_MIN_AGE:
        return "an adolescent"
    return "an adult"
