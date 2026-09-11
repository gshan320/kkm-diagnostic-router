"""Regression tests for the ETD redirection guard added 2026-09-11.

Source: JKN Selangor, "Emergency Medicine and Trauma Services: Redirection
Policy", July 2024, section 4.2 - the sixteen patient types that must be SEEN
IN THE ETD and may not be redirected to a klinik kesihatan or an outpatient
clinic.

Three properties matter more than any individual clause:

    COMPLETE      all sixteen clauses are accounted for - fourteen as rules and
                  two named as uncheckable. A clause that quietly vanishes is
                  worse than one that is openly unimplemented.
    NO CRY-WOLF   the ordinary redirectable patient - the URTI, the routine
                  review, the chronic itch that section 4.1 explicitly sends to
                  a klinik kesihatan - must produce NO warning. This guard
                  fires on a majority of ED attendances if it is careless, and
                  a warning that always fires is read by nobody.
    ADVISORY      it never changes the disposition. The policy is state-level;
                  MTS 2022 has no redirection criteria at all.

Run from the backend directory:

    ../.venv/bin/python -m tests.test_redirection
"""
import sys

from app import redirection
from app import rag_engine as R
from app.schemas import (
    ArrivalMode, Comorbidity, DiagnosticSchema, Disposition,
    PregnancyStatus, TriageRequest,
)

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def req(age=40, complaint="unwell", **kw):
    return TriageRequest(age=age, complaint=complaint, **kw)


def clauses(**kw):
    return sorted(e.clause for e in redirection.matches(req(**kw)))


def report(disposition=Disposition.DISCHARGE_WITH_FOLLOW_UP, **kw):
    d = DiagnosticSchema(disposition=disposition)
    R._check_redirection(d, req(**kw))
    return d


# ===========================================================================
print("1. All sixteen clauses of section 4.2 are accounted for.")
# ===========================================================================
covered = {e.clause for e in redirection.EXCLUSIONS} | {c for c, _ in redirection.UNCHECKABLE}
expected = {f"4.2.{n}" for n in range(1, 17)}
check("every clause is either a rule or openly uncheckable",
      covered == expected, str(sorted(expected - covered)))
check("fourteen are implemented", len(redirection.EXCLUSIONS) == 14,
      str(len(redirection.EXCLUSIONS)))
check("two are named as uncheckable, not dropped",
      {c for c, _ in redirection.UNCHECKABLE} == {"4.2.4", "4.2.12"})
check("no clause is both implemented and uncheckable",
      not ({e.clause for e in redirection.EXCLUSIONS}
           & {c for c, _ in redirection.UNCHECKABLE}))

# ===========================================================================
print("\n2. Each implemented clause fires on the patient it names.")
# ===========================================================================
for name, kw, clause in [
    ("wheelchair dependent",  dict(complaint="Wheelchair bound, needs catheter change"), "4.2.1"),
    ("child under 5",         dict(age=3, complaint="Mild rash"), "4.2.2"),
    ("child with comorbid",   dict(age=8, complaint="Mild cough",
                                   comorbidities=[Comorbidity.AIRWAY_DISEASE]), "4.2.3"),
    ("over 60",               dict(age=72, complaint="Routine review"), "4.2.5"),
    ("chest pain",            dict(complaint="Central chest pain 2 hours"), "4.2.6"),
    ("acute abdominal pain",  dict(complaint="Abdominal pain since last night"), "4.2.7"),
    ("OSCC",                  dict(complaint="Assessment", arrival_mode=ArrivalMode.OSCC), "4.2.8"),
    ("asthma exacerbation",   dict(complaint="Mild wheeze, known asthma"), "4.2.9"),
    ("urinary retention",     dict(complaint="Unable to pass urine since morning"), "4.2.10"),
    ("fever with warning",    dict(complaint="Fever 3 days with vomiting and lethargy"), "4.2.11"),
    ("referral letter",       dict(complaint="Sent by clinic",
                                   arrival_mode=ArrivalMode.REFERRED_FROM_CLINIC), "4.2.13"),
    ("police / medicolegal",  dict(complaint="Assessment",
                                   arrival_mode=ArrivalMode.POLICE_OKT), "4.2.14"),
    ("trauma",                dict(complaint="Twisted ankle playing football"), "4.2.15"),
    ("pregnancy",             dict(age=25, complaint="Nausea",
                                   pregnancy=PregnancyStatus.PREGNANT), "4.2.16"),
]:
    check(f"  {clause}  {name}", clause in clauses(**kw), str(clauses(**kw)))

# ===========================================================================
print("\n3. NO CRY-WOLF. The patients section 4.1 sends to a klinik kesihatan")
print("   must produce no warning at all - they are the point of the policy.")
# ===========================================================================
# Every one of these is named in section 4.1 Step 3 as redirectable.
for name, kw in [
    ("URTI / sore throat, no resp symptoms", dict(age=28, complaint="Sore throat 3 days, no respiratory symptoms")),
    ("routine wound dressing",   dict(age=35, complaint="Routine wound dressing change, clean and healing")),
    ("chronic simple itch",      dict(age=33, complaint="Chronic simple itch on forearm for a month")),
    ("lumps and bumps",          dict(age=45, complaint="Painless lump on back, present 6 months")),
    ("pink eye",                 dict(age=30, complaint="Pink eye, no loss of vision")),
    ("diarrhoea no dehydration", dict(age=29, complaint="Diarrhoea 2 days, no dehydration, tolerating fluids")),
    ("medication continuation",  dict(age=40, complaint="Continuation of regular medication supply")),
    ("second opinion",           dict(age=38, complaint="Requesting second opinion, well")),
    ("medical check-up",         dict(age=32, complaint="Routine medical check-up")),
]:
    check(f"  clear: {name}", clauses(**kw) == [], str(clauses(**kw)))

# ===========================================================================
print("\n4. Negation. A complaint field records what the patient does NOT have")
print("   as often as what they do.")
# ===========================================================================
check("'no chest pain' does not fire the chest-pain clause",
      clauses(complaint="Heartburn after meals, no chest pain") == [],
      str(clauses(complaint="Heartburn after meals, no chest pain")))
check("'denies trauma or fall' does not fire the trauma clause",
      clauses(complaint="Headache, denies trauma or fall") == [])
check("but a negation of ONE finding does not suppress the next",
      "4.2.6" in clauses(complaint="No fever, chest pain since morning"))
check("'eye strain' is not a trauma mechanism",
      clauses(age=35, complaint="Eye strain from computer work") == [])
check("a real muscle strain is",
      "4.2.15" in clauses(age=30, complaint="Back strain after lifting"))
check("fever ALONE is not an exclusion - only fever with warning signs",
      clauses(age=50, complaint="Fever 1 day, otherwise well") == [])

# Clause 4.2.16 matches on `\bpregnan`, the substring recorded as a trap in the
# triage-modifiers work. The lean here is the OPPOSITE of the teratogen check's
# and deliberately so: 4.2.16 is "ALL pregnancy related cases", so a patient who
# MIGHT be pregnant is exactly who it protects. Only an explicit exclusion
# clears it.
check("'pregnancy excluded' clears clause 4.2.16",
      clauses(complaint="Abdominal cramps, pregnancy excluded on UPT") == [],
      str(clauses(complaint="Abdominal cramps, pregnancy excluded on UPT")))
check("'not pregnant' clears it too",
      clauses(complaint="Nausea, patient is not pregnant") == [])
check("'suspected ectopic pregnancy' DOES flag - that is who the clause is for",
      "4.2.16" in clauses(complaint="Lower abdominal pain, suspected ectopic pregnancy"))
check("'pregnancy status not established' DOES flag",
      "4.2.16" in clauses(complaint="Vomiting, pregnancy status not established"))
check("'chest pain resolved' still flags - the patient HAD chest pain",
      "4.2.6" in clauses(complaint="Chest pain resolved after 10 minutes at rest"))
check("'ACS ruled out' does not clear the chest pain itself",
      "4.2.6" in clauses(complaint="Chest pain, ACS ruled out by serial troponin"))

# ===========================================================================
print("\n5. It fires only on a disposition that sends the patient OUT,")
print("   and it never changes that disposition.")
# ===========================================================================
kw = dict(age=70, complaint="Central chest pain 2 hours")
for disp, should_warn in [
    (Disposition.DISCHARGE_WITH_FOLLOW_UP, True),
    (Disposition.REFER_SPECIALIST, True),
    (Disposition.ADMIT_WARD, False),
    (Disposition.ED_OBSERVATION, False),
    (Disposition.ADMIT_ICU_HDU, False),
    (Disposition.RESUSCITATION_BAY, False),
]:
    d = report(disposition=disp, **kw)
    check(f"  {disp.value:<24} warns={should_warn}",
          bool(d.redirection_warning) is should_warn)
    check(f"    and leaves the disposition at {disp.value}",
          d.disposition == disp)

# ===========================================================================
print("\n6. The warning is checkable: it names its clauses and its limits.")
# ===========================================================================
d = report(age=70, complaint="Central chest pain 2 hours")
w = d.redirection_warning
check("names every clause it matched", "4.2.5" in w and "4.2.6" in w)
check("names the policy it comes from", redirection.POLICY in w)
check("says the policy is state-level, not national", "Selangor" in w)
check("names the clauses it could NOT check", "4.2.4" in w and "4.2.12" in w)
check("a clear patient gets no warning text at all",
      report(age=28, complaint="Sore throat 3 days").redirection_warning == "")

# ===========================================================================
print("\n6b. The composite path: the guardrails that SET the disposition, then")
print("    this check reading what they settled on.")
# ===========================================================================
# The realistic route to a warning is not a model that proposes discharge on a
# sick patient - `_enforce_disposition` would raise that. It is a genuinely
# low-acuity patient whom `_apply_disposition_ceiling` CAPS at discharge, who
# nonetheless appears in section 4.2. Age is the commonest reason.
from app.schemas import Vitals, TriageLevel


def composite(age, complaint, vitals):
    d = DiagnosticSchema(mts_triage_level=TriageLevel.ROUTINE,
                         disposition=Disposition.DISCHARGE_WITH_FOLLOW_UP)
    r = TriageRequest(age=age, complaint=complaint, vitals=vitals)
    R._enforce_mts_level(d, r)
    R._enforce_disposition(d, r)
    R._check_redirection(d, r)
    R._ground_triage_timing(d)
    return d


knee = "Chronic knee ache for 6 months, wants review"
mild = Vitals(systolic_bp=124, diastolic_bp=76, heart_rate=74, spo2=98, pain_score=2)

old_age = composite(72, knee, mild)
young = composite(34, knee, mild)
child = composite(3, "Mild rash on arm for 2 days, feeding well",
                  Vitals(heart_rate=110, spo2=99, pain_score=0))

check("a 72-year-old capped at discharge is flagged on 4.2.5",
      "4.2.5" in old_age.redirection_warning)
check("a 34-year-old with the IDENTICAL complaint is not flagged",
      young.redirection_warning == "",
      young.redirection_warning[:80])
check("  so the warning discriminates on the clause, not on the disposition",
      old_age.disposition == young.disposition)
check("a 3-year-old is flagged on 4.2.2", "4.2.2" in child.redirection_warning)
check("the pain 1-3 floor still pulls all three off Level 5",
      all(int(d.mts_triage_level) == 4 for d in (old_age, young, child)),
      str([int(d.mts_triage_level) for d in (old_age, young, child)]))
check("and the time comes from that final level, not the model's",
      all(d.time_to_treatment == "under 60 minutes"
          for d in (old_age, young, child)))

# ===========================================================================
print("\n7. A malformed or minimal intake must not break the report.")
# ===========================================================================
check("a bare intake matches nothing", clauses(age=30, complaint="unwell") == [])


class _Broken:
    age = "not a number"
    complaint = None


check("a broken intake returns rather than raising",
      isinstance(redirection.matches(_Broken()), list))

print("\n" + "=" * 70)
if fails:
    print(f"{len(fails)} FAILED: " + "; ".join(fails))
    sys.exit(1)
print("All checks passed.")
