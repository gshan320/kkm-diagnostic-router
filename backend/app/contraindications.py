"""Drug-condition and drug-drug contraindication checks.

Why this exists
---------------
On a shocked inferior STEMI with V4R elevation and sildenafil taken 12 hours
earlier, the model recommended sublingual GTN. Both contraindications - the
PDE5 inhibitor and the RV infarct - were stated in the intake AND present in
the retrieved guideline text. Retrieval worked; nothing read it.

The recommendation also arrived as an *immediate action*, not a drug entry, so
any check that only inspects `drug_recommendations` would have missed it. This
module scans actions, investigations and drugs alike.

Design
------
Rules are readable, auditable statements with a written clinical reason, in the
same spirit as red_flags.py and _enforce_mts_level: clinical safety is enforced
in code, not requested in a prompt. Prompts were measured not to hold.

Nothing is ever deleted from the model's output. A finding annotates it, so a
clinician sees both the recommendation and the objection and decides. Silently
removing a recommendation would hide a model failure rather than expose it.

Severity
--------
ABSOLUTE - giving this drug to this patient is expected to cause harm.
CAUTION  - defensible in some circumstances, but must be a deliberate decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    name: str
    drugs: tuple[str, ...]      # matched against the recommendation text
    when: tuple[str, ...]       # matched against the clinical context; ANY hit fires
    severity: str               # "ABSOLUTE" | "CAUTION"
    reason: str
    unless: tuple[str, ...] = ()  # context that legitimately permits it


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    where: str        # "action 2" / "drug: Aspirin"
    item: str
    trigger: str      # the phrase in the intake that fired it
    reason: str


# ---------------------------------------------------------------------------
# Drug-class vocabularies. Generic names as used in the FUKKM and the CPGs,
# plus the abbreviations clinicians actually write.
# ---------------------------------------------------------------------------
NITRATES = (r"\bGTN\b", r"glyceryl\s+trinitrate", r"nitroglycerin", r"isosorbide",
            r"\bnitrate", r"\bISDN\b", r"\bISMN\b")
BETA_BLOCKERS = (r"\bmetoprolol", r"\bbisoprolol", r"\batenolol", r"\bcarvedilol",
                 r"\bpropranolol", r"\blabetalol", r"\bnebivolol", r"\bsotalol",
                 r"beta[- ]?blocker", r"\bB[- ]?blocker\b")
ANTIPLATELETS = (r"\baspirin", r"acetylsalicylic", r"\bclopidogrel", r"\bticagrelor",
                 r"\bprasugrel", r"\bdipyridamole")
ANTICOAGULANTS = (r"\bwarfarin", r"\bheparin", r"\benoxaparin", r"\bdabigatran",
                  r"\brivaroxaban", r"\bapixaban", r"\bedoxaban", r"\bfondaparinux",
                  r"\bDOAC\b", r"\bNOAC\b", r"\bLMWH\b")
# The oral subset. Parenteral anticoagulation (heparin, enoxaparin,
# fondaparinux) is standard ACS care alongside dual antiplatelets; only an
# ORAL anticoagulant makes "triple therapy". Found 2026-09-30: the triple-
# therapy caution fired on heparin in a shocked STEMI.
ORAL_ANTICOAGULANTS = (r"\bwarfarin", r"\bdabigatran", r"\brivaroxaban", r"\bapixaban", r"\bedoxaban",
                       r"\bDOAC\b", r"\bNOAC\b", r"oral anticoagula")
THROMBOLYTICS = (r"\balteplase", r"\bstreptokinase", r"\btenecteplase", r"\breteplase",
                 r"thrombolysi", r"fibrinolysi", r"fibrinolytic")
NSAIDS = (r"\bibuprofen", r"\bdiclofenac", r"mefenamic", r"\bnaproxen", r"\bketorolac",
          r"\bindomethacin", r"\bcelecoxib", r"\bNSAID", r"non[- ]steroidal anti[- ]?inflammat")
ACE_ARB = (r"\benalapril", r"\bcaptopril", r"\blisinopril", r"\bperindopril", r"\bramipril",
           r"\blosartan", r"\bvalsartan", r"\btelmisartan", r"\birbesartan",
           r"\bACE[- ]?inhibitor", r"\bARB\b")
STATINS = (r"\bsimvastatin", r"\batorvastatin", r"\brosuvastatin", r"\blovastatin", r"\bstatin\b")
TETRACYCLINES = (r"\bdoxycycline", r"\btetracycline", r"\bminocycline")
QUINOLONES = (r"\bciprofloxacin", r"\blevofloxacin", r"\bmoxifloxacin", r"\bofloxacin")
SEDATIVES = (r"\bmorphine", r"\bmidazolam", r"\bdiazepam", r"\bfentanyl", r"\bpethidine",
             r"\blorazepam", r"\bpropofol", r"\bopioid", r"\bsedati")
RATE_LIMITING_CCB = (r"\bverapamil", r"\bdiltiazem")


RULES: tuple[Rule, ...] = (
    # ---------------------------------------------------------- cardiovascular
    Rule("nitrate_with_pde5_inhibitor", NITRATES,
         (r"\bsildenafil", r"\btadalafil", r"\bvardenafil", r"phosphodiesterase", r"\bPDE5\b",
          r"\bviagra", r"\bcialis"),
         "ABSOLUTE",
         "Nitrate given after a PDE5 inhibitor causes profound, refractory hypotension. "
         "The interaction persists 24h for sildenafil/vardenafil and up to 48h for tadalafil."),
    Rule("nitrate_in_rv_infarction", NITRATES,
         (r"right ventricular infarct", r"\bRV infarct", r"\bV4R\b", r"right[- ]sided lead"),
         "ABSOLUTE",
         "The right ventricle is preload-dependent in RV infarction. Nitrate-induced "
         "venodilatation drops preload and can precipitate irreversible hypotension."),
    Rule("nitrate_in_hypotension", NITRATES,
         (r"\bhypotension\b", r"\bcardiogenic shock", r"\bshocked\b", r"systolic below"),
         "ABSOLUTE",
         "Nitrates lower blood pressure further in an already hypotensive patient."),
    Rule("beta_blocker_in_shock_or_bradycardia", BETA_BLOCKERS,
         (r"cardiogenic shock", r"\bhypotension\b", r"\bbradycardia\b", r"\bAV block",
          r"acute heart failure", r"pulmonary o?edema", r"decompensated"),
         "ABSOLUTE",
         "Beta-blockade worsens cardiogenic shock, bradycardia and acute decompensated "
         "heart failure by further reducing rate and contractility."),
    Rule("beta_blocker_in_obstructive_airway_disease", BETA_BLOCKERS,
         (r"\basthma\b", r"bronchospasm", r"\bwheez", r"\bCOPD\b", r"chronic obstructive"),
         "CAUTION",
         "Non-selective beta-blockade can precipitate bronchospasm. If a beta-blocker is "
         "genuinely indicated, a cardioselective agent must be chosen deliberately."),
    Rule("rate_limiting_ccb_with_beta_blocker", RATE_LIMITING_CCB,
         (r"beta[- ]?blocker", r"\bmetoprolol", r"\bbisoprolol", r"\batenolol",
          r"\bbradycardia\b", r"\bAV block", r"heart failure"),
         "ABSOLUTE",
         "Verapamil or diltiazem combined with a beta-blocker, or given in bradycardia or "
         "heart failure, risks severe bradycardia, AV block and asystole."),

    # ------------------------------------------------------------- bleeding risk
    Rule("antithrombotic_in_active_bleeding", ANTIPLATELETS + ANTICOAGULANTS + THROMBOLYTICS,
         (r"active bleeding", r"\bhaemorrhag", r"\bhemorrhag", r"\bmelaena", r"\bmelena",
          r"haematemesis", r"\bGI bleed", r"gastrointestinal bleed", r"variceal"),
         "ABSOLUTE",
         "Antiplatelet, anticoagulant and thrombolytic therapy all worsen active bleeding."),
    Rule("thrombolysis_when_haemorrhage_possible", THROMBOLYTICS,
         (r"intracerebral haemorrhage", r"\bICH\b", r"haemorrhagic stroke", r"subarachnoid",
          r"head injury", r"head trauma", r"recent surgery", r"\bwarfarin", r"\bINR\b"),
         "ABSOLUTE",
         "Thrombolysis into an intracranial or surgical bleed is catastrophic. Haemorrhagic "
         "stroke must be excluded by imaging before any fibrinolytic is given."),
    Rule("anticoagulant_in_thrombocytopenia", ANTICOAGULANTS + ANTIPLATELETS,
         (r"thrombocytopeni", r"platelet\s*(?:count)?\s*(?:of|is|:)?\s*(?:[1-9]?\d)\b",
          r"\bdengue\b"),
         "CAUTION",
         "Bleeding risk rises sharply with a falling platelet count; in dengue the count "
         "falls further over the critical phase."),

    Rule("anticoagulant_added_to_antiplatelets",
         ORAL_ANTICOAGULANTS,
         (r"\bSTEMI\b", r"\bNSTEMI\b", r"acute coronary", r"myocardial infarction",
          r"\bACS\b", r"\bPCI\b"),
         "CAUTION",
         "An oral anticoagulant on top of dual antiplatelet therapy (triple therapy) "
         "markedly raises major bleeding. It is justified only for a specific separate "
         "indication such as atrial fibrillation or a mechanical valve - confirm that "
         "indication exists for this patient, and the duration, before prescribing."),

    # -------------------------------------------------------------------- dengue
    Rule("nsaid_or_aspirin_in_dengue", NSAIDS + (r"\baspirin", r"acetylsalicylic"),
         (r"\bdengue\b", r"dengue fever", r"dengue shock", r"thrombocytopeni"),
         "ABSOLUTE",
         "NSAIDs and aspirin are contraindicated in dengue: they impair platelet function "
         "and cause gastric erosion, compounding an already falling platelet count. "
         "Paracetamol is the antipyretic of choice."),
    Rule("im_injection_in_dengue", (r"intramuscular", r"\bIM\s+injection", r"\bIM\b"),
         (r"\bdengue\b", r"thrombocytopeni"),
         "CAUTION",
         "Intramuscular injection risks haematoma in a thrombocytopenic patient; use the "
         "oral or intravenous route."),

    # --------------------------------------------------------------------- renal
    Rule("metformin_in_renal_impairment_or_shock", (r"\bmetformin",),
         (r"renal impairment", r"\bCKD\b", r"chronic kidney", r"\beGFR\b", r"\bcreatinine",
          r"acute kidney", r"\bAKI\b", r"\bdialysis", r"\bshock\b", r"\bsepsis", r"contrast"),
         "ABSOLUTE",
         "Metformin accumulates when renal clearance falls or perfusion drops, causing "
         "lactic acidosis. Withhold in shock, sepsis, AKI and before contrast."),
    Rule("nsaid_in_renal_impairment", NSAIDS,
         (r"renal impairment", r"\bCKD\b", r"chronic kidney", r"\beGFR\b", r"acute kidney",
          r"\bAKI\b", r"\bhypotension\b", r"dehydrat", r"\bshock\b"),
         "ABSOLUTE",
         "NSAIDs reduce renal perfusion and precipitate acute-on-chronic kidney injury, "
         "especially in hypovolaemia or hypotension."),
    Rule("potassium_sparing_in_renal_impairment",
         (r"\bspironolactone", r"\bamiloride", r"potassium chloride", r"\bKCl\b") + ACE_ARB,
         (r"hyperkalaem", r"hyperkalem", r"\bCKD\b", r"chronic kidney", r"acute kidney"),
         "CAUTION",
         "ACE inhibitors, ARBs and potassium-sparing agents raise serum potassium; in renal "
         "impairment this can reach fatal levels."),

    # ---------------------------------------------------------------- respiratory
    Rule("uncontrolled_oxygen_in_copd", (r"\boxygen\b", r"\bO2\b", r"high[- ]flow"),
         (r"\bCOPD\b", r"chronic obstructive", r"carbon dioxide retention", r"\bhypercapni"),
         "CAUTION",
         "Uncontrolled high-flow oxygen in COPD risks CO2 retention and narcosis. Target a "
         "controlled saturation range rather than the highest achievable."),

    # ------------------------------------------------------------------ pregnancy
    Rule("teratogen_in_pregnancy",
         ACE_ARB + STATINS + TETRACYCLINES + QUINOLONES +
         (r"\bwarfarin", r"sodium valproate", r"\bvalproate", r"isotretinoin",
          r"\bmethotrexate", r"\bmisoprostol", r"\bfinasteride"),
         (r"\bpregnan", r"\bgestation", r"antenatal", r"\btrimester"),
         "ABSOLUTE",
         "Known or probable teratogen. ACE inhibitors/ARBs cause fetal renal failure, "
         "warfarin and valproate are teratogenic, tetracyclines stain fetal teeth and bone."),

    Rule("teratogen_when_gravid_status_unknown",
         ACE_ARB + STATINS + TETRACYCLINES + QUINOLONES +
         (r"\bwarfarin", r"sodium valproate", r"\bvalproate", r"isotretinoin",
          r"\bmethotrexate", r"\bmisoprostol", r"\bfinasteride"),
         (r"gravid status not established",),
         "CAUTION",
         "The same drug list as teratogen_in_pregnancy, one severity down. The "
         "intake records a patient of childbearing age whose pregnancy status "
         "was never established, which is not a contraindication - it is a "
         "question to answer before giving this. Kept deliberately separate "
         "from the ABSOLUTE rule, and keyed on a phrase carrying no "
         "\"pregnan\" substring, so an unasked question can never be reported "
         "as a known pregnancy: methotrexate is the correct treatment for an "
         "ectopic, and alarming on a right answer is how alarm fatigue starts."),

    # ----------------------------------------------------------------- paediatric
    Rule("aspirin_in_child", (r"\baspirin", r"acetylsalicylic"),
         (r"\bpaediatric patient\b", r"\bchild patient\b", r"\bage under 16\b"),
         "ABSOLUTE",
         "Aspirin in a febrile child under 16 is associated with Reye syndrome. Use "
         "paracetamol unless aspirin is specifically indicated (e.g. Kawasaki disease)."),
    Rule("tetracycline_in_young_child", TETRACYCLINES,
         (r"\bage under 8\b",),
         "ABSOLUTE",
         "Tetracyclines cause permanent dental staining and affect bone growth under 8 years."),
    Rule("codeine_or_tramadol_in_child", (r"\bcodeine", r"\btramadol"),
         (r"\bpaediatric patient\b", r"\bchild patient\b", r"\bage under 16\b"),
         "ABSOLUTE",
         "Unpredictable CYP2D6 ultra-rapid metabolism in children causes fatal opioid toxicity."),

    # ---------------------------------------------------------------- neuro/other
    Rule("sedation_masking_neuro_observation", SEDATIVES,
         (r"head injury", r"head trauma", r"\bGCS\b", r"reduced consciousness",
          r"intracerebral haemorrhage", r"\bICH\b"),
         "CAUTION",
         "Sedation obscures the conscious level that head-injury observation depends on. "
         "If unavoidable, document the pre-sedation GCS and the indication."),
    Rule("beta_blocker_masking_hypoglycaemia", BETA_BLOCKERS,
         (r"\bhypoglycaem", r"\bhypoglycem", r"glucose below 4"),
         "CAUTION",
         "Beta-blockade masks the adrenergic warning signs of hypoglycaemia."),
)

_COMPILED = tuple(
    (r, [re.compile(p, re.I) for p in r.drugs],
     [re.compile(p, re.I) for p in r.when],
     [re.compile(p, re.I) for p in r.unless])
    for r in RULES
)

_ALLERGY_RE = re.compile(
    r"allerg(?:y|ic|ies)\s*(?:to|:)?\s*([A-Za-z][A-Za-z0-9 ,/&+-]{2,80})", re.I)
_NO_ALLERGY_RE = re.compile(r"no known (?:drug )?aller|\bNKDA\b|allerg\w*\s*:?\s*(none|nil)", re.I)
_ALLERGY_STOP = frozenset("and or the to a an with of no known drug allergies allergy".split())


def stated_allergies(text: str) -> list[str]:
    """Drug names the intake says the patient reacts to."""
    if _NO_ALLERGY_RE.search(text or ""):
        return []
    out: list[str] = []
    for m in _ALLERGY_RE.finditer(text or ""):
        for token in re.split(r"[,/&+]| and ", m.group(1)):
            token = token.strip(" .;:")
            if len(token) > 3 and token.lower() not in _ALLERGY_STOP:
                out.append(token)
    return out


def check(context: str, items: list[tuple[str, str]]) -> list[Finding]:
    """`items` are (where, text) pairs - every place a treatment can be named.

    Both are matched case-insensitively: `context` is the patient picture
    (complaint, history, diagnosis, differentials, plus derived observations
    like "hypotension"), `items` are the recommendations to test against it.
    """
    findings: list[Finding] = []
    ctx = context or ""
    for rule, drugs, when, unless in _COMPILED:
        trig = next((m.group(0) for p in when if (m := p.search(ctx))), None)
        if not trig:
            continue
        if unless and any(p.search(ctx) for p in unless):
            continue
        for where, text in items:
            hit = next((m.group(0) for p in drugs if (m := p.search(text or ""))), None)
            if not hit:
                continue
            findings.append(Finding(rule.name, rule.severity, where, hit.strip(),
                                    trig.strip(), rule.reason))

    for allergen in stated_allergies(ctx):
        pat = re.compile(rf"(?<![a-z]){re.escape(allergen)}(?![a-z])", re.I)
        for where, text in items:
            if pat.search(text or ""):
                findings.append(Finding("stated_allergy", "ABSOLUTE", where, allergen,
                                        f"allergy to {allergen}",
                                        "The intake records an allergy to this drug."))
    return findings
