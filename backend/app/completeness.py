"""Per-presentation completeness checks: what the answer LEFT OUT.

Why this exists
---------------
Measured on a shocked inferior STEMI: the model produced aspirin, GTN and oxygen
and stopped. No second antiplatelet, no reperfusion strategy, no atropine for a
rate of 42, no volume loading for the RV infarct. Roughly a third of the required
management, in a report that read as complete.

Omission is the failure mode a reader cannot see. A contraindication announces
itself - a missing P2Y12 inhibitor looks exactly like a page that ends.

Design
------
These checks NEVER generate clinical content. They state that a required element
is absent and name the guideline to read, because a checklist that invents the
missing dose would be a second unreliable reasoner rather than a check on the
first. The clinician supplies the judgement; this only points at the gap.

Elements can be conditional (`when`), so volume loading is required only for an
RV infarct and atropine only for a bradycardia actually present in the vitals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Element:
    name: str
    present: tuple[str, ...]        # evidence in the output that it WAS addressed
    why: str
    when: tuple[str, ...] = ()      # required only if the context matches


@dataclass(frozen=True)
class Presentation:
    key: str
    label: str
    match: tuple[str, ...]          # matched against diagnosis + complaint
    guideline: str
    elements: tuple[Element, ...]


@dataclass(frozen=True)
class Gap:
    presentation: str
    element: str
    why: str
    guideline: str


PRESENTATIONS: tuple[Presentation, ...] = (
    Presentation(
        "acs", "Acute coronary syndrome",
        (r"\bSTEMI\b", r"\bNSTEMI\b", r"\bNSTE[- ]?ACS\b", r"acute coronary",
         r"myocardial infarction", r"\bMI\b(?!\w)"),
        "Management of Acute Coronary Syndromes (1st Ed, 2026)",
        (
            Element("Aspirin loading dose", (r"\baspirin", r"acetylsalicylic"),
                    "Antiplatelet loading is the first pharmacological step in ACS."),
            Element("Second antiplatelet (P2Y12)",
                    (r"\bclopidogrel", r"\bticagrelor", r"\bprasugrel", r"\bP2Y12\b",
                     r"dual antiplatelet", r"\bDAPT\b"),
                    "Dual antiplatelet therapy, not aspirin alone, is the standard of care."),
            Element("Reperfusion decision (PCI or fibrinolysis)",
                    (r"\bPCI\b", r"percutaneous coronary", r"fibrinolysi", r"thrombolysi",
                     r"reperfusion", r"cath ?lab", r"angiograph"),
                    "Time to reperfusion determines myocardial salvage; a transfer time "
                    "must be weighed against giving a fibrinolytic locally."),
            Element("12-lead ECG", (r"\bECG\b", r"electrocardiogram"),
                    "Confirms territory and guides the reperfusion decision."),
            Element("Volume loading for RV infarction",
                    (r"\bfluid", r"\bbolus", r"crystalloid", r"normal saline", r"preload"),
                    "The RV is preload-dependent: an inferior infarct with V4R elevation "
                    "needs volume, and nitrates/diuretics are harmful.",
                    when=(r"\bV4R\b", r"right ventricular infarct", r"\bRV infarct")),
            Element("Atropine or pacing for bradycardia",
                    (r"\batropine", r"\bpacing\b", r"pacemaker", r"transcutaneous"),
                    "Inferior infarction commonly causes AV block; a rate of under 50 with "
                    "shock needs a rate strategy.",
                    when=(r"\bbradycardia\b", r"\bAV block")),
        ),
    ),
    Presentation(
        "dengue", "Dengue infection",
        (r"\bdengue\b", r"dengue shock", r"dengue fever"),
        "Management of Dengue Infection in Adults (3rd Ed) / in Children (2nd Ed)",
        (
            Element("Weight-based fluid resuscitation",
                    (r"ml\s*/\s*kg", r"\bbolus", r"crystalloid", r"normal saline",
                     r"hartmann", r"\bfluid"),
                    "Plasma leakage is the cause of dengue shock; fluid therapy is the "
                    "treatment and must be weight-based."),
            Element("Haematocrit and platelet monitoring",
                    (r"h[ae]matocrit", r"\bHCT\b", r"platelet", r"\bFBC\b", r"full blood count"),
                    "Serial haematocrit detects plasma leakage before blood pressure falls."),
            Element("Warning-sign assessment",
                    (r"warning sign", r"abdominal pain", r"persistent vomiting",
                     r"mucosal bleed", r"lethargy", r"restless"),
                    "Warning signs define the critical phase and the level of care."),
            Element("Paracetamol as antipyretic (NSAIDs withheld)",
                    (r"paracetamol", r"acetaminophen"),
                    "Paracetamol is the antipyretic of choice; NSAIDs and aspirin are "
                    "contraindicated in dengue."),
        ),
    ),
    Presentation(
        "stroke", "Acute stroke",
        (r"ischaemic stroke", r"ischemic stroke", r"\bCVA\b", r"cerebrovascular accident",
         r"acute stroke"),
        "Management of Ischaemic Stroke (3rd Ed, 2020)",
        (
            Element("Urgent brain imaging",
                    (r"\bCT\b", r"computed tomograph", r"\bMRI\b", r"\bimaging\b", r"\bscan\b"),
                    "Ischaemic and haemorrhagic stroke are indistinguishable clinically and "
                    "are treated oppositely."),
            Element("Thrombolysis eligibility / time of onset",
                    (r"thrombolysi", r"alteplase", r"\bonset\b", r"time window",
                     r"last (?:seen )?well", r"thrombectom"),
                    "Reperfusion is time-critical and eligibility turns on onset time."),
            Element("Blood glucose check",
                    (r"\bglucose", r"\bBSL\b", r"\bRBS\b", r"hypoglyc"),
                    "Hypoglycaemia is a stroke mimic that is instantly reversible."),
            Element("Swallow assessment / NBM",
                    (r"swallow", r"\bNBM\b", r"nil by mouth", r"aspiration"),
                    "Unassessed swallowing after stroke causes aspiration pneumonia."),
        ),
    ),
    Presentation(
        "ich", "Intracerebral haemorrhage",
        (r"intracerebral haemorrhage", r"intracerebral hemorrhage", r"\bICH\b",
         r"haemorrhagic stroke"),
        "Management of Spontaneous Intracerebral Haemorrhage (1st Ed, 2025)",
        (
            Element("Blood pressure control", (r"blood pressure", r"\bBP\b", r"antihypertens",
                                               r"labetalol", r"\bGTN\b", r"nicardipine"),
                    "Early controlled BP lowering limits haematoma expansion."),
            Element("Anticoagulation reversal",
                    (r"revers", r"vitamin k", r"prothrombin complex", r"\bPCC\b",
                     r"idarucizumab", r"protamine", r"\bINR\b"),
                    "An anticoagulated intracerebral bleed expands until reversed.",
                    when=(r"\bwarfarin", r"\bDOAC\b", r"anticoagulat", r"\bheparin",
                          r"\bdabigatran", r"\brivaroxaban", r"\bapixaban")),
            Element("Neurosurgical referral",
                    (r"neurosurg", r"\brefer", r"\bsurgical\b"),
                    "Haematoma location and size determine whether surgery is indicated."),
        ),
    ),
    Presentation(
        "asthma_copd", "Acute asthma / COPD exacerbation",
        (r"\basthma\b", r"\bCOPD\b", r"chronic obstructive", r"bronchospasm",
         r"acute exacerbation"),
        "Management of Asthma in Adults (2nd Ed, 2024) / COPD (2nd Ed, 2009)",
        (
            Element("Inhaled bronchodilator",
                    (r"salbutamol", r"albuterol", r"ipratropium", r"nebuli", r"bronchodilat"),
                    "Bronchodilation is the immediate treatment of airflow obstruction."),
            Element("Systemic corticosteroid",
                    (r"prednisolone", r"hydrocortisone", r"dexamethasone", r"corticosteroid",
                     r"\bsteroid"),
                    "Steroids shorten exacerbations and reduce relapse."),
            Element("Severity assessment",
                    (r"peak flow", r"\bPEFR\b", r"\bFEV1\b", r"silent chest",
                     r"accessory muscle", r"severity", r"life[- ]threatening"),
                    "Severity determines disposition and escalation."),
            Element("Controlled oxygen target",
                    (r"\boxygen", r"\bSpO2\b", r"saturation", r"88[-–]92"),
                    "Oxygen targets differ between asthma and COPD."),
        ),
    ),
    Presentation(
        "hyperglycaemia", "Diabetic emergency",
        (r"\bDKA\b", r"diabetic ketoacidosis", r"\bHHS\b", r"hyperosmolar",
         r"hyperglycaem", r"hyperglycem"),
        "Management of Type 2 Diabetes Mellitus (6th Ed) / T1DM in Children (2016)",
        (
            Element("Intravenous fluid resuscitation",
                    (r"\bfluid", r"normal saline", r"\bbolus", r"crystalloid", r"ml\s*/\s*kg"),
                    "Dehydration, not hyperglycaemia, is the immediate threat in DKA."),
            Element("Insulin therapy", (r"\binsulin",),
                    "Insulin stops ketogenesis; the fluid must come first."),
            Element("Potassium monitoring / replacement",
                    (r"potassium", r"\bK\+", r"\bKCl\b", r"electrolyte"),
                    "Insulin drives potassium intracellularly and can cause fatal hypokalaemia."),
            Element("Ketone and acid-base assessment",
                    (r"\bketone", r"\bABG\b", r"blood gas", r"\bpH\b", r"bicarbonate",
                     r"acid[- ]base", r"anion gap"),
                    "Defines whether this is DKA and tracks resolution."),
        ),
    ),
    Presentation(
        "heart_failure", "Acute heart failure",
        (r"heart failure", r"pulmonary o?edema", r"\bAPO\b", r"decompensated"),
        "Management of Heart Failure (5th Ed, 2023)",
        (
            Element("Diuretic therapy", (r"frusemide", r"furosemide", r"diuretic"),
                    "Decongestion is the primary treatment of fluid overload."),
            Element("Oxygen / ventilatory support",
                    (r"\boxygen", r"\bCPAP\b", r"\bNIV\b", r"non[- ]invasive", r"ventilat"),
                    "Non-invasive ventilation relieves pulmonary oedema rapidly."),
            Element("Precipitant identified",
                    (r"precipitat", r"\bECG\b", r"ischaem", r"arrhythmi", r"infection",
                     r"non[- ]compliance", r"\bcause\b"),
                    "Treating the episode without its cause guarantees recurrence."),
        ),
    ),
    Presentation(
        "head_injury", "Head injury",
        (r"head injury", r"head trauma", r"traumatic brain"),
        "Early Management of Head Injury in Adults (2015)",
        (
            Element("GCS documented", (r"\bGCS\b", r"glasgow", r"conscious level"),
                    "GCS drives both triage acuity and imaging decisions."),
            Element("CT head decision",
                    (r"\bCT\b", r"computed tomograph", r"imaging", r"\bscan\b"),
                    "Imaging criteria decide who needs urgent neurosurgical assessment."),
            Element("Cervical spine consideration",
                    (r"cervical", r"c[- ]spine", r"collar", r"immobilis"),
                    "Significant head injury implies possible cervical spine injury."),
            Element("Neurological observation",
                    (r"observ", r"monitor", r"neuro", r"pupil"),
                    "Deterioration after a lucid interval is the classic missed event."),
        ),
    ),
    Presentation(
        "tuberculosis", "Tuberculosis",
        (r"tuberculosis", r"\bTB\b(?!\w)", r"pulmonary tb"),
        "Management of Tuberculosis (4th Ed, 2021)",
        (
            Element("Respiratory isolation",
                    (r"isolat", r"negative pressure", r"\bmask", r"infection control",
                     r"airborne"),
                    "Isolation is a triage decision, not a treatment decision - it protects "
                    "staff and other patients from the moment TB is suspected."),
            Element("Sputum for AFB / GeneXpert",
                    (r"sputum", r"\bAFB\b", r"genexpert", r"smear", r"culture",
                     r"acid[- ]fast"),
                    "Microbiological confirmation and drug-resistance testing."),
            Element("Chest radiograph", (r"chest x[- ]?ray", r"\bCXR\b", r"chest radiograph"),
                    "Assesses extent and cavitation."),
            Element("Notification", (r"notif", r"\bregist", r"public health"),
                    "TB is a statutorily notifiable disease in Malaysia."),
        ),
    ),
    Presentation(
        "hip_fracture", "Fragility hip fracture",
        (r"hip fracture", r"neck of femur", r"\bNOF\b", r"femoral neck"),
        "Management of Geriatric Hip Fracture (2023)",
        (
            Element("Analgesia", (r"analges", r"pain relief", r"paracetamol", r"morphine",
                                  r"nerve block", r"fascia iliaca"),
                    "Untreated pain in the elderly precipitates delirium."),
            Element("Imaging", (r"x[- ]?ray", r"radiograph", r"\bCT\b", r"\bMRI\b"),
                    "Confirms the fracture and its configuration."),
            Element("Orthopaedic referral / surgical timing",
                    (r"orthopaed", r"orthoped", r"surg", r"\brefer", r"theatre"),
                    "Delay to surgery beyond 48 hours increases mortality."),
            Element("VTE risk assessment",
                    (r"\bVTE\b", r"thromboprophylax", r"enoxaparin", r"heparin",
                     r"venous thrombo"),
                    "Immobility after hip fracture carries a high thrombosis risk."),
        ),
    ),
    Presentation(
        "neonatal_jaundice", "Neonatal jaundice",
        (r"neonatal jaundice", r"kernicterus", r"hyperbilirubin"),
        "Management of Neonatal Jaundice (2nd Ed, 2014)",
        (
            Element("Serum bilirubin level",
                    (r"bilirubin", r"\bSB\b", r"\bTSB\b", r"transcutaneous"),
                    "Treatment thresholds are defined by level against age in hours."),
            Element("Phototherapy decision", (r"phototherap", r"exchange transfusion"),
                    "Phototherapy against the age-specific threshold prevents kernicterus."),
            Element("Cause sought",
                    (r"\bG6PD\b", r"blood group", r"coombs", r"sepsis", r"haemolys",
                     r"\bcause\b", r"aetiolog"),
                    "G6PD deficiency and haemolysis change urgency and prognosis."),
        ),
    ),
)

_COMPILED = tuple(
    (p,
     [re.compile(x, re.I) for x in p.match],
     [(e, [re.compile(x, re.I) for x in e.present], [re.compile(x, re.I) for x in e.when])
      for e in p.elements])
    for p in PRESENTATIONS
)


def check(context: str, answer: str) -> tuple[str, list[Gap]]:
    """`context` is the patient picture, `answer` is everything the model wrote.

    Returns the presentation label matched (empty when none) and the gaps. Only
    the FIRST matching presentation is used: reporting the union across several
    would bury the relevant gaps in irrelevant ones.
    """
    ctx = context or ""
    ans = answer or ""
    for pres, match, elements in _COMPILED:
        if not any(p.search(ctx) for p in match):
            continue
        gaps: list[Gap] = []
        for element, present, when in elements:
            if when and not any(p.search(ctx) for p in when):
                continue  # not required for this patient
            if any(p.search(ans) for p in present):
                continue  # addressed
            gaps.append(Gap(pres.label, element.name, element.why, pres.guideline))
        return pres.label, gaps
    return "", []
