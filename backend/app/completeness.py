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
    ),    # Added 2026-09-30. Each element's reason names the KKM passage it rests
    # on; where no KKM document covers it, the reason says so ("KKM gap") and
    # the report links out (references.py) - no foreign source is quoted.
    Presentation(
        "rhabdomyolysis", "Rhabdomyolysis",
        (r"rhabdomyoly", r"myoglobinuri"),
        "No KKM rhabdomyolysis guideline - MOH Heat Related Illness 2016 s6.1 (investigations); "
        "Malaysian Hyperkalaemia Consensus 2024; MOH Snakebite 2017 s4.5 (serial CK)",
        (
            Element("IV isotonic fluid with a rate or urine-output target",
                    (r"\bfluid", r"saline", r"crystalloid", r"hartmann", r"ringer"),
                    "Fluid resuscitation prevents myoglobin-induced kidney injury. KKM gap: no indexed "
                    "KKM document gives an adult rhabdomyolysis fluid regimen, so the rate is the "
                    "clinician's - the report must not present a general range as the regimen."),
            # "Creatinine kinase" is the MOH 2016 guideline's own (mis)spelling,
            # s6.1.8 - matched so its sentence can be quoted for this gap.
            Element("Creatine kinase", (r"creatine (?:phospho)?kinase", r"creatinine kinase",
                                        r"muscle enzyme", r"\bCK\b", r"\bCPK\b"),
                    "CK confirms the diagnosis and is followed to its peak (MOH 2016 s6.1.8)."),
            # SERUM renal function. "Urine analysis for myoglobin and creatinine"
            # satisfied this on run 10 - urine creatinine is not kidney function.
            Element("Serum urea and creatinine (renal profile)",
                    # A named SERUM test - "to assess renal function" in the
                    # rationale of a urine test is not one (run 10 dry run).
                    (r"(?:serum|plasma|blood) (?:urea|creatinine)(?! kinase)", r"renal (?:profile|panel|function test)",
                     r"\burea and (?:serum )?creatinine\b", r"\bBUSE\b", r"\bRP\b", r"\bU&E\b", r"\bBUN\b",
                     r"\bRFT\b"),
                    "Kidney injury is the commonest early complication (MOH 2016 s6.1.7: renal function test)."),
            Element("Potassium", (r"potassium", r"\bK\+", r"electrolyte", r"\bBUSE\b"),
                    "Hyperkalaemia is the early killer (Hyperkalaemia Consensus 2024)."),
            Element("Serial CK and bloods",
                    (r"(?:repeat|serial|trend|recheck|every \d+.{0,6}h\w*).{0,40}(?:\bCK\b|creatine|labs?|bloods?)",
                     r"(?:labs?|laboratory results?|bloods?|\bCK\b).{0,30}every \d+",
                     r"(?:\bCK\b|creatine kinase).{0,60}(?:repeat|serial|trend|recheck|every)"),
                    "MOH Snakebite 2017 s4.5.4: \"Creatine kinase: For early detection of "
                    "rhabdomyolysis. Serial monitoring to monitor trend.\""),
            Element("12-lead ECG / cardiac monitoring",
                    (r"\bECG\b", r"electrocardiogra", r"cardiac monitor"),
                    "Hyperkalaemia shows on the ECG before it causes symptoms (Hyperkalaemia Consensus 2024)."),
            Element("Urine output monitoring",
                    (r"urine output", r"input[- ]?output", r"fluid balance", r"\bcatheter"),
                    "Fluid therapy is titrated to urine output."),
            Element("Compartment syndrome assessment", (r"compartment",),
                    "A complication of muscle injury needing surgical review. KKM gap: no indexed KKM "
                    "document addresses it for rhabdomyolysis."),
            Element("Analgesia",
                    (r"paracetamol", r"acetaminophen", r"\bopioid", r"morphine", r"analges"),
                    "Pain is scored and treated (MOH Pain Management in ETD 2020). KKM gap: no indexed "
                    "KKM document states NSAID avoidance for rhabdomyolysis itself.",
                    when=(r"\bpain\b", r"\bache", r"pain score (?:[1-9]|10)\b")),
            Element("Haemolysis / G6PD deficiency considered",
                    (r"h[ae]moglobinuri", r"h[ae]moly", r"\bG6PD\b"),
                    "Haemoglobinuria looks like myoglobinuria on the dipstick, and G6PD deficiency is "
                    "common in Malaysia; the answer should say how it is excluded."),
            Element("Other causes of dark urine considered",
                    (r"h[ae]moglobinuri", r"h[ae]moly", r"h[ae]maturi", r"glomerulonephrit", r"\bG6PD\b",
                     r"heat[- ](?:stroke|illness|injur|related)", r"dengue", r"statin", r"myositis", r"hepatitis",
                     r"bilirubin"),
                    "Myoglobinuria, haemoglobinuria (haemolysis, e.g. G6PD) and haematuria look alike; "
                    "heat illness, dengue myositis and statin myopathy are local causes to exclude."),
            Element("Admission for IV fluids and serial CK",
                    (r"\badmi(?:t|ssion)", r"\bward\b", r"\bHDU\b", r"\bICU\b", r"ADMIT_"),
                    "Fluids are titrated over days against CK and kidney function. KKM gap: no indexed "
                    "KKM adult document states admission criteria for rhabdomyolysis; the Dengue CPG "
                    "(2015) lists \"Acute rhabdomyolysis with renal insufficiency\" among its ICU "
                    "criteria."),
        ),
    ),
    Presentation(
        "heat_illness", "Heat stroke / heat-related illness",
        (r"heat ?stroke", r"heat exhaustion", r"heat[- ]related illness", r"exertional hyperthermia"),
        "MOH Clinical Guidelines on Management of Heat Related Illness (2016)",
        (
            Element("Immediate active cooling",
                    (r"\bcool", r"\bice\b", r"ice pack", r"evaporat", r"tepid", r"mist"),
                    "Cooling without delay is what makes survival approach 100% (s1.1, s7.2.8)."),
            Element("Core temperature monitoring",
                    (r"core (?:body )?temp", r"rectal", r"tympanic", r"oesophageal", r"temperature monitor"),
                    "Target a fall to about 38-39 C without overcorrecting (s7.2.7, s7.2.9)."),
            Element("IV fluids guided by haemodynamics and urine output",
                    (r"\bfluid", r"saline", r"crystalloid"),
                    "Ensure urine output > 0.5 ml/kg/h in adults (s7.2.6.1.1)."),
            Element("Antipyretics withheld",
                    (r"(?:no|avoid\w*|withh\w*|not)\s[^.]{0,30}(?:paracetamol|antipyretic|nsaid|aspirin)",),
                    "\"DO NOT administer Paracetamol or Aspirin or other NSAIDS\" (s7.2.10)."),
            Element("End-organ work-up (CK, renal, liver, coagulation, glucose)",
                    (r"creatine kinase", r"creatinine kinase", r"muscle enzyme", r"\bCK\b", r"renal",
                     r"liver function", r"\bLFT", r"coagul", r"glucose", r"\bCBG\b"),
                    "No test diagnoses heat stroke; the work-up detects end-organ damage (s6)."),
        ),
    ),
    Presentation(
        "snakebite", "Snakebite",
        (r"snake", r"envenom", r"\bviper", r"\bcobra", r"\bkrait"),
        "MOH Guideline: Management of Snakebite (2017)",
        (
            Element("Immobilise the bitten limb", (r"immobili", r"splint"),
                    "Limits venom spread; tourniquets and incisions are not used."),
            Element("Clotting assessment (20WBCT / coagulation profile)",
                    (r"20\s?WBCT", r"whole blood clotting", r"clotting time", r"coagulation",
                     r"\bPT\b", r"\bINR\b", r"\bAPTT\b"),
                    "Coagulopathy decides antivenom in haematotoxic bites."),
            Element("Antivenom indication assessed",
                    (r"anti-?venom", r"antivenin"),
                    "Antivenom is the specific treatment when systemic or progressive local "
                    "envenoming is present."),
            Element("Serial observation", (r"serial", r"snakebite chart", r"observ", r"monitor"),
                    "Envenoming evolves over hours; a normal first assessment does not exclude it."),
        ),
    ),
    Presentation(
        "hyperkalaemia", "Hyperkalaemia",
        (r"hyperkal[ae]?emi", r"high potassium"),
        "Malaysian Consensus on the Management of Acute and Persistent Hyperkalaemia (2024)",
        (
            Element("12-lead ECG and cardiac monitoring", (r"\bECG\b", r"cardiac monitor"),
                    "ECG changes mean immediate treatment."),
            Element("IV calcium for ECG changes",
                    (r"calcium gluconate", r"calcium chloride", r"\bIV calcium"),
                    "Calcium stabilises the myocardium within minutes."),
            Element("Insulin-dextrose to shift potassium", (r"\binsulin",),
                    "The first-line shifting therapy."),
            Element("Repeat potassium", (r"repeat\w* (?:serum )?potassium", r"recheck", r"serial potassium",
                                         r"repeat (?:K|RP|renal)"),
                    "Shifting therapy is temporary; the level rebounds."),
        ),
    ),
    # ---- Added 2026-09-30 with the MOH obstetric, psychiatric, rheumatology
    # and dental documents. Every "why" quotes or points at its page; an
    # element the source does not state is left out rather than supplied.
    Presentation(
        "pph", "Postpartum haemorrhage",
        (r"post-?partum ha?emorrhag", r"\bPPH\b", r"uterine atony"),
        "MOH Quick Reference Guide Postpartum Haemorrhage (PPH) 2016; MOH Perinatal Care Manual (4th Edition) 2020",
        (
            Element("Call for help - obstetric emergency (Red Alert)",
                    (r"red alert", r"call for (?:senior |obstetric |extra )?help", r"obstetric (?:team|registrar|specialist|on-?call)",
                     r"\bO&G\b", r"\bO ?& ?G\b", r"gyn(?:a)?ecolog"),
                    "PPH QRG p36: \"Extra Help & Resuscitation - Activate RED ALERT\"."),
            Element("Two large-bore IV lines with fluid / blood",
                    (r"large[- ]bore", r"two (?:IV|intravenous|peripheral)", r"\b1[46] ?G\b",
                     r"crystalloid|hartmann|normal saline|fluid resuscitat", r"blood transfusion|packed (?:red )?cells|whole blood"),
                    "PPH QRG p36: \"IV access (2 large bore >=16G): Fluid +/- blood\"."),
            Element("Uterotonic and uterine massage",
                    (r"oxytocin|syntocinon|carbetocin|syntometrine|ergometrine|misoprostol|carboprost",
                     r"uterine massage|massage (?:the )?uterus|rub(?:bing)? up"),
                    "PPH QRG p20: \"Uterine Atony (70%) - Massage uterus - Ensure 3rd stage oxytocin given\"."),
            Element("Group and crossmatch, FBC and coagulation",
                    (r"cross[- ]?match|group and (?:cross|save)|\bGXM\b|\bGSH\b", r"coagulation|\bPT\b|\bAPTT\b|fibrinogen"),
                    "Blood and blood products are given against these (PPH QRG, Resuscitation and Monitoring)."),
            Element("Tranexamic acid", (r"tranexamic",),
                    "PPH QRG p23 lists IV tranexamic acid among the resuscitation drugs."),
            Element("Cause sought - tone, tissue, trauma, thrombin",
                    (r"retained (?:placenta|products|tissue)", r"genital tract (?:trauma|tear)|perineal tear|laceration",
                     r"\b4 ?T'?s\b", r"tone.{0,30}tissue", r"uterine (?:inversion|rupture)"),
                    "PPH QRG pp16-30: atony, retained placenta, genital tract trauma, uterine inversion and rupture."),
            Element("Massive transfusion protocol considered", (r"massive transfusion", r"\bMTP\b"),
                    "PPH QRG p20: \"Consider Massive Transfusion Protocol activation\".",
                    when=(r"\bshock\b", r"hypotens", r"SBP ?<? ?[5-8]\d\b", r"BP [5-8]\d ?/", r"HR 1[2-9]\d")),
            Element("Ergometrine avoided in hypertension or cardiac disease",
                    (r"(?:avoid|contraindicat\w*|not|no)\b.{0,40}(?:ergometrine|syntometrine)",
                     r"(?:ergometrine|syntometrine).{0,60}(?:avoid|contraindicat)"),
                    "PPH QRG p11: Syntometrine \"Contraindicated in patients with cardiac disease or hypertension\".",
                    when=(r"hypertens", r"pre-?eclampsi", r"cardiac disease|heart disease", r"BP 1[4-9]\d ?/")),
        ),
    ),
    Presentation(
        "hdp", "Eclampsia / severe pre-eclampsia",
        (r"eclampsi", r"\bHELLP\b"),
        "MOH Training Manual Hypertensive Disorders in Pregnancy (3rd Edition) 2018; MOH Perinatal Care Manual (4th Edition) 2020",
        (
            Element("Magnesium sulphate loading dose",
                    (r"magnesium sul(?:ph|f)ate", r"\bMgSO4\b", r"\bMgSO\s*4\b"),
                    "HDP Manual p103: \"Loading dose: IV 4g MgSO4 slow bolus - over 10-15 minutes (rapid "
                    "injection causes cardiac arrest)\"."),
            Element("Magnesium maintenance and toxicity monitoring",
                    (r"1 ?g ?(?:/|per) ?h", r"maintenance (?:infusion|dose)", r"(?:patellar|tendon|knee) (?:jerk|reflex)",
                     r"magnesium toxicity", r"calcium gluconate"),
                    "HDP Manual p41: \"maintenance IV infusion of MgSO4 1g per hour\"; p43: \"Monitor magnesium toxicity\"."),
            Element("Antihypertensive for severe blood pressure",
                    (r"labetalol|hydralazine|nifedipine|antihypertensiv",),
                    "HDP Manual p40: \"give oral nifedipine (10mg stat) or IM hydralazine 6.25mg\".",
                    when=(r"\b1[6-9]\d ?/", r"\b2\d\d ?/", r"/ ?1[1-9]\d\b", r"severe hypertension", r"eclampsi")),
            Element("Fetal heart monitoring",
                    (r"\bCTG\b", r"fetal heart|foetal heart", r"cardiotocograph", r"fetal (?:monitoring|wellbeing)"),
                    "HDP Manual p41: monitor \"the fetal heart rate every 15 minutes\"; p59: continuous fetal heart "
                    "monitoring is mandatory until BP is stable.",
                    when=(r"\bpregnan", r"weeks? (?:pregnant|gestation)", r"\bgestation", r"antenatal")),
            Element("Bloods for HELLP - FBC/platelets, liver enzymes, renal",
                    (r"platelet", r"\bFBC\b|full blood count", r"\bLFT\b|liver (?:function|enzyme)|transaminase", r"HELLP"),
                    "HDP Manual p15: HELLP \"is a severe form of PE manifested by Haemolysis, Elevated Liver Enzymes "
                    "and Low Platelets\"."),
            Element("Obstetric team and delivery plan",
                    (r"obstetric", r"\bO ?& ?G\b", r"labou?r (?:room|ward|suite)", r"deliver(?:y|ed|ing)\b", r"caesarean"),
                    "HDP Manual p36: \"If at anytime the maternal and fetal condition is compromised, early delivery is "
                    "mandatory\"."),
        ),
    ),
    Presentation(
        "agitation", "Acute agitation / psychosis",
        (r"agitat", r"psychos[ie]s|psychotic", r"schizophren", r"\bmani(?:a|c)\b", r"aggressi|violent|combative"),
        "Management of Schizophrenia (2nd Edition) 2021; Management of Bipolar Disorder (2nd Edition) 2024",
        (
            Element("Organic cause and intoxication excluded",
                    (r"glucose|\bCBG\b|hypoglyc", r"toxicolog|intoxicat|drug screen|urine drug|substance|alcohol",
                     r"delirium|organic|encephalopath"),
                    "Schizophrenia CPG p83 (ICD-10 G3): the disorder must not be attributable to organic brain disease "
                    "or to alcohol- or drug-related intoxication."),
            Element("Risk assessment - to self and others",
                    (r"risk assessment|risk (?:to|of harm to) (?:self|others)", r"suicid", r"harm to (?:self|others)",
                     r"violence risk"),
                    "Schizophrenia CPG p27: severity is assessed on psychopathology \"and risk assessment (risk to self "
                    "and/or others)\"."),
            Element("Rapid tranquillisation if needed - IM haloperidol with lorazepam or promethazine",
                    (r"haloperidol|lorazepam|midazolam|olanzapine|promethazine|zuclopenthixol|rapid tranquil",),
                    "Schizophrenia CPG p32: \"When rapid tranquillisation is urgently needed, a combination of IM "
                    "haloperidol plus lorazepam or promethazine should be\" considered; a single agent is preferred "
                    "where possible.",
                    when=(r"agitat", r"aggressi|violent|combative|threaten|restrain")),
            Element("Psychiatric referral", (r"psychiatr",),
                    "Assessment and treatment of psychosis rest with the psychiatric team (Schizophrenia CPG, Referral)."),
        ),
    ),
    Presentation(
        "lithium", "Lithium toxicity",
        (r"lithium toxic", r"lithium (?:level|overdose|poisoning)"),
        "Management of Bipolar Disorder (2nd Edition) 2024",
        (
            Element("Serum lithium level", (r"(?:serum |plasma )?lithium (?:level|concentration)",),
                    "Bipolar CPG p62: toxicity (tremor, tinnitus, seizure, ataxia); acute-mania range 0.8-1.2 mmol/L."),
            Element("Renal function and hydration", (r"renal|creatinine|\bRP\b|\bBUSE\b|urea", r"fluid|hydrat"),
                    "Bipolar CPG p62: \"Caution use during periods of dehydration\"."),
        ),
    ),
    Presentation(
        "gout", "Acute gout flare",
        (r"\bgout", r"podagra"),
        "Management of Gout (2nd Edition) 2021",
        (
            Element("Septic arthritis excluded",
                    (r"septic arthritis", r"joint aspirat|arthrocentesis|synovial fluid|aspirat\w* (?:of )?(?:the )?joint"),
                    "Gout CPG p38: \"septic arthritis (key differential diagnosis)\" - hot, swollen joint with fever."),
            Element("Flare treatment - colchicine, NSAID or corticosteroid",
                    (r"colchicine", r"\bNSAID", r"naproxen|diclofenac|etoricoxib|indomethacin|ibuprofen",
                     r"prednisolone|corticosteroid"),
                    "Gout CPG p53: prednisolone 30 mg OD for 5 days was as effective as NSAIDs for flare pain."),
            Element("Renal function checked before colchicine or NSAID",
                    (r"renal|creatinine|eGFR|\bRP\b|\bBUSE\b|\bCKD\b",),
                    "Gout CPG p56: colchicine \"should be avoided in severe CKD\"; NSAIDs \"should also be avoided in "
                    "CKD\"."),
            Element("Diuretic reviewed as a precipitant", (r"thiazide|diuretic",),
                    "Gout CPG p8: \"diuretics should be avoided if possible, or replaced by an alternative drug\".",
                    when=(r"thiazide|hydrochlorothiazide|indapamide|furosemide|frusemide|diuretic",)),
        ),
    ),
    Presentation(
        "avulsion", "Avulsed permanent tooth",
        (r"avuls", r"(?:tooth|teeth|incisor)\w*.{0,30}knocked out|knocked[- ]out (?:tooth|teeth)"),
        "Management of Avulsed Permanent Anterior Teeth (3rd Edition) 2019",
        (
            Element("Replantation (or correct storage) without delay", (r"replant\w*|reimplant\w*|re-?implant\w*",),
                    "Avulsed Teeth CPG p23: immediate or early replantation - \"<15 minutes extra-alveolar dry time or "
                    "<60 minutes stored in recommended storage medium\"."),
            Element("Storage medium - milk, saline or saliva", (r"\bmilk\b", r"saline", r"saliva", r"storage medi"),
                    "Avulsed Teeth CPG p13: \"Place in suitable storage medium (fresh milk/saline/ patient's saliva)\"."),
            Element("Handle by the crown, not the root", (r"\bcrown\b", r"(?:do not|don't|avoid) touch\w* (?:the )?root"),
                    "Avulsed Teeth CPG p23: \"Handle the tooth by its crown. Do not touch the root.\""),
            Element("Tetanus status", (r"tetanus|\bATT\b|\bADT\b",),
                    "Avulsed Teeth CPG p52: the referral form asks for tetanus injection given."),
            Element("Dental referral", (r"dental|dentist|oral (?:surgeon|surgery)|maxillofacial",),
                    "Avulsed Teeth CPG p13-14: go to the nearest dental clinic; refer to a dental specialist."),
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


def element(presentation_label: str, element_name: str) -> Element | None:
    """The Element behind a Gap, for callers that need its evidence patterns
    (rag_engine uses them to find the guideline sentence that covers it)."""
    for pres in PRESENTATIONS:
        if pres.label != presentation_label:
            continue
        for el in pres.elements:
            if el.name == element_name:
                return el
    return None
