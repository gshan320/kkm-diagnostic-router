"""Deterministic recall floor for can't-miss presentations.

Layers 1-3 of retrieval (medical embeddings, BM25, cross-encoder rerank) make a
miss rare. None of them makes a miss *impossible* - no embedder offers a recall
guarantee. This module is the guarantee.

Each rule is a plain, auditable statement: "if the intake says X, the guideline
for Y is put in front of the model, whatever the retriever scored it." A
practitioner can read these rules, argue with them, and version them. That is
the point - the safety net is not allowed to be a black box.

Rules only ever ADD sources. They never remove or reorder what retrieval found,
so a firing rule cannot mask a correct retrieval.

The same shape as `_enforce_mts_level` in rag_engine: clinical safety is
enforced in code, not requested in a prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RedFlag:
    name: str
    # Fires when any `triggers` pattern matches. If `require` is non-empty, one
    # of those must ALSO match - that is how combination findings are expressed
    # ("calf swelling" alone is weak; with pleuritic chest pain it is not).
    triggers: tuple[str, ...]
    titles: tuple[str, ...]
    note: str
    require: tuple[str, ...] = ()
    # Does NOT fire when any of these match: "soaking pads" two hours after a
    # delivery is postpartum haemorrhage, not menorrhagia (2026-09-30).
    unless: tuple[str, ...] = ()
    _compiled: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def label(self) -> str:
        """For screens: the rule name as words ("exertional muscle injury")."""
        return self.name.replace("_", " ")


def _rx(patterns: tuple[str, ...]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


# ---------------------------------------------------------------------------
# The rules. Titles must match `cpg_title` in the index exactly; validate_titles()
# is called at engine start-up so a filename rename can never silently break the
# safety net.
# ---------------------------------------------------------------------------
# A pregnant or recently delivered patient, as the intake says it. The intake's
# matching prose never carries "pregnan" for an unestablished or excluded status
# (schemas._MATCHING_PROSE), so this cannot fire on "pregnancy excluded".
_OBSTETRIC = (r"\bpregnan", r"\b\d{1,2}\s*(?:weeks?|/52)\s*(?:pregnant|gestation|of gestation)",
              r"\bgestation", r"\bantenatal", r"post-?partum", r"after (?:the )?(?:delivery|delivering|giving birth|childbirth)",
              r"\bgave birth", r"\bdeliver(?:ed|ing) (?:her |a |the )?(?:baby|twins|child)", r"\bjust delivered",
              r"following (?:delivery|childbirth|caesarean)", r"\bpuerper", r"\bin labou?r\b")

RED_FLAGS: tuple[RedFlag, ...] = (
    RedFlag(
        name="haemoptysis_or_chronic_cough",
        triggers=(
            r"\bhaemoptysis\b", r"\bhemoptysis\b", r"cough(?:ing)?\s+(?:up\s+)?blood",
            r"blood[- ]stained sputum", r"night sweats?",
            r"cough(?:ing)?[^.]{0,40}\b(?:[3-9]|[1-9]\d)\s*(?:weeks?|months?)",
            r"chronic cough",
        ),
        titles=("Management of Tuberculosis",),
        note="TB is endemic in Malaysia and drives an isolation decision at triage, "
             "not merely a treatment decision. Missing it exposes staff and other patients.",
    ),
    RedFlag(
        name="venous_thromboembolism",
        triggers=(
            r"\bcalf (?:swelling|pain|tenderness)\b", r"unilateral leg swelling",
            r"pleuritic", r"\bdvt\b", r"deep vein thrombosis", r"pulmonary embolism",
            # NOT r"\bpe\b": "PE" is physical examination as often as pulmonary
            # embolism in clinical notes; the spelled-out term above covers it.
            r"long[- ]haul", r"long flight", r"immobilis",
            r"recent surgery", r"bed[- ]bound",
        ),
        titles=("Prevention and Treatment of Venous Thromboembolism (VTE)",),
        note="PE is the can't-miss cause of chest pain and breathlessness. Symptom "
             "wording sits far from this CPG's prophylaxis/treatment vocabulary.",
    ),
    RedFlag(
        name="acute_stroke_syndrome",
        triggers=(
            r"\bhemiplegi", r"\bhemipares", r"facial droop", r"slurred speech",
            r"\bdysarthri", r"\bdysphasi", r"\baphasi", r"sudden(?:ly)? weak",
            r"one[- ]sided weakness", r"\bfast\s+positive", r"visual field loss",
        ),
        titles=(
            "Management of Ischaemic Stroke",
            "Management of Spontaneous Intracerebral Haemorrhage",
        ),
        note="Ischaemic and haemorrhagic stroke present identically and are "
             "opposite treatments. BOTH guidelines must be in front of the model - "
             "thrombolysis into an ICH is catastrophic.",
    ),
    RedFlag(
        name="thunderclap_or_anticoagulated_headache",
        triggers=(
            r"thunderclap", r"worst headache", r"sudden(?:ly)? severe headache",
            r"\bwarfarin\b", r"\bdoac\b", r"anticoagulat",
        ),
        require=(
            r"\bheadache\b", r"\bcollapse\b", r"reduced consciousness", r"\bgcs\b",
            r"\bvomit", r"neuro", r"weak", r"\bfit\b", r"\bseizure",
        ),
        titles=("Management of Spontaneous Intracerebral Haemorrhage",),
        note="Anticoagulation plus any neurological finding is intracranial "
             "haemorrhage until proven otherwise.",
    ),
    RedFlag(
        name="acute_coronary_syndrome",
        triggers=(
            r"chest pain", r"chest discomfort", r"chest tightness",
            r"crushing", r"radiat\w* to (?:the )?(?:jaw|arm|shoulder|back)",
            r"\bdiaphore", r"cold sweat",
        ),
        titles=(
            "Management of Acute Coronary Syndromes",
            "Management of Non ST Elevation Myocardial Infarction (NSTE ACS)",
        ),
        note="STEMI and NSTE-ACS diverge on timing and risk score, and the "
             "distinction is not visible from the presenting complaint alone.",
    ),
    RedFlag(
        name="dengue_syndrome",
        triggers=(
            r"\bdengue\b", r"retro[- ]orbital", r"\bmyalgi", r"\barthralgi",
            r"\bpetechia", r"platelet", r"\bthrombocytopeni", r"tourniquet test",
            r"warning signs?", r"haematocrit", r"hematocrit",
        ),
        require=(r"\bfever\b", r"febrile", r"\btemperature\b", r"\bdengue\b", r"day \d+ of illness"),
        titles=(
            "Management of Dengue Infection in Adults",
            "Management of Dengue Fever in Children",
        ),
        note="Highest-volume undifferentiated febrile presentation in Malaysia. "
             "Both adult and paediatric CPGs are offered; cohort is decided downstream.",
    ),
    RedFlag(
        name="head_injury",
        triggers=(
            r"head injury", r"head trauma", r"\bgcs\b", r"loss of consciousness",
            r"\blocs?\b", r"unequal pupils", r"\banisocori", r"skull",
            r"\brta\b", r"road traffic", r"motorcycle", r"motorbike", r"fell from",
        ),
        titles=("Early Management of Head Injury in Adults",),
        note="Malaysia's road-traffic burden makes this the highest-frequency "
             "major trauma presentation; GCS drives triage acuity directly.",
    ),
    RedFlag(
        name="abdominal_trauma",
        triggers=(
            r"abdominal trauma", r"blunt abdomin", r"penetrating", r"seat ?belt sign",
            r"stab wound", r"\bevisceration\b", r"abdominal distension",
        ),
        titles=("Management of Abdominal Trauma in Adults",),
        note="Haemodynamic instability with abdominal injury is time-critical.",
    ),
    RedFlag(
        name="hyperglycaemic_emergency",
        triggers=(
            r"\bketo", r"\bdka\b", r"kussmaul", r"polyuria", r"polydipsia",
            r"glucose\s*(?:of\s*)?(?:[2-9]\d|\d{3})", r"\bhhs\b", r"blood sugar high",
        ),
        titles=(
            "Management of Type 2 Diabetes Mellitus",
            "Management of Type 1 Diabetes Mellitus in Children and Adolescents",
        ),
        note="DKA presents identically in undiagnosed T1DM and decompensated T2DM.",
    ),
    RedFlag(
        name="acute_heart_failure",
        triggers=(
            r"orthopn", r"\bpnd\b", r"paroxysmal nocturnal", r"bibasal",
            r"\bcrepitation", r"\bcrackles\b", r"pedal o?edema", r"leg swelling",
            r"raised jvp", r"\bjvp\b", r"frothy sputum", r"pink froth",
        ),
        titles=("Management of Heart Failure",),
        note="Acute decompensated heart failure is a top breathlessness cause and "
             "is treated oppositely to an asthma/COPD exacerbation.",
    ),
    RedFlag(
        name="obstructive_airway_disease",
        triggers=(
            r"\bwheez", r"\basthma\b", r"\bcopd\b", r"inhaler", r"\bnebuli",
            r"silent chest", r"peak flow", r"accessory muscle",
        ),
        titles=(
            "Management of Asthma in Adults",
            "Management of Chronic Obstructive Pulmonary Disease (COPD)",
        ),
        note="Asthma and COPD overlap clinically and diverge on oxygen targets.",
    ),
    RedFlag(
        name="fragility_hip_fracture",
        triggers=(
            r"shortened and externally rotated", r"externally rotated",
            r"\bhip (?:pain|fracture)\b", r"neck of femur", r"\bnof\b",
            r"cannot weight ?bear", r"unable to weight ?bear",
        ),
        require=(r"\bfell\b", r"\bfall\b", r"\bfallen\b", r"\bhip\b", r"\bfemur\b"),
        titles=("Management of Geriatric Hip Fracture",),
        note="Elderly fall with a shortened, externally rotated leg is a hip "
             "fracture until imaging says otherwise; delay worsens mortality.",
    ),
    RedFlag(
        name="neonatal_jaundice",
        triggers=(
            r"\bneonat", r"\bnewborn\b", r"day[- ]?\d+ of life", r"\bjaundice",
            r"\bkernicterus\b", r"\bbilirubin\b",
        ),
        require=(r"\bjaundice", r"\byellow", r"\bbilirubin\b", r"\bneonat", r"\bnewborn\b"),
        titles=("Management of Neonatal Jaundice",),
        note="Untreated neonatal hyperbilirubinaemia causes irreversible kernicterus.",
    ),
    RedFlag(
        name="self_harm_risk",
        triggers=(
            r"suicid", r"self[- ]harm", r"overdose", r"took .{0,20}tablets",
            r"wants? to die", r"kill (?:him|her|them)self", r"hopeless",
            # NOT r"\bod\b": in a Malaysian prescription "OD" is omni die
            # (once daily), so it would fire self-harm on routine dosing text.
        ),
        titles=("Management of Major Depressive Disorder",),
        note="Self-harm risk changes triage acuity regardless of physical findings.",
    ),
    RedFlag(
        name="thyroid_emergency",
        triggers=(
            r"thyrotoxic", r"thyroid storm", r"\bgoitre\b", r"heat intoleran",
            r"myx"r"oedema", r"\btsh\b", r"exophthalmos",
        ),
        titles=("Management of Thyroid Disorders",),
        note="Thyroid storm and myxoedema coma are reversible causes of shock and coma.",
    ),
    RedFlag(
        name="atrial_fibrillation",
        triggers=(
            r"palpitation", r"irregular(?:ly)? irregular", r"\batrial fibrillation\b",
            r"\bafib\b", r"\baf\b(?!\w)", r"fast heart rate",
        ),
        titles=("Management of Atrial Fibrillation",),
        note="AF is both a symptom cause and a stroke risk requiring anticoagulation.",
    ),
    RedFlag(
        name="diabetic_foot",
        triggers=(r"foot ulcer", r"diabetic foot", r"\bgangrene\b", r"toe (?:ulcer|black)"),
        titles=("Management of Diabetic Foot",),
        note="Common Malaysian presentation and a frequently missed sepsis source.",
    ),
    RedFlag(
        name="diabetes_in_pregnancy",
        triggers=(r"\bpregnan", r"\bgestation", r"\bantenatal\b", r"\bweeks pregnant\b"),
        require=(r"glucose", r"\bdiabet", r"\bsugar\b", r"\bogtt\b", r"\bgdm\b"),
        titles=("Management of Diabetes in Pregnancy",),
        note="Pregnancy changes both glycaemic targets and safe drug choices.",
    ),
    RedFlag(
        name="renal_impairment",
        triggers=(
            r"\bcreatinine\b", r"\begfr\b", r"reduced urine", r"\boliguri", r"\banuri",
            r"\bdialysis\b", r"chronic kidney", r"\bckd\b", r"hyperkalaem", r"hyperkalem",
        ),
        titles=("Management of Chronic Kidney Disease in Adults",),
        note="Renal function gates the dose of most of the formulary; "
             "hyperkalaemia is immediately life-threatening.",
    ),    # ------------------------------------------------------------------
    # Added 2026-09-30 with the four documents that closed the gaps the
    # rhabdomyolysis baseline exposed. No KKM guideline covers
    # rhabdomyolysis itself; the MOH heat-illness guideline is the closest
    # adult source (it lists CK, renal function and urine myoglobin), and the
    # hyperkalaemia consensus covers the early killer.
    RedFlag(
        name="rhabdomyolysis",
        triggers=(
            r"(?:dark|tea|cola|coke|brown)[- ]?(?:colou?red)?\s+urine", r"myoglobinuri",
            r"\brhabdo", r"crush injur", r"compartment syndrome",
        ),
        # The Dyslipidaemia CPG holds the only KKM definition of rhabdomyolysis
        # ("CK > 10X of ULN", p59) and the statin-myopathy guidance. No KKM
        # guideline is written for rhabdomyolysis itself; what it lacks is
        # stated as a gap (cards.silent_elements), never imported.
        titles=("MOH Clinical Guidelines on Management of Heat Related Illness at Health Clinic and Emergency and Trauma Department",
                "Malaysian Consensus on the Management of Acute and Persistent Hyperkalaemia",
                "Management of Dyslipidaemia"),
        note="Pigmented urine with muscle symptoms is myoglobinuria until proven otherwise; "
             "the risks are acute kidney injury and hyperkalaemia.",
    ),
    RedFlag(
        name="exertional_muscle_injury",
        triggers=(
            r"\bmyalgia", r"muscle (?:pain|ache|weakness|tender\w*|swelling)",
            r"(?:leg|thigh|calf|limb)s? (?:pain|weakness)", r"generali[sz]ed (?:muscle )?weakness",
        ),
        require=(
            r"marathon", r"\bexercis", r"workout", r"exertion", r"\btraining\b", r"\bdrill",
            r"\bheat\b", r"\bhot\b", r"statin", r"\bcrush", r"found (?:down|on the floor)",
            r"prolonged immobil",
        ),
        titles=("MOH Clinical Guidelines on Management of Heat Related Illness at Health Clinic and Emergency and Trauma Department",),
        note="Muscle pain or weakness after exertion, heat, a statin or a long lie is the "
             "setting for rhabdomyolysis.",
    ),
    RedFlag(
        name="heat_illness",
        triggers=(
            r"heat ?stroke", r"heat exhaustion", r"heat[- ]related",
            r"(?:marathon|race|running|run\b|exercis\w*|training|drill)[^.]{0,80}"
            r"(?:collaps\w*|confus\w*|unconscious|not making sense|faint\w*)",
            r"collaps\w*[^.]{0,80}(?:marathon|race|running|heat|hot weather|\bsun\b)",
            r"(?:hot|heat|\bsun\b)[^.]{0,40}(?:collaps\w*|confus\w*|faint\w*)",
        ),
        titles=("MOH Clinical Guidelines on Management of Heat Related Illness at Health Clinic and Emergency and Trauma Department",),
        note="Heat stroke is a cooling emergency: mortality falls from ~70% to near zero "
             "when cooling starts without delay (MOH 2016 s1.1).",
    ),
    RedFlag(
        name="snakebite",
        triggers=(
            r"snake ?bite", r"bitten by (?:a )?snake", r"\bsnake\b", r"envenom",
            r"\bviper\b", r"\bcobra\b", r"\bkrait\b",
        ),
        titles=("MOH Guideline Management of Snakebite", "Paediatric Protocols for Malaysian Hospitals"),
        note="Envenoming can progress to coagulopathy, paralysis or shock within hours; "
             "the decision is antivenom, not symptom relief.",
    ),
    RedFlag(
        name="hyperkalaemia",
        triggers=(
            r"hyperkal[ae]?emi", r"high potassium", r"potassium (?:of |is |was )?(?:[6-9](?:\.\d)?)\b",
            r"\bk\+?\s*(?:of\s*)?[6-9](?:\.\d)?\b", r"tall tented t", r"peaked t",
        ),
        titles=("Malaysian Consensus on the Management of Acute and Persistent Hyperkalaemia",),
        note="Hyperkalaemia kills by arrhythmia before it causes symptoms; ECG changes "
             "mean treatment now, not after the repeat level.",
    ),
    # ------------------------------------------------------------------
    # Added 2026-09-30 with the ten CPGs indexed from the AMM list.
    RedFlag(
        name="upper_gi_bleeding",
        triggers=(
            r"ha?ematemesis", r"vomit\w* (?:out )?(?:fresh )?blood", r"coffee[- ]ground",
            r"mela?ena", r"black,? (?:tarry )?stools?", r"tarry stools?", r"upper gi bleed",
        ),
        titles=("Management of Non Variceal Upper Gastrointestinal Bleeding",
                "Management of Acute Variceal Bleeding"),
        note="Upper GI bleeding needs haemodynamic risk scoring and a decision on early "
             "endoscopy; variceal and non-variceal bleeding are treated differently.",
    ),
    RedFlag(
        name="infective_endocarditis",
        triggers=(
            r"endocardit", r"new (?:heart )?murmur", r"prosthetic (?:heart )?valve",
            r"valve replacement", r"\bivdu\b", r"(?:injecting|intravenous) drug use",
        ),
        require=(r"fever", r"febrile", r"rigou?r", r"endocardit", r"\btemp"),
        titles=("Prevention, Diagnosis and Management of Infective Endocarditis",),
        note="Fever with a murmur, a prosthetic valve or injecting drug use is endocarditis "
             "until blood cultures say otherwise - cultures come before antibiotics.",
    ),
    RedFlag(
        name="bleeding_disorder",
        triggers=(
            r"ha?emophilia", r"factor (?:viii|ix|8|9)", r"\bitp\b",
            r"thrombocytopenic purpura", r"von willebrand",
        ),
        titles=("Management of Haemophilia", "Management of Immune Thrombocytopenic Purpura"),
        note="A known bleeding disorder changes what counts as a minor injury and needs "
             "factor replacement or platelet-directed treatment, not only observation.",
    ),
    RedFlag(
        name="heavy_vaginal_bleeding",
        triggers=(
            r"menorrhagia", r"heavy (?:menstrual|period|vaginal|pv) (?:bleed\w*|loss)",
            r"soaking (?:through )?pads", r"passing clots",
        ),
        titles=("Management of Menorrhagia",),
        note="Heavy bleeding needs haemoglobin, pregnancy status and haemodynamic "
             "assessment before it is treated as a gynaecological routine.",
        unless=_OBSTETRIC,
    ),
    # ---- Added 2026-09-30 with the MOH obstetric, psychiatric, dental and
    # andrology documents. Each rule's FIRST title is its own guideline.
    RedFlag(
        name="postpartum_haemorrhage",
        triggers=(r"\bpph\b", r"post-?partum (?:ha?emorrhag\w*|bleed\w*)",
                  r"bleed\w*|ha?emorrhag\w*|soaking|clots|blood loss"),
        require=(r"post-?partum", r"after (?:the )?(?:delivery|delivering|giving birth|childbirth)",
                 r"\bgave birth", r"\bdeliver(?:ed|ing) (?:her |a |the )?(?:baby|twins|child)", r"\bjust delivered",
                 r"following (?:delivery|childbirth|caesarean)", r"\bpuerper", r"\bpph\b"),
        titles=("MOH Quick Reference Guide Postpartum Haemorrhage (PPH)", "MOH Perinatal Care Manual"),
        note="Bleeding after delivery is postpartum haemorrhage until proven otherwise - "
             "resuscitation, uterotonics and the cause (tone, tissue, trauma, thrombin) together.",
    ),
    RedFlag(
        name="hypertensive_disorder_of_pregnancy",
        triggers=(r"eclampsi\w*", r"pre-?eclampsi\w*", r"\bhellp\b",
                  r"\bfit(?:s|ted|ting)?\b", r"seizure", r"convuls\w*", r"severe headache",
                  r"blurr\w* (?:of )?vision", r"visual disturb\w*", r"epigastric pain",
                  r"right upper quadrant pain"),
        require=_OBSTETRIC + (r"eclampsi", r"\bhellp\b"),
        titles=("MOH Training Manual Hypertensive Disorders in Pregnancy", "MOH Perinatal Care Manual"),
        note="A fit, severe headache or visual disturbance in pregnancy or after delivery is "
             "eclampsia or severe pre-eclampsia until proven otherwise: magnesium sulphate and BP control.",
    ),
    RedFlag(
        name="obstetric_emergency",
        triggers=(r"antepartum", r"(?:cord|umbilical cord) prolapse", r"prolapsed cord",
                  r"placenta (?:praevia|previa)", r"abruption", r"reduced fetal movement",
                  r"(?:waters?|membranes?) (?:broke|rupture)", r"\bcontractions\b",
                  r"bleed\w*|spotting|abdominal pain"),
        require=(r"\bpregnan", r"\b\d{1,2}\s*(?:weeks?|/52)\s*(?:pregnant|gestation|of gestation)",
                 r"\bgestation", r"\bantenatal", r"\bin labou?r\b", r"antepartum", r"cord prolapse",
                 r"prolapsed cord"),
        titles=("MOH Perinatal Care Manual",),
        note="Bleeding, pain or labour in pregnancy is obstetric until proven otherwise.",
    ),
    RedFlag(
        name="heart_disease_in_pregnancy",
        triggers=(r"breathless\w*", r"short(?:ness)? of breath", r"chest pain", r"palpitation\w*",
                  r"orthopn\w*", r"syncope|faint\w*", r"peripartum cardiomyopathy", r"mitral stenosis"),
        require=_OBSTETRIC,
        titles=("Heart Disease in Pregnancy",),
        note="Breathlessness or chest pain in pregnancy or the puerperium may be cardiac "
             "(peripartum cardiomyopathy, mitral stenosis) or pulmonary embolism.",
    ),
    RedFlag(
        name="acute_behavioural_disturbance",
        triggers=(r"agitat\w*", r"aggressi\w*", r"violent", r"combative", r"threaten\w*",
                  r"psychos[ie]s|psychotic", r"hallucinat\w*", r"hearing voices", r"delusion\w*",
                  r"paranoi\w*", r"schizophren\w*", r"restrain\w*"),
        titles=("Management of Schizophrenia", "Management of Bipolar Disorder"),
        note="Acute agitation needs an organic cause excluded (hypoglycaemia, hypoxia, "
             "intoxication, delirium) alongside de-escalation and, if needed, rapid tranquillisation.",
    ),
    RedFlag(
        name="mania_or_lithium_toxicity",
        triggers=(r"\bmani(?:a|c)\b", r"bipolar", r"lithium"),
        titles=("Management of Bipolar Disorder",),
        note="Lithium toxicity (tremor, ataxia, confusion, seizure) is a medical emergency; "
             "dehydration and NSAIDs raise the level.",
    ),
    RedFlag(
        name="dental_avulsion",
        triggers=(r"avuls\w*", r"(?:tooth|teeth|incisor)\w*\s+(?:was |were |got )?(?:knocked|came|fell) out",
                  r"knocked out (?:a |his |her |the |one |two )?(?:front )?(?:tooth|teeth)"),
        titles=("Management of Avulsed Permanent Anterior Teeth",),
        note="An avulsed permanent tooth is time-critical: replant or store it correctly at once.",
    ),
    RedFlag(
        name="jaw_fracture",
        triggers=(r"jaw (?:fracture|injur\w*|pain|swelling|deformity)", r"mandib\w*", r"condyl\w*",
                  r"(?:cannot|can't|unable to) (?:open|close) (?:the |his |her )?mouth",
                  r"teeth (?:do not|don't|no longer) (?:meet|fit)", r"malocclusion"),
        titles=("Management of Mandibular Condyle Fractures",),
        note="Jaw trauma can threaten the airway and hides cervical-spine and head injury.",
    ),
    RedFlag(
        name="seizure_or_status_epilepticus",
        triggers=(r"status epilepticus", r"seizure\w*", r"convuls\w*", r"\bfit(?:s|ted|ting)?\b", r"epilep\w*"),
        # A fit in pregnancy or after delivery is the eclampsia rule's.
        unless=_OBSTETRIC,
        titles=("MSN Consensus Guidelines on the Management of Epilepsy",),
        note="A seizure lasting over 5 minutes, or repeated without recovery, is status epilepticus: "
             "benzodiazepine first, then a second-line antiseizure drug; check glucose.",
    ),
    RedFlag(
        name="poisoning_or_overdose",
        triggers=(r"overdos\w*", r"poison\w*", r"organophosph\w*", r"pesticide", r"paraquat", r"weed ?killer",
                  r"took .{0,25}(?:tablets|pills)", r"swallow\w* .{0,25}(?:tablets|pills|chemical|kerosene|bleach)",
                  r"intoxicat\w*", r"methanol", r"antidote"),
        titles=("MOH Antidotes Quick Guide (Adult Dose)",),
        note="Identify the agent, time and amount; the antidote and its dose depend on all three.",
    ),
    RedFlag(
        name="priapism",
        triggers=(r"priapism", r"(?:prolonged|persistent|painful) erection", r"erection (?:lasting|for) (?:over |more than )?\d+"),
        titles=("Management of Erectile Dysfunction",),
        note="An erection lasting over four hours is a urological emergency.",
    ),
    RedFlag(
        name="sore_throat",
        triggers=(r"sore throat", r"tonsillit", r"pharyngit", r"quinsy", r"peritonsillar"),
        titles=("Management of Sore Throat", "National Antimicrobial Guideline 2024 (Primary Care Pathways)"),
        note="Most sore throats need no antibiotic; the CPG and NAG C3 say which do, and "
             "quinsy or airway compromise must not be missed.",
    ),
    RedFlag(
        name="acute_gout",
        triggers=(r"\bgout", r"podagra", r"uric acid", r"hot,? swollen (?:big toe|joint|knee|ankle)"),
        titles=("Management of Gout",),
        note="An acute hot joint is septic arthritis until excluded; gout is the "
             "commonest mimic.",
    ),
    RedFlag(
        name="foreign_body_ingestion_child",
        triggers=(
            r"swallow\w* (?:a |an |the )?(?:coin|battery|magnet\w*|toy|object|pin|button)",
            r"button battery", r"foreign body ingest", r"ingested (?:a |an )?(?:coin|battery|magnet)",
        ),
        titles=("Management of Foreign Body Ingestion in Children",),
        note="A button battery or multiple magnets is an emergency; most coins are not.",
    ),
    RedFlag(
        name="cancer_pain",
        triggers=(r"cancer pain", r"malignan\w*[^.]{0,40}pain", r"metasta\w*[^.]{0,40}pain",
                  r"pain[^.]{0,40}(?:cancer|malignan|metasta)"),
        titles=("Management of Cancer Pain",),
        note="Cancer pain follows its own ladder and opioid-conversion rules; "
             "undertreatment is the commonest failure.",
    ),
    RedFlag(
        name="sepsis_or_septic_shock",
        triggers=(r"septic shock", r"\bsepsis\b", r"\bseptic\b", r"\bqsofa\b", r"\bsirs\b"),
        # The NAG 2024 is the MOH source for the empirical antibiotic; the
        # population filter keeps the adult and paediatric sections apart.
        titles=("MSIC ICU Management Protocols",
                "National Antimicrobial Guideline 2024 (Adults)",
                "National Antimicrobial Guideline 2024 (Paediatrics)"),
        note="Sepsis is time-critical: fluids, cultures and antibiotics within the hour.",
    ),
)

_COMPILED = [(f, _rx(f.triggers), _rx(f.require), _rx(f.unless)) for f in RED_FLAGS]


def match(text: str) -> list[tuple[RedFlag, str]]:
    """Return the rules the intake fires, each with the phrase that fired it."""
    if not text:
        return []
    fired: list[tuple[RedFlag, str]] = []
    for flag, triggers, require, unless in _COMPILED:
        hit = next((m.group(0) for p in triggers if (m := p.search(text))), None)
        if not hit:
            continue
        if require and not any(p.search(text) for p in require):
            continue
        if unless and any(p.search(text) for p in unless):
            continue
        fired.append((flag, hit))
    return fired


def titles_for(text: str) -> dict[str, list[str]]:
    """cpg_title -> the reasons it was forced in. Order is rule order, stable."""
    out: dict[str, list[str]] = {}
    for flag, hit in match(text):
        for title in flag.titles:
            out.setdefault(title, []).append(f"{flag.name} ({hit!r})")
    return out


def all_titles() -> set[str]:
    return {t for f in RED_FLAGS for t in f.titles}


def validate_titles(known: set[str]) -> list[str]:
    """Every title a rule points at must exist in the index, or the rule is dead
    weight that looks like a safety net. Called at engine start-up."""
    return sorted(all_titles() - known)
