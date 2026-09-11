/** Labels and option order for the optional triage modifiers.
 *
 * Mirrors the enums in backend/app/schemas.py, which take their MTS-derived
 * values from backend/app/mts_table.py. Keep all three in step: the backend
 * test tests/test_triage_modifiers.py guards the two Python sides, and the
 * value strings below are what the API validates against.
 *
 * The MTS level shown in a hint is the level that CELL carries in MTS 2022,
 * not the level the patient will get - the floor takes the most acute match
 * across every field. It is shown so a triage officer can see why the option
 * exists and check it against the appendix.
 */
import type {
  ArrivalMode, BehaviouralRisk, Bleeding, Breathing, Comorbidity, EcgFinding,
  ExposureRisk, Facility, FeverHistory, GeneralAppearance, PaediatricSign,
  Perfusion, PregnancyStatus,
} from "@/lib/types";

export type Option<T> = { value: T; label: string; hint?: string };

/** Page 4 CRITICAL FIRST LOOK + page 7's two global-impression cells. */
export const APPEARANCE: Option<GeneralAppearance>[] = [
  { value: "walking_talking_not_distressed", label: "Walking, talking, not distressed" },
  { value: "appears_unwell", label: "Appears unwell", hint: "p7 L3" },
  { value: "cannot_sit_or_stand_unsupported", label: "Cannot sit or stand unsupported", hint: "p4 L3" },
  { value: "not_responding_to_call", label: "Not responding to call", hint: "p4 L2" },
  { value: "appears_septic_or_critically_ill", label: "Appears septic, toxic or critically ill", hint: "p7 L2" },
];

/** Page 4 RAPID ASSESSMENT / RESPIRATORY DISTRESS - the speech-effort ladder.
 *  This is what SpO2 alone misses: a compensating asthmatic holds a normal
 *  saturation while speaking in short phrases. */
export const BREATHING: Option<Breathing>[] = [
  { value: "not_breathless", label: "Not breathless, no oxygen needed" },
  { value: "wheeze_expiratory_rhonchi_airway_intact", label: "Wheeze or expiratory rhonchi, airway intact", hint: "p4 L3" },
  { value: "needs_oxygen_support", label: "Needs oxygen support", hint: "p4 L3" },
  { value: "difficulty_breathing_short_phrases_only", label: "Difficulty breathing, short phrases only", hint: "p4 L2" },
  { value: "abnormal_airway_sounds", label: "Abnormal airway sounds", hint: "p4 L1" },
  { value: "excessive_work_of_breathing_sweating", label: "Excessive work of breathing, sweating", hint: "p4 L1" },
  { value: "cannot_speak_one_word_reply", label: "Cannot speak, one-word replies", hint: "p4 L1" },
  { value: "requires_assisted_breathing", label: "Requires assisted breathing", hint: "p4 L1" },
];

/** Page 4 SHOCK STATE and page 13 CIRCULATION. The early sign of paediatric
 *  shock: a child holds its blood pressure until it does not. */
export const PERFUSION: Option<Perfusion>[] = [
  { value: "warm_pink_pulses_normal", label: "Warm and pink, pulses normal, CRT normal" },
  { value: "crt_over_2_seconds", label: "Capillary refill over 2 seconds", hint: "p4 L2" },
  { value: "tachycardia_weak_pulses", label: "Weak pulses with tachycardia", hint: "p4 L2" },
  { value: "pale_cyanosed_cold_peripheries", label: "Pale, cyanosed or cold peripheries", hint: "p4 L1" },
  { value: "absent_radial_pulse", label: "Absent radial pulse", hint: "p4 L1" },
];

/** Page 5 BLEEDING - a whole printed row that had no field until now. */
export const BLEEDING: Option<Bleeding>[] = [
  { value: "minimal_or_no_active_bleeding", label: "Minimal or no active bleeding" },
  { value: "bleeding_from_fracture_joint_wound_ent_or_menorrhagia", label: "From a wound, fracture, joint, ENT site, or menorrhagia", hint: "p5 L3" },
  { value: "expanding_haematoma_or_bleeding_disorder", label: "Expanding haematoma, or a known bleeding disorder", hint: "p5 L3" },
  { value: "active_vomiting_or_coughing_blood", label: "Vomiting or coughing blood", hint: "p5 L2" },
  { value: "suspected_internal_bleeding_ectopic_or_aaa", label: "Suspected internal — intra-abdominal, ectopic, AAA, vascular", hint: "p5 L2" },
  { value: "arterial_uncontrolled_or_massive_bleeding", label: "Arterial, uncontrolled, or massive", hint: "p5 L1" },
];

/** Page 7 INITIAL TESTS. Leave every box clear if no ECG was taken —
 *  "Normal ECG" is a positive finding you choose, not the default. */
export const ECG: Option<EcgFinding>[] = [
  { value: "normal_ecg", label: "Normal ECG", hint: "p7 L5" },
  { value: "no_st_t_wave_changes", label: "No ST-T wave changes", hint: "p7 L5" },
  { value: "no_ecg_findings_continuing_chest_pain", label: "No findings, chest pain continuing", hint: "p7 L4" },
  { value: "atrial_fibrillation_over_100", label: "AF > 100", hint: "p7 L3" },
  { value: "frequent_ectopics", label: "Frequent ectopics", hint: "p7 L3" },
  { value: "blocks_or_sinus_pauses", label: "Block or sinus pause", hint: "p7 L3" },
  { value: "tall_tented_t_waves", label: "Tall tented T waves", hint: "p7 L3" },
  { value: "st_elevations_or_depressions", label: "ST elevation or depression", hint: "p7 L2" },
  { value: "wide_complex_tachycardia", label: "Wide complex tachycardia", hint: "p7 L2" },
  { value: "narrow_complex_tachycardia_over_150", label: "Narrow complex tachycardia > 150", hint: "p7 L2" },
];

/** Page 7's Temp row. "Fever reported" only bites when nothing is measured
 *  now — a documented fever is already Level 3 or 2 on the temperature. */
export const FEVER_HISTORY: Option<FeverHistory>[] = [
  { value: "fever_reported_before_arrival", label: "Fever reported before arrival", hint: "p7 L4" },
  { value: "no_fever_reported", label: "No fever reported", hint: "p7 L5" },
];

/** Only "Immunocompromised" is an MTS cell. The rest change no triage level —
 *  they let the drug guardrails fire on a recorded fact instead of on whether
 *  the word happened to be typed into the history box. */
export const COMORBIDITIES: Option<Comorbidity>[] = [
  { value: "immunocompromised", label: "Immunocompromised", hint: "p7 L2" },
  { value: "on_anticoagulant", label: "On anticoagulant" },
  { value: "chronic_kidney_disease_or_dialysis", label: "CKD or dialysis" },
  { value: "copd_or_asthma", label: "COPD or asthma" },
  { value: "diabetes", label: "Diabetes" },
  { value: "liver_disease", label: "Liver disease" },
];

/** Page 13, the Paediatric Assessment Triangle. Shown only under 12. */
export const PAEDIATRIC_SIGNS: Option<PaediatricSign>[] = [
  { value: "stridor_grunting_or_wheezing", label: "Stridor, grunting or wheezing", hint: "p13 L1" },
  { value: "snoring_muffled_or_hoarse_speech", label: "Snoring, muffled or hoarse speech", hint: "p13 L1" },
  { value: "sniffing_or_tripod_position", label: "Sniffing or tripod position", hint: "p13 L1" },
  { value: "head_bobbing", label: "Head bobbing", hint: "p13 L1" },
  { value: "patchy_or_bluish_skin_discolouration", label: "Mottled or bluish skin", hint: "p13 L1" },
  { value: "supraclavicular_intercostal_or_substernal_retractions", label: "Retractions", hint: "p13 L2" },
  { value: "nasal_flaring_or_accessory_muscles", label: "Nasal flaring or accessory muscles", hint: "p13 L2" },
  { value: "drooling", label: "Drooling", hint: "p13 L2" },
  { value: "difficulty_in_swallowing", label: "Difficulty swallowing", hint: "p13 L2" },
  { value: "unable_to_walk_or_refusal_to_lie_down", label: "Unable to walk, or refusing to lie down", hint: "p13 L2" },
  { value: "pale_mucous_membranes_sole_or_palm", label: "Pale mucous membranes, soles or palms", hint: "p13 L2" },
];

/** Page 6. Changes no level. */
export const ARRIVAL: Option<ArrivalMode>[] = [
  { value: "walk_in", label: "Walk-in" },
  { value: "ambulance", label: "Ambulance" },
  { value: "referred_from_clinic", label: "Referred from a clinic or hospital" },
  { value: "police_okt", label: "Police (OKT)" },
  { value: "oscc", label: "One-Stop Crisis Centre" },
];

/** Page 5. The printed output column is "TO BE PLACED AT" — this drives
 *  isolation and decontamination, not the triage level. */
export const EXPOSURE: Option<ExposureRisk>[] = [
  { value: "suspected_tb", label: "Suspected active TB" },
  { value: "febrile_respiratory_illness", label: "Febrile respiratory illness" },
  { value: "outbreak_or_travel_contact", label: "Outbreak contact or recent travel" },
  { value: "known_mdro", label: "Known MDRO (CRE, MRSA)" },
  { value: "chemical_or_hazmat", label: "Chemical or HAZMAT exposure" },
];

/** Page 6, Code GREY. */
export const BEHAVIOURAL: Option<BehaviouralRisk>[] = [
  { value: "agitated_or_aggressive", label: "Agitated or aggressive" },
  { value: "weapon_or_police_escort", label: "Weapon involved or police escort" },
];

export const PREGNANCY: Option<PregnancyStatus>[] = [
  { value: "unknown", label: "Not established" },
  { value: "not_pregnant", label: "Not pregnant" },
  { value: "pregnant", label: "Pregnant" },
];

/** Set once and remembered. Changes the recommendation, never the level. */
export const FACILITY: Option<Facility>[] = [
  { value: "klinik_kesihatan", label: "Klinik Kesihatan" },
  { value: "district_hospital_no_specialist", label: "District hospital, no specialist" },
  { value: "district_hospital_with_specialist", label: "District hospital with specialist" },
  { value: "state_or_tertiary_with_pci_and_ct", label: "State / tertiary, PCI + CT on site" },
];

/** Every option in one lookup, for rendering a saved intake in the report. */
export const LABELS: Record<string, string> = Object.fromEntries(
  [
    ...APPEARANCE, ...BREATHING, ...PERFUSION, ...BLEEDING, ...ECG,
    ...FEVER_HISTORY, ...COMORBIDITIES, ...PAEDIATRIC_SIGNS, ...ARRIVAL,
    ...EXPOSURE, ...BEHAVIOURAL, ...PREGNANCY, ...FACILITY,
  ].map((o) => [o.value, o.label]),
);

export const labelFor = (value: string | null | undefined): string =>
  (value && LABELS[value]) || "";
