// Mirrors backend/app/schemas.py. Keep the two in step.

export type TriageColour = "RED" | "YELLOW" | "GREEN";
export type Confidence = "HIGH" | "MODERATE" | "LOW";

export type TriageLabel =
  | "RESUSCITATION"
  | "EMERGENCY"
  | "URGENT"
  | "EARLY_CARE"
  | "ROUTINE";

export type Disposition =
  | "RESUSCITATION_BAY"
  | "ADMIT_ICU_HDU"
  | "ADMIT_WARD"
  | "ED_OBSERVATION"
  | "REFER_SPECIALIST"
  | "DISCHARGE_WITH_FOLLOW_UP";

export type InquiryScope =
  | "all"
  | "cpg_only"
  | "quick_reference_only"
  | "paediatric_only"
  | "triage_only"
  | "formulary_only";

export interface Vitals {
  systolic_bp?: number | null;
  diastolic_bp?: number | null;
  heart_rate?: number | null;
  respiratory_rate?: number | null;
  temperature?: number | null;
  spo2?: number | null;
  gcs?: number | null;
  capillary_blood_glucose?: number | null;
  pain_score?: number | null;
}

// ---- optional triage modifiers. Mirrors the enums in schemas.py, which
// take their MTS-derived values from mts_table.py so the printed grid stays
// the single source. Every one is optional: unset means NOT ASSESSED, never
// "normal".

export type GeneralAppearance =
  | "walking_talking_not_distressed"
  | "appears_unwell"
  | "cannot_sit_or_stand_unsupported"
  | "not_responding_to_call"
  | "appears_septic_or_critically_ill";

export type Breathing =
  | "not_breathless"
  | "wheeze_expiratory_rhonchi_airway_intact"
  | "needs_oxygen_support"
  | "difficulty_breathing_short_phrases_only"
  | "abnormal_airway_sounds"
  | "excessive_work_of_breathing_sweating"
  | "cannot_speak_one_word_reply"
  | "requires_assisted_breathing";

export type Perfusion =
  | "warm_pink_pulses_normal"
  | "crt_over_2_seconds"
  | "tachycardia_weak_pulses"
  | "pale_cyanosed_cold_peripheries"
  | "absent_radial_pulse";

export type Bleeding =
  | "minimal_or_no_active_bleeding"
  | "bleeding_from_fracture_joint_wound_ent_or_menorrhagia"
  | "expanding_haematoma_or_bleeding_disorder"
  | "active_vomiting_or_coughing_blood"
  | "suspected_internal_bleeding_ectopic_or_aaa"
  | "arterial_uncontrolled_or_massive_bleeding";

export type EcgFinding =
  | "normal_ecg"
  | "no_st_t_wave_changes"
  | "no_ecg_findings_continuing_chest_pain"
  | "atrial_fibrillation_over_100"
  | "frequent_ectopics"
  | "blocks_or_sinus_pauses"
  | "tall_tented_t_waves"
  | "st_elevations_or_depressions"
  | "wide_complex_tachycardia"
  | "narrow_complex_tachycardia_over_150";

export type FeverHistory = "fever_reported_before_arrival" | "no_fever_reported";

export type Comorbidity =
  | "immunocompromised"
  | "on_anticoagulant"
  | "chronic_kidney_disease_or_dialysis"
  | "copd_or_asthma"
  | "diabetes"
  | "liver_disease";

export type PaediatricSign =
  | "snoring_muffled_or_hoarse_speech"
  | "stridor_grunting_or_wheezing"
  | "sniffing_or_tripod_position"
  | "head_bobbing"
  | "patchy_or_bluish_skin_discolouration"
  | "difficulty_in_swallowing"
  | "drooling"
  | "unable_to_walk_or_refusal_to_lie_down"
  | "supraclavicular_intercostal_or_substernal_retractions"
  | "nasal_flaring_or_accessory_muscles"
  | "pale_mucous_membranes_sole_or_palm";

export type ArrivalMode =
  | "walk_in"
  | "ambulance"
  | "police_okt"
  | "referred_from_clinic"
  | "oscc";

export type ExposureRisk =
  | "suspected_tb"
  | "febrile_respiratory_illness"
  | "outbreak_or_travel_contact"
  | "known_mdro"
  | "chemical_or_hazmat";

export type BehaviouralRisk = "agitated_or_aggressive" | "weapon_or_police_escort";

/** Tri-state. `unknown` is the default and means the question was not put -
 *  it is NOT the same as `not_pregnant`, which is a recorded answer. */
export type PregnancyStatus = "pregnant" | "not_pregnant" | "unknown";

export type Facility =
  | "klinik_kesihatan"
  | "district_hospital_no_specialist"
  | "district_hospital_with_specialist"
  | "state_or_tertiary_with_pci_and_ct";

export interface TriageRequest {
  age: number;
  gender: "male" | "female" | "other" | "unknown";
  weight_kg?: number | null;
  vitals: Vitals;
  complaint: string;
  history: string;

  // Primary Triage (MTS pp4-5, p13) and Initial Tests (p7)
  appearance?: GeneralAppearance | null;
  breathing?: Breathing | null;
  perfusion?: Perfusion | null;
  bleeding?: Bleeding | null;
  fever_history?: FeverHistory | null;
  ecg_findings?: EcgFinding[];
  paediatric_signs?: PaediatricSign[];

  // timing and context
  onset?: string;
  trauma_mechanism?: string;
  arrival_mode?: ArrivalMode | null;

  // background the prescribing checks read
  allergies?: string;
  current_medications?: string;
  comorbidities?: Comorbidity[];
  egfr?: number | null;

  // pregnancy
  pregnancy?: PregnancyStatus;
  gestation_weeks?: number | null;
  breastfeeding?: boolean;

  // safety / placement
  exposure_risk?: ExposureRisk[];
  behavioural_risk?: BehaviouralRisk[];

  // remembered setting, not a per-patient field
  facility?: Facility | null;
}

export interface Citation {
  source_id: string;
  document: string;
  page: string;
  edition_year: string;
}

export interface VitalInterpretation {
  parameter: string;
  value: string;
  interpretation: string;
  mts_level_triggered: string;
}

export interface RedFlag {
  flag: string;
  why_it_matters: string;
  source_id: string;
}

export interface DifferentialDiagnosis {
  condition: string;
  discriminating_feature: string;
}

export interface PrimaryDiagnosis {
  condition: string;
  confidence: Confidence;
  reasoning: string;
  supporting_cpg: string;
}

export interface ImmediateAction {
  sequence: number;
  action: string;
  timeframe: string;
  source_id: string;
}

export interface Investigation {
  test: string;
  rationale: string;
  urgency: string;
}

export interface DrugRecommendation {
  drug_name: string;
  indication: string;
  adult_dose: string;
  paediatric_dose: string;
  route: string;
  frequency: string;
  duration: string;
  prescriber_category: string;
  prescriber_category_meaning: string;
  cautions: string;
  source_id: string;
  /** Server-set. VERIFIED / EXCEEDS_MAXIMUM / DIFFERS_FROM_SOURCE / NOT_COMPARABLE. */
  dose_verdict?: string | null;
  dose_verdict_detail?: string;
  /** Authoritative dose text, quoted VERBATIM from FUKKM / the CPG. */
  dose_source_fukkm?: string;
  dose_source_cpg?: string;
  /** Server-set. True only when the drug is named in the excerpt cited for THIS
   *  patient. Presence elsewhere in the guideline is deliberately not enough. */
  indication_supported?: boolean | null;
}

/** Server-set. A required element of this presentation the answer omitted.
 *  Never generated content - only the statement that something is missing. */
export interface CompletenessGap {
  element: string;
  why: string;
  guideline: string;
}

/** Server-set conflict between a recommendation and this patient's own intake. */
export interface Contraindication {
  rule: string;
  severity: "ABSOLUTE" | "CAUTION";
  where: string;
  item: string;
  trigger: string;
  reason: string;
}

export interface DiagnosticSchema {
  mts_triage_level: 1 | 2 | 3 | 4 | 5;
  mts_triage_label: TriageLabel;
  time_to_treatment: string;
  /** Server-set from the MTS 2022 p3 re-triage rule. Empty at Levels 1-2. */
  reassessment: string;
  triage_rationale: string;
  vitals_interpretation: VitalInterpretation[];
  red_flags: RedFlag[];
  primary_diagnosis: PrimaryDiagnosis;
  differential_diagnoses: DifferentialDiagnosis[];
  immediate_actions: ImmediateAction[];
  investigations: Investigation[];
  drug_recommendations: DrugRecommendation[];
  prescriber_category_warning: string;
  drug_indication_warning?: string;
  dose_completeness_warning?: string;
  contraindication_warning?: string;
  contraindications?: Contraindication[];
  completeness_warning?: string;
  completeness_gaps?: CompletenessGap[];
  citation_warning?: string;
  /** Server-set. Urgency claimed that the triage level does not support. */
  urgency_warning?: string;
  /**
   * Server-set. The report sends this patient out of the ETD, but the JKN
   * Selangor Redirection Policy 2024 s4.2 names them as one who must be seen
   * there. Advisory - that policy is state-level, not national.
   */
  redirection_warning?: string;
  parse_warning?: string;
  disposition: Disposition;
  disposition_justification: string;
  referral_required: boolean;
  referral_to: string;
  evidence_gaps: string;
  citations: Citation[];
}

export interface RetrievedSource {
  source_id: string;
  filename: string;
  cpg_title: string;
  edition_year: string;
  doc_type: string;
  page_number?: number | null;
  drug_name?: string | null;
  prescriber_category?: string | null;
  score?: number | null;
  excerpt: string;
}

export interface TriageResponse {
  diagnostic: DiagnosticSchema;
  triage_colour: TriageColour;
  /** The intake echoed back by the API, so an exported report carries its
   *  own inputs and does not depend on client state surviving a reload. */
  request: TriageRequest;
  sources: RetrievedSource[];
  model: string;
  latency_ms: number;
  corpus_warnings: string[];
  disclaimer: string;
}

export interface InquiryRequest {
  query: string;
  scope: InquiryScope;
  max_sources: number;
}

export interface InquiryResponse {
  query: string;
  answer_markdown: string;
  sources: RetrievedSource[];
  documents_scanned: string[];
  model: string;
  latency_ms: number;
  corpus_warnings: string[];
}

export interface CorpusStats {
  collection: string;
  total_chunks: number;
  documents: Array<{
    filename: string;
    cpg_title: string;
    doc_type: string;
    edition_year: string;
    edition: string;
    chunks: number;
  }>;
  embedding_model: string;
  chunk_tokens: number;
  chunk_overlap_tokens: number;
  warnings: string[];
}

/** Live progress of a Mode A run. `percent` is exact through retrieval and
 *  prefill; during decode it is an estimate against a calibrated typical output
 *  length, because the model's output length is unknown in advance. */
export interface ProgressResponse {
  job_id: string;
  stage: "queued" | "retrieval" | "prefill" | "decode" | "checks" | "done" | "error" | "unknown";
  percent: number;
  detail: string;
  done: boolean;
  estimated: boolean;
  error: string;
  elapsed_s: number;
}
