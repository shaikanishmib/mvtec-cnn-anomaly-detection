// Maintained against reports/backend/openapi.json; checked by schema validation.
export interface Model { id: string; version: string; family: string; artifact_policy: string }
export interface ModelInfo {
  model: string; version: string; category: string; seed: number
  image_threshold: number; pixel_threshold: number; calibration_normal_count: number
  manifest_sha256: string; bank_sha256: string; extractor_sha256: string
  preprocessing: Record<string, unknown>; software: Record<string, string>
  research_notice: string; device: string; loaded: boolean
}
export interface Provenance {
  manifest_sha256: string; bank_sha256: string; extractor_sha256: string
  pretrained_weight_sha256: string; calibration_sha256: string; preprocessing_sha256: string; seed: 2026
}
export interface Prediction {
  schema_version: '1.0'; prediction_id: string; request_id: string; model: 'patchcore'; model_version: 'patchcore-inference-v1'
  category: string; anomaly_score: number; threshold: number; is_anomaly: boolean; pixel_threshold: number
  decision_rule: 'score > threshold'; score_semantics: string; input_width: number; input_height: number
  device: string; latency_ms: number; timings_ms: Record<string, number>; provenance: Provenance
  visualizations: { heatmap_url: string | null; overlay_url: string | null; expires_in_seconds: number | null; scaling: Record<string, unknown> | null }
  limitations: string[]
}
export const PREDICTION_FIELDS = ['schema_version', 'prediction_id', 'request_id', 'model', 'model_version', 'category', 'anomaly_score', 'threshold', 'is_anomaly', 'pixel_threshold', 'decision_rule', 'score_semantics', 'input_width', 'input_height', 'device', 'latency_ms', 'timings_ms', 'provenance', 'visualizations', 'limitations'] as const
