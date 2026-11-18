// Deterministic contract fixtures derived from the real recorded backend response.
// Modified scores below are mocked UI cases, never research results.
import recorded from '../../reports/backend/example_response.json'
import manifest from '../../configs/patchcore_deployment_v1.json'
import type { ModelInfo, Prediction } from '../src/api/types'
export const prediction = { ...recorded, visualizations: {
  heatmap_url: `/api/v1/predictions/${recorded.prediction_id}/heatmap`,
  overlay_url: `/api/v1/predictions/${recorded.prediction_id}/overlay`, expires_in_seconds: 300,
  scaling: { method: 'per-image min/max display only', minimum: 0, maximum: 2, alpha: .45 },
} } as Prediction
export const models = [{ id: 'patchcore', version: manifest.release, family: 'Simplified PatchCore', artifact_policy: 'Verified on load' }]
export const categories = Object.keys(manifest.categories)
export const info: ModelInfo = {
  model: 'patchcore', version: manifest.release, category: 'bottle', seed: 2026,
  image_threshold: manifest.categories.bottle.thresholds.image, pixel_threshold: manifest.categories.bottle.thresholds.pixel,
  calibration_normal_count: manifest.categories.bottle.calibration_normal_count,
  manifest_sha256: prediction.provenance.manifest_sha256, bank_sha256: prediction.provenance.bank_sha256,
  extractor_sha256: prediction.provenance.extractor_sha256, preprocessing: manifest.preprocessing,
  software: manifest.software, research_notice: prediction.limitations[0], device: 'cpu', loaded: false,
}
