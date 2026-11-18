import type { Model, ModelInfo, Prediction } from './types'

type ObjectValue = Record<string, unknown>
const object = (v: unknown): v is ObjectValue => typeof v === 'object' && v !== null && !Array.isArray(v)
const text = (v: unknown): v is string => typeof v === 'string' && v.length > 0 && v.length < 2000
const number = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const uuid = (v: unknown): v is string => typeof v === 'string' && /^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$/i.test(v)
const sha = (v: unknown): v is string => typeof v === 'string' && /^[a-f0-9]{64}$/i.test(v)
const messages: Record<string, string> = {
  server_busy: 'The inspection worker is busy. Wait a moment, then try again.',
  model_unavailable: 'Model artifacts or the processing device are unavailable. Check the backend configuration.',
  upload_timeout: 'The upload timed out. Try again with a smaller image.',
  request_too_large: 'The request exceeds the backend upload limit.', upload_too_large: 'The image exceeds the backend upload limit.',
  image_dimensions: 'The image exceeds the backend dimension or pixel limit.',
  visualization_not_found: 'This visualization expired, was evicted, or is unavailable. Run the inspection again.',
  unsupported_model: 'This model is not available on the backend.', unsupported_category: 'This category is not supported.',
  invalid_image: 'The backend could not safely decode this image.', empty_image: 'Choose a non-empty image.',
  format_mismatch: 'Image content does not match its file extension and MIME type.',
  unsupported_image_type: 'Choose a matching PNG or JPEG file.', invalid_parameters: 'The request parameters were rejected.',
}
export class ApiError extends Error {
  constructor(public code: string, public status = 0) {
    super(messages[code] || ({ invalid_response: 'The backend returned an invalid response. No result was accepted.', connection_error: 'Cannot reach the backend. Check that FastAPI is running.', timeout: 'The request timed out. Backend work may still be running; wait before retrying.', visualization_error: 'The visualization is unavailable. It may have expired or the server may have restarted.' }[code]) || 'The backend could not complete the inspection. Try again or check its status.')
  }
}
const invalid = (): never => { throw new ApiError('invalid_response') }
export function parsePrediction(value: unknown): Prediction {
  if (!object(value)) return invalid()
  if (value.schema_version !== '1.0' || value.model !== 'patchcore' || value.model_version !== 'patchcore-inference-v1' || !uuid(value.prediction_id) || !uuid(value.request_id) || !text(value.category) || !text(value.device)) return invalid()
  for (const key of ['anomaly_score', 'threshold', 'pixel_threshold', 'latency_ms']) if (!number(value[key])) return invalid()
  if (value.decision_rule !== 'score > threshold' || typeof value.is_anomaly !== 'boolean' || value.is_anomaly !== ((value.anomaly_score as number) > (value.threshold as number)) || !text(value.score_semantics)) return invalid()
  if (![value.input_width, value.input_height].every(v => number(v) && Number.isInteger(v) && v > 0) || (value.latency_ms as number) < 0) return invalid()
  if (!object(value.provenance) || value.provenance.seed !== 2026) return invalid()
  for (const key of ['manifest_sha256', 'bank_sha256', 'extractor_sha256', 'pretrained_weight_sha256', 'calibration_sha256', 'preprocessing_sha256']) if (!sha(value.provenance[key])) return invalid()
  if (!object(value.timings_ms) || !Object.values(value.timings_ms).every(number) || !Array.isArray(value.limitations) || !value.limitations.length || !value.limitations.every(text)) return invalid()
  const visuals = value.visualizations
  if (!object(visuals)) return invalid()
  for (const kind of ['heatmap', 'overlay']) {
    const url = visuals[`${kind}_url`]
    if (url !== null && url !== `/api/v1/predictions/${value.prediction_id}/${kind}`) return invalid()
  }
  if (visuals.expires_in_seconds !== null && (!number(visuals.expires_in_seconds) || visuals.expires_in_seconds <= 0)) return invalid()
  if (visuals.scaling !== null && !object(visuals.scaling)) return invalid()
  return value as unknown as Prediction
}

export class ApiClient {
  constructor(private base = import.meta.env.VITE_API_BASE_URL || '', private timeoutMs = 120_000) {}
  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    const controller = new AbortController(); let timedOut = false
    const abort = () => controller.abort()
    init.signal?.addEventListener('abort', abort, { once: true })
    if (init.signal?.aborted) controller.abort()
    const timer = setTimeout(() => { timedOut = true; controller.abort() }, this.timeoutMs)
    try {
      const response = await fetch(this.base + path, { ...init, signal: controller.signal })
      if (!response.ok) {
        let code = response.status === 404 ? 'visualization_not_found' : 'server_error'
        try { const body: unknown = await response.json(); if (object(body) && object(body.error) && typeof body.error.code === 'string') code = body.error.code } catch { /* Never expose raw error bodies. */ }
        throw new ApiError(code, response.status)
      }
      return response
    } catch (error) {
      if (timedOut) throw new ApiError('timeout')
      if (controller.signal.aborted) throw new DOMException('Cancelled', 'AbortError')
      if (error instanceof ApiError) throw error
      throw new ApiError('connection_error')
    } finally { clearTimeout(timer); init.signal?.removeEventListener('abort', abort) }
  }
  private async json(path: string, signal?: AbortSignal): Promise<unknown> {
    const r = await this.request(path, { signal })
    try { return await r.json() } catch { return invalid() }
  }
  async live(signal?: AbortSignal) { const v = await this.json('/health/live', signal); if (!object(v) || v.status !== 'live') invalid(); return true }
  async ready(signal?: AbortSignal) { const v = await this.json('/health/ready', signal); if (!object(v) || v.status !== 'ready' || !text(v.device)) invalid(); return true }
  async models(signal?: AbortSignal): Promise<Model[]> {
    const v = await this.json('/api/v1/models', signal)
    if (!object(v) || !Array.isArray(v.models) || !v.models.length || !v.models.every(m => object(m) && ['id', 'version', 'family', 'artifact_policy'].every(k => text(m[k])))) return invalid()
    return v.models as Model[]
  }
  async categories(signal?: AbortSignal): Promise<string[]> {
    const v = await this.json('/api/v1/categories', signal)
    if (!object(v) || !Array.isArray(v.categories) || !v.categories.length || !v.categories.every(text) || new Set(v.categories).size !== v.categories.length) return invalid()
    return v.categories as string[]
  }
  async info(model: string, category: string, signal?: AbortSignal): Promise<ModelInfo> {
    const v = await this.json(`/api/v1/model-info?${new URLSearchParams({ model, category })}`, signal)
    if (!object(v) || v.model !== model || v.category !== category || !text(v.version) || !text(v.device) || !Number.isInteger(v.seed) || !number(v.image_threshold) || !number(v.pixel_threshold) || !number(v.calibration_normal_count) || !sha(v.manifest_sha256) || !sha(v.bank_sha256) || !sha(v.extractor_sha256) || !object(v.preprocessing) || !object(v.software) || !text(v.research_notice) || typeof v.loaded !== 'boolean') return invalid()
    return v as unknown as ModelInfo
  }
  async predict(file: File, model: string, category: string, signal?: AbortSignal): Promise<Prediction> {
    const before = performance.now(); const form = new FormData()
    form.append('image', file); form.append('model', model); form.append('category', category); form.append('visualize', 'true')
    performance.measure('inspection:submission', { start: before, end: performance.now() })
    const r = await this.request('/api/v1/predict', { method: 'POST', body: form, signal })
    let value: unknown
    try { value = await r.json() } catch { return invalid() }
    const result = parsePrediction(value)
    if (result.model !== model || result.category !== category) return invalid()
    return result
  }
  async image(url: string, signal: AbortSignal): Promise<Blob> {
    if (!/^\/api\/v1\/predictions\/[a-f0-9-]{36}\/(heatmap|overlay)$/i.test(url)) return invalid()
    const r = await this.request(url, { signal })
    if (!r.headers.get('content-type')?.startsWith('image/png')) throw new ApiError('visualization_error')
    const blob = await r.blob()
    if (!blob.size) throw new ApiError('visualization_error')
    return blob
  }
}
export const api = new ApiClient()
