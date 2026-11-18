import { describe, expect, it, vi } from 'vitest'
import { ApiClient, ApiError, parsePrediction } from '../src/api/client'
import { prediction, models, categories, info } from './fixtures'

describe('recorded backend contract', () => {
  it('accepts raw distances above one without probability conversion', () => { expect(parsePrediction(prediction).anomaly_score).toBe(prediction.anomaly_score); expect(prediction.anomaly_score).toBeGreaterThan(1) })
  it.each([ { ...prediction, threshold: undefined }, { ...prediction, anomaly_score: NaN }, { ...prediction, is_anomaly: !prediction.is_anomaly }, { ...prediction, schema_version: '9' }, { ...prediction, visualizations: { ...prediction.visualizations, heatmap_url: 'file:///private.png' } } ])('rejects malformed or unsafe responses', value => { expect(() => parsePrediction(value)).toThrow(ApiError) })
  it('uses the real multipart field names and never sets a boundary header manually', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(prediction)))
    vi.stubGlobal('fetch', fetcher)
    await new ApiClient().predict(new File(['png'], 'part.png', { type: 'image/png' }), 'patchcore', 'bottle')
    const [path, request] = fetcher.mock.calls[0]
    expect(path).toBe('/api/v1/predict'); expect(request.method).toBe('POST')
    expect(request.headers).toBeUndefined(); expect(request.body.get('model')).toBe('patchcore')
    expect(request.body.get('category')).toBe('bottle'); expect(request.body.get('visualize')).toBe('true')
    expect(request.body.get('image').name).toBe('part.png')
  })
  it('validates discovery and metadata', async () => {
    vi.stubGlobal('fetch', vi.fn((url: string) => Promise.resolve(new Response(JSON.stringify(url.includes('models') ? { models } : url.includes('categories') ? { categories } : info)))))
    const c = new ApiClient(); expect(await c.models()).toHaveLength(1); expect(await c.categories()).toHaveLength(15); expect((await c.info('patchcore', 'bottle')).seed).toBe(2026)
  })
  it('rejects unexpected category results', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...prediction, category: 'carpet' }))))
    await expect(new ApiClient().predict(new File(['png'], 'x.png'), 'patchcore', 'bottle')).rejects.toThrow('invalid response')
  })
  it('sanitizes backend errors rather than showing paths or traces', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { code: 'model_unavailable', message: 'private internal path' } }), { status: 503 })))
    await expect(new ApiClient().ready()).rejects.toThrow('Model artifacts')
  })
  it('handles network failure', async () => { vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('private host'))); await expect(new ApiClient().live()).rejects.toThrow('Cannot reach') })
  it('cancels a pending request', async () => {
    vi.stubGlobal('fetch', vi.fn((_url, init) => new Promise((_resolve, reject) => init.signal.addEventListener('abort', () => reject(new DOMException('Cancelled', 'AbortError'))))))
    const controller = new AbortController(); const pending = new ApiClient().ready(controller.signal); controller.abort()
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' })
  })
  it('handles request deadlines', async () => {
    vi.stubGlobal('fetch', vi.fn((_url, init) => new Promise((_resolve, reject) => init.signal.addEventListener('abort', () => reject(new DOMException('Timeout', 'AbortError'))))))
    await expect(new ApiClient('', 5).ready()).rejects.toMatchObject({ code: 'timeout' })
  })
})
