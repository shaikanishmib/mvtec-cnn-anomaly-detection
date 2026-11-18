import { test, expect, type Page } from '@playwright/test'
import AxeBuilder from '@axe-core/playwright'
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { resolve } from 'node:path'

const root = resolve(import.meta.dirname, '../..')
const output = resolve(root, 'outputs/frontend')
mkdirSync(output, { recursive: true })
interface Input { category: string; path: string; name: string; case: string }
const inputs: Input[] = JSON.parse(readFileSync(resolve(output, 'e2e-inputs.json'), 'utf8'))
async function connect(page: Page) {
  await page.goto('/')
  await expect(page.getByTestId('connection')).toHaveText('Backend ready', { timeout: 90_000 })
  await expect(page.getByLabel('Detection model')).toHaveValue('patchcore')
  await expect(page.getByLabel('Inspection category').locator('option')).toHaveCount(15)
}
async function synthetic(page: Page) {
  const bytes = await page.evaluate(() => {
    const c = document.createElement('canvas'); c.width = 384; c.height = 384
    const ctx = c.getContext('2d')!
    ctx.fillStyle = '#ccd1ce'; ctx.fillRect(0, 0, 384, 384)
    ctx.fillStyle = '#8c9995'; ctx.fillRect(36, 36, 312, 312)
    ctx.fillStyle = '#bfc9c4'; ctx.fillRect(42, 42, 300, 300)
    for (let y = 48; y < 340; y += 6) { ctx.fillStyle = y % 12 === 0 ? '#b9c4be' : '#bdc8c1'; ctx.fillRect(43, y, 298, 1) }
    for (const [x, y] of [[65, 65], [319, 65], [65, 319], [319, 319]]) {
      ctx.fillStyle = '#697c73'; ctx.beginPath(); ctx.arc(x, y, 12, 0, Math.PI * 2); ctx.fill()
      ctx.strokeStyle = '#dee4dc'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x - 6, y); ctx.lineTo(x + 6, y); ctx.stroke()
    }
    ctx.strokeStyle = '#876546'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(174, 142); ctx.lineTo(227, 207); ctx.lineTo(210, 245); ctx.stroke()
    ctx.fillStyle = '#3f5749'; ctx.font = '11px sans-serif'; ctx.fillText('SYNTHETIC UI FIXTURE', 127, 286)
    return c.toDataURL('image/png').split(',')[1]
  })
  return Buffer.from(bytes, 'base64')
}

test('public desktop/mobile UI, authored fixture, keyboard, contrast and responsive layout', async ({ page }) => {
  await connect(page)
  const fixture = await synthetic(page)
  await page.getByLabel('Inspection image', { exact: true }).setInputFiles({ name: 'synthetic-ui-fixture.png', mimeType: 'image/png', buffer: fixture })
  await expect(page.getByAltText('Original selected inspection image')).toBeVisible()
  const response = page.waitForResponse(r => r.url().endsWith('/api/v1/predict') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Run inspection' }).click()
  const actual = await (await response).json()
  await expect(page.getByTestId('score')).toHaveText(actual.anomaly_score.toFixed(6))
  await expect(page.getByAltText(/anomaly heatmap/)).toBeVisible()
  await expect(page.getByAltText(/response overlay/)).toBeVisible()
  const accessibility = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
  expect(accessibility.violations).toEqual([])
  const publicDir = resolve(root, 'docs/images'); mkdirSync(publicDir, { recursive: true })
  await page.screenshot({ path: resolve(publicDir, 'inspection-dashboard-desktop.png'), fullPage: true })
  await page.getByRole('tab', { name: 'Compare all' }).focus(); await page.keyboard.press('ArrowRight')
  await expect(page.getByRole('tab', { name: 'Original' })).toHaveAttribute('aria-selected', 'true')
  await page.getByRole('tab', { name: 'Compare all' }).click()
  await page.getByLabel('Image zoom').selectOption('150'); await page.getByLabel('Image zoom').selectOption('100')
  await page.setViewportSize({ width: 390, height: 844 })
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await page.screenshot({ path: resolve(publicDir, 'inspection-dashboard-mobile.png'), fullPage: true })
  const mobile = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
  expect(mobile.violations).toEqual([])
  await page.getByRole('button', { name: 'Reset inspection' }).click()
  await expect(page.getByTestId('score')).toHaveCount(0)
  writeFileSync(resolve(output, 'public-capture.json'), JSON.stringify({ source: 'Original procedural canvas fixture; no MVTec pixels', input_sha256: createHash('sha256').update(fixture).digest('hex'), actual_backend_result: true, desktop_axe_violations: accessibility.violations.length, mobile_axe_violations: mobile.violations.length, mobile_horizontal_overflow: false }, null, 2))
})

for (const input of inputs) test(`real backend ${input.category}/${input.case}: displayed/API values, PNG fidelity and turnaround`, async ({ page }) => {
  await connect(page)
  const navigation = await page.evaluate(() => ({ navigation: performance.getEntriesByType('navigation').map(x => x.toJSON()), paint: performance.getEntriesByType('paint').map(x => x.toJSON()) }))
  await page.getByLabel('Inspection category').selectOption(input.category)
  await page.getByLabel('Inspection image', { exact: true }).setInputFiles({ name: input.name, mimeType: 'image/png', buffer: readFileSync(input.path) })
  await expect(page.getByAltText('Original selected inspection image')).toBeVisible()
  const heatResponse = page.waitForResponse(r => /\/predictions\/[^/]+\/heatmap$/.test(r.url()))
  const overlayResponse = page.waitForResponse(r => /\/predictions\/[^/]+\/overlay$/.test(r.url()))
  const predictionResponse = page.waitForResponse(r => r.url().endsWith('/api/v1/predict') && r.request().method() === 'POST')
  const started = performance.now()
  await page.getByRole('button', { name: 'Run inspection' }).click()
  await expect(page.getByRole('button', { name: /Inspecting/ })).toBeDisabled()
  const actual = await (await predictionResponse).json()
  expect(actual.category).toBe(input.category); expect(actual.provenance.seed).toBe(2026)
  await expect(page.getByTestId('score')).toHaveText(actual.anomaly_score.toFixed(6))
  await expect(page.getByTestId('threshold')).toHaveText(actual.threshold.toFixed(6))
  await expect(page.getByRole('heading', { name: actual.is_anomaly ? 'Research anomaly' : 'Research normal' })).toBeVisible()
  const heat = page.getByAltText(/anomaly heatmap/), overlay = page.getByAltText(/response overlay/)
  await expect(heat).toBeVisible(); await expect(overlay).toBeVisible()
  await expect.poll(() => heat.evaluate((img: HTMLImageElement) => img.naturalWidth)).toBe(actual.input_width)
  await expect.poll(() => overlay.evaluate((img: HTMLImageElement) => img.naturalHeight)).toBe(actual.input_height)
  const total = performance.now() - started
  const hashes: Record<string, string> = {}
  for (const [kind, response, locator] of [['heatmap', await heatResponse, heat], ['overlay', await overlayResponse, overlay]] as const) {
    expect(response.status()).toBe(200)
    const authoritative = await page.request.get(response.url())
    expect(authoritative.status()).toBe(200)
    const expected = createHash('sha256').update(await authoritative.body()).digest('hex')
    const displayed = await locator.evaluate(async (img: HTMLImageElement) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await (await fetch(img.src)).arrayBuffer()))).map(x => x.toString(16).padStart(2, '0')).join(''))
    expect(displayed).toBe(expected); hashes[kind] = expected
  }
  const metrics = await page.evaluate(() => performance.getEntriesByType('measure').filter(x => x.name.startsWith('inspection:') || x.name === 'app:initial-render').map(x => ({ name: x.name, duration_ms: x.duration })))
  const report = { category: input.category, case: input.case, model: actual.model, device: actual.device, score: actual.anomaly_score, threshold: actual.threshold, is_anomaly: actual.is_anomaly, backend_latency_ms: actual.latency_ms, backend_timings_ms: actual.timings_ms, total_browser_turnaround_ms: total, metrics, navigation, hashes, displayed_score: actual.anomaly_score.toFixed(6), displayed_threshold: actual.threshold.toFixed(6), png_bytes_unchanged: true }
  writeFileSync(resolve(output, `real-${input.category}-${input.case}.json`), JSON.stringify(report, null, 2))
  // Genuine dataset imagery stays only in ignored local outputs.
  await page.screenshot({ path: resolve(output, `real-${input.category}-${input.case}.png`), fullPage: true })
  await page.getByLabel('Inspection category').selectOption(input.category === 'bottle' ? 'carpet' : 'bottle')
  await expect(page.getByTestId('score')).toHaveCount(0)
})

test('expired PNGs are explained without changing the accepted prediction', async ({ page }) => {
  await connect(page)
  await page.route('**/api/v1/predictions/*/heatmap', route => route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ error: { code: 'visualization_not_found', message: 'expired', request_id: 'fixture' } }) }))
  await page.getByLabel('Inspection image', { exact: true }).setInputFiles({ name: 'synthetic.png', mimeType: 'image/png', buffer: await synthetic(page) })
  await page.getByRole('button', { name: 'Run inspection' }).click()
  await expect(page.getByText('Heatmap unavailable')).toBeVisible()
  await expect(page.getByTestId('score')).toBeVisible()
})
