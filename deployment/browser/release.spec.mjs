import { test, expect } from '../../frontend/node_modules/@playwright/test/index.mjs'
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
const root=resolve(import.meta.dirname,'../..')
const inputs=JSON.parse(readFileSync(resolve(root,'outputs/deployment/e2e-inputs.json'),'utf8'))

for (const category of ['bottle','carpet']) test(`production assets, genuine ${category} inspection, reset and repeat`,async({page})=>{
  await page.goto('/')
  await expect(page.getByTestId('connection')).toHaveText('Backend ready',{timeout:90000})
  await expect(page.getByLabel('Detection model')).toHaveValue('patchcore')
  await expect(page.getByLabel('Inspection category').locator('option')).toHaveCount(15)
  await page.getByLabel('Inspection category').selectOption(category)
  const input=inputs.find(x=>x.category===category && x.case==='normal')
  await page.getByLabel('Inspection image',{exact:true}).setInputFiles(input.path)
  await expect(page.getByAltText('Original selected inspection image')).toBeVisible()
  const response=page.waitForResponse(r=>r.url().endsWith('/api/v1/predict') && r.request().method()==='POST')
  const start=performance.now()
  await page.getByRole('button',{name:'Run inspection'}).click()
  const prediction=await(await response).json()
  await expect(page.getByTestId('score')).toHaveText(prediction.anomaly_score.toFixed(6))
  await expect(page.getByTestId('threshold')).toHaveText(prediction.threshold.toFixed(6))
  await expect(page.getByRole('heading',{name:prediction.is_anomaly?'Research anomaly':'Research normal'})).toBeVisible()
  for(const name of [/anomaly heatmap/,/response overlay/]){
    const image=page.getByAltText(name);await expect(image).toBeVisible()
    await expect.poll(()=>image.evaluate(img=>img.naturalWidth)).toBe(prediction.input_width)
  }
  const total=performance.now()-start
  const measures=await page.evaluate(()=>({measures:performance.getEntriesByType('measure').map(x=>({name:x.name,duration_ms:x.duration})),paint:performance.getEntriesByType('paint').map(x=>({name:x.name,start_ms:x.startTime}))}))
  await expect(page.getByText('Evidence, not a verdict.')).toBeVisible()
  await page.getByRole('button',{name:'Reset inspection'}).click()
  await expect(page.getByTestId('score')).toHaveCount(0)
  await page.getByLabel('Inspection image',{exact:true}).setInputFiles(input.path)
  await expect(page.getByAltText('Original selected inspection image')).toBeVisible()
  await page.getByRole('button',{name:'Run inspection'}).click()
  await expect(page.getByTestId('score')).toHaveText(prediction.anomaly_score.toFixed(6))
  await page.reload()
  await expect(page.getByTestId('connection')).toHaveText('Backend ready')
  await expect(page.getByTestId('score')).toHaveCount(0)
  writeFileSync(resolve(root,`outputs/deployment/browser-${category}.json`),JSON.stringify({category,device:prediction.device,score:prediction.anomaly_score,threshold:prediction.threshold,decision:prediction.is_anomaly,backend_latency_ms:prediction.latency_ms,driver_turnaround_ms:total,...measures},null,2))
})

test('connection loss is safe and reconnect recovers',async({page})=>{
  await page.route('**/health/live',route=>route.abort('connectionfailed'))
  await page.goto('/')
  await expect(page.getByTestId('connection')).toHaveText('Backend unavailable')
  await expect(page.getByRole('button',{name:'Run inspection'})).toBeDisabled()
  await page.unroute('**/health/live')
  await page.getByRole('button',{name:'Reconnect'}).click()
  await expect(page.getByTestId('connection')).toHaveText('Backend ready')
})
