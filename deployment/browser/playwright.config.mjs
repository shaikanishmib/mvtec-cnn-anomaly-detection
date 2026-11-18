import { defineConfig } from '../../frontend/node_modules/@playwright/test/index.mjs'
export default defineConfig({
  testDir: '.', testMatch: 'release.spec.mjs', workers: 1, retries: 0,
  timeout: 120000, expect: { timeout: 20000 },
  outputDir: '../../outputs/deployment/browser-results',
  reporter: [['list'], ['json', { outputFile: '../../outputs/deployment/browser-tests.json' }]],
  use: { baseURL: 'http://127.0.0.1:8080', channel: 'msedge', headless: true, viewport: { width: 1440, height: 1000 } },
})
