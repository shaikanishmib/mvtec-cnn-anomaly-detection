import { defineConfig } from '@playwright/test'
export default defineConfig({
  testDir: './e2e', timeout: 120_000, expect: { timeout: 15_000 }, workers: 1, retries: 0,
  outputDir: '../outputs/frontend/playwright-results',
  reporter: [['list'], ['json', { outputFile: '../outputs/frontend/browser-tests.json' }]],
  use: { baseURL: 'http://127.0.0.1:5173', channel: 'msedge', headless: true, screenshot: 'only-on-failure', trace: 'retain-on-failure', viewport: { width: 1440, height: 1000 } },
})
