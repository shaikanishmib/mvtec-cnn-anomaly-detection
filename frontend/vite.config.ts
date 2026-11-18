import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
const backend = process.env.BACKEND_URL || loadEnv(mode, process.cwd(), '').BACKEND_URL || 'http://127.0.0.1:8000'
return {
  plugins: [react()],
  server: { port: 5173, strictPort: true, proxy: { '/api': backend, '/health': backend } },
  preview: { port: 4173, strictPort: true, proxy: { '/api': backend, '/health': backend } },
  test: { environment: 'jsdom', setupFiles: './tests/setup.ts', include: ['tests/**/*.test.{ts,tsx}'], restoreMocks: true },
}
})
