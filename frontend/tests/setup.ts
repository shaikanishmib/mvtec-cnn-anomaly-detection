import '@testing-library/jest-dom/vitest'
import { afterEach, beforeEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'
export const imageState = { width: 64, height: 48, corrupt: false }
class TestImage {
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  get naturalWidth() { return imageState.width }
  get naturalHeight() { return imageState.height }
  set src(_value: string) { queueMicrotask(() => imageState.corrupt ? this.onerror?.() : this.onload?.()) }
}
let nextUrl = 0
Object.defineProperty(URL, 'createObjectURL', { writable: true, value: vi.fn(() => `blob:test-${++nextUrl}`) })
Object.defineProperty(URL, 'revokeObjectURL', { writable: true, value: vi.fn() })
Object.defineProperty(Blob.prototype, 'arrayBuffer', { value: function () {
  return new Promise<ArrayBuffer>((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result as ArrayBuffer); reader.onerror = reject; reader.readAsArrayBuffer(this) })
} })
Object.defineProperty(performance, 'measure', { writable: true, value: vi.fn() })
Object.defineProperty(performance, 'mark', { writable: true, value: vi.fn() })
beforeEach(() => {
  Object.assign(imageState, { width: 64, height: 48, corrupt: false })
  vi.stubGlobal('Image', TestImage)
  vi.mocked(URL.createObjectURL).mockClear(); vi.mocked(URL.revokeObjectURL).mockClear()
})
afterEach(() => { cleanup(); vi.unstubAllGlobals() })
