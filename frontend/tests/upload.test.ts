import { describe, expect, it } from 'vitest'
import { validateDimensions, validateFileMetadata, validateUpload } from '../src/upload'
import { imageState } from './setup'
export const pngFile = () => new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0])], 'fixture.png', { type: 'image/png' })
describe('upload checks aligned with backend', () => {
  it('accepts a PNG and records decoded dimensions', async () => { const value = await validateUpload(pngFile()); expect(value.width).toBe(64); expect(value.height).toBe(48); URL.revokeObjectURL(value.url) })
  it.each([new File(['x'], 'x.exe', { type: 'image/png' }), new File(['x'], 'x.png', { type: 'image/jpeg' }), new File([], 'x.png', { type: 'image/png' })])('rejects type, extension and empty files', file => { expect(() => validateFileMetadata(file)).toThrow() })
  it('rejects oversized files', () => { expect(() => validateFileMetadata(new File([new Uint8Array(8 * 1024 * 1024 + 1)], 'x.png', { type: 'image/png' }))).toThrow('8 MiB') })
  it.each([[4097, 1], [2001, 2000], [0, 100]])('rejects dimension and pixel limits', (width, height) => { expect(() => validateDimensions(width, height)).toThrow('4 million') })
  it('checks signature independently from declared content type', async () => { await expect(validateUpload(new File(['corrupt'], 'x.png', { type: 'image/png' }))).rejects.toThrow('content') })
  it('cleans up object URLs for corrupt or oversized decoded images', async () => {
    imageState.corrupt = true; await expect(validateUpload(pngFile())).rejects.toThrow('decoded'); expect(URL.revokeObjectURL).toHaveBeenCalled()
    imageState.corrupt = false; imageState.width = 5000; await expect(validateUpload(pngFile())).rejects.toThrow('4 million')
  })
})
