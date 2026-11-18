export const LIMITS = { bytes: 8 * 1024 * 1024, pixels: 4_000_000, dimension: 4096 }
export interface SelectedImage { file: File; url: string; width: number; height: number }
export class UploadError extends Error {}
export function validateFileMetadata(file: File) {
  const ext = file.name.split('.').pop()?.toLowerCase()
  if (!((ext === 'png' && file.type === 'image/png') || ((ext === 'jpg' || ext === 'jpeg') && file.type === 'image/jpeg'))) throw new UploadError('Choose a PNG or JPEG with a matching file type and extension.')
  if (!file.size) throw new UploadError('This file is empty. Choose another image.')
  if (file.size > LIMITS.bytes) throw new UploadError('The image is larger than 8 MiB.')
}
export function validateDimensions(width: number, height: number) {
  if (!width || !height || width > LIMITS.dimension || height > LIMITS.dimension || width * height > LIMITS.pixels) throw new UploadError('Use at most 4 million pixels and 4,096 pixels per dimension.')
}
export async function validateUpload(file: File): Promise<SelectedImage> {
  validateFileMetadata(file)
  const bytes = new Uint8Array(await file.slice(0, 12).arrayBuffer())
  const png = [137, 80, 78, 71, 13, 10, 26, 10].every((v, i) => bytes[i] === v)
  const jpeg = bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255
  if ((file.type === 'image/png' && !png) || (file.type === 'image/jpeg' && !jpeg)) throw new UploadError('The file content does not match its declared image type.')
  const url = URL.createObjectURL(file)
  try {
    const image = new Image()
    await new Promise<void>((resolve, reject) => { image.onload = () => resolve(); image.onerror = () => reject(new UploadError('This image could not be decoded. Choose a valid PNG or JPEG.')); image.src = url })
    validateDimensions(image.naturalWidth, image.naturalHeight)
    return { file, url, width: image.naturalWidth, height: image.naturalHeight }
  } catch (e) { URL.revokeObjectURL(url); throw e }
}
