import { useRef, useState } from 'react'
import type { Model } from '../api/types'
import type { SelectedImage } from '../upload'
import type { Phase } from '../useInspection'

export function Selectors({ models, categories, model, category, onModel, onCategory }: {
  models: Model[]; categories: string[]; model: string; category: string; onModel: (s: string) => void; onCategory: (s: string) => void
}) {
  return <div className="selectors"><label htmlFor="model">Detection model</label><select id="model" value={model} disabled={!models.length} onChange={e => onModel(e.target.value)}>{!models.length && <option value="">Discovering models…</option>}{models.map(m => <option key={m.id} value={m.id}>{m.family}</option>)}</select>
    <div className="field-title"><label htmlFor="category">Inspection category</label><span>{categories.length ? `${categories.length} available` : 'Connecting'}</span></div><select id="category" value={category} disabled={!categories.length} onChange={e => onCategory(e.target.value)}>{!categories.length && <option value="">Discovering categories…</option>}{categories.map(c => <option key={c} value={c}>{c.replaceAll('_', ' ')}</option>)}</select></div>
}
export function Upload({ selected, phase, onChoose }: { selected?: SelectedImage; phase: Phase; onChoose: (files: File[]) => void }) {
  const input = useRef<HTMLInputElement>(null); const [drag, setDrag] = useState(false)
  return <section className="upload-section" aria-label="Image upload"><div className="field-title"><label id="upload-title" htmlFor="inspection-image">Inspection image</label><span>PNG / JPEG</span></div>
    <div className={`dropzone ${drag ? 'drag' : ''}`} onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)} onDrop={e => { e.preventDefault(); setDrag(false); onChoose(Array.from(e.dataTransfer.files)) }}>
      <svg viewBox="0 0 40 40" aria-hidden="true"><path d="M9 25v7h22v-7M20 26V8m-7 7 7-7 7 7" /></svg>
      <strong>{phase === 'validating' ? 'Checking your image…' : selected ? 'Replace inspection image' : 'Drop an image here'}</strong><span>or choose a file from your device</span>
      <button type="button" className="button secondary small" onClick={() => input.current?.click()}>Choose image</button>
      <input ref={input} id="inspection-image" aria-label="Inspection image" className="sr-only" type="file" accept="image/png,image/jpeg,.png,.jpg,.jpeg" onChange={e => { if (e.target.files?.length) onChoose(Array.from(e.target.files)); e.target.value = '' }} />
    </div><p className="help" id="upload-limits">Up to 8 MiB · 4 million pixels · 4,096 px per side.<br />The backend performs the final validation.</p>
    {selected && <div className="file-meta"><span className="file-symbol" aria-hidden="true">▧</span><div><strong title={selected.file.name}>{selected.file.name}</strong><span>{selected.width.toLocaleString()} × {selected.height.toLocaleString()} px · {(selected.file.size / 1024).toFixed(1)} KiB</span></div><span className="file-check" aria-label="File validated">✓</span></div>}
  </section>
}
