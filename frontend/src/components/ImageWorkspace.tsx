import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { api, ApiClient, ApiError } from '../api/client'
import type { Prediction } from '../api/types'
import type { SelectedImage } from '../upload'

type View = 'compare' | 'original' | 'heatmap' | 'overlay'
function RemoteImage({ url, kind, zoom, client, onReady }: { url: string | null; kind: 'heatmap' | 'overlay'; zoom: number; client: ApiClient; onReady: (kind: string) => void }) {
  const [source, setSource] = useState(''); const [error, setError] = useState(''); const start = useRef(0)
  useEffect(() => {
    setSource(''); setError(''); start.current = performance.now()
    if (!url) return
    const controller = new AbortController(); let objectUrl = ''
    void client.image(url, controller.signal).then(blob => {
      if (controller.signal.aborted) return
      objectUrl = URL.createObjectURL(blob); setSource(objectUrl)
    }).catch(e => {
      if (!controller.signal.aborted) setError(e instanceof ApiError ? e.message : 'This temporary visualization is unavailable.')
    })
    return () => { controller.abort(); if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [client, url])
  useEffect(() => { if (!url || error) onReady(kind) }, [url, error, onReady, kind])
  if (!url) return <div className="image-placeholder"><span aria-hidden="true">⌁</span><p>No {kind} was returned.</p><small>Temporary images are not guaranteed to persist.</small></div>
  if (error) return <div className="image-placeholder unavailable" role="status"><span aria-hidden="true">!</span><p>{kind === 'heatmap' ? 'Heatmap' : 'Overlay'} unavailable</p><small>{error}</small></div>
  if (!source) return <div className="image-placeholder"><span className="loading-ring" aria-hidden="true" /><p>Loading {kind}…</p></div>
  return <div className="image-scroll"><img src={source} alt={kind === 'heatmap' ? 'Model anomaly heatmap with per-image display scaling' : 'Model response overlay on the inspection image'} style={{ width: `${zoom}%`, height: `${zoom}%` }} onLoad={() => { performance.measure(`inspection:${kind}`, { start: start.current, end: performance.now() }); onReady(kind) }} onError={() => setError('This image could not be displayed. Run the inspection again.')} /></div>
}
export function ImageWorkspace({ selected, result, running, onPreview, onReady, client = api }: {
  selected?: SelectedImage; result?: Prediction; running: boolean; onPreview: () => void; onReady: () => void; client?: ApiClient
}) {
  const [view, setView] = useState<View>('compare'); const [zoom, setZoom] = useState(100)
  const loaded = useRef(new Set<string>()); const reported = useRef(false)
  useLayoutEffect(() => { loaded.current.clear(); reported.current = false; setZoom(100) }, [result?.prediction_id, selected?.url])
  const imageReady = useCallback((kind: string) => {
    loaded.current.add(kind)
    if (loaded.current.has('heatmap') && loaded.current.has('overlay') && !reported.current) { reported.current = true; onReady() }
  }, [onReady])
  const kinds = ['original', 'heatmap', 'overlay'] as const
  return <section className="workspace" aria-labelledby="workspace-title"><div className="workspace-heading"><div><p className="eyebrow">VISUAL INSPECTION</p><h2 id="workspace-title">Look closer.</h2></div><label className="zoom-label">Zoom<select aria-label="Image zoom" value={zoom} onChange={e => setZoom(Number(e.target.value))}><option value="100">Fit</option><option value="150">150%</option><option value="200">200%</option></select></label></div>
    <div className="view-tabs" role="tablist" aria-label="Inspection image views">{(['compare', 'original', 'heatmap', 'overlay'] as View[]).map(item => <button key={item} id={`tab-${item}`} type="button" role="tab" aria-selected={view === item} aria-controls="image-panel" tabIndex={view === item ? 0 : -1} onClick={() => setView(item)} onKeyDown={e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); const all: View[] = ['compare', 'original', 'heatmap', 'overlay']; const next = all[(all.indexOf(item) + (e.key === 'ArrowRight' ? 1 : 3)) % 4]; setView(next); document.getElementById(`tab-${next}`)?.focus() } }}>{item === 'compare' ? 'Compare all' : item[0].toUpperCase() + item.slice(1)}</button>)}</div>
    <div className={`image-grid ${view !== 'compare' ? 'single' : ''}`} id="image-panel" role="tabpanel" aria-labelledby={`tab-${view}`}>
      {kinds.map((kind, index) => <figure hidden={view !== 'compare' && view !== kind} className="image-card" key={`${kind}-${result?.prediction_id || 'empty'}`}><figcaption><span><b>{String(index + 1).padStart(2, '0')}</b>{kind[0].toUpperCase() + kind.slice(1)}</span><small>{kind === 'original' ? 'Input image' : result ? 'Backend output' : 'Awaiting result'}</small></figcaption>
        <div className="image-stage">{kind === 'original' ? selected ? <div className="image-scroll"><img src={selected.url} alt="Original selected inspection image" onLoad={onPreview} style={{ width: `${zoom}%`, height: `${zoom}%` }} /></div> : <div className="image-placeholder"><span className="frame-mark" aria-hidden="true" /><p>Your image goes here</p><small>Select a supported inspection image.</small></div> : result ? <RemoteImage url={result.visualizations[`${kind}_url`]} kind={kind} zoom={zoom} client={client} onReady={imageReady} /> : <div className="image-placeholder"><span aria-hidden="true">{kind === 'heatmap' ? '▥' : '◫'}</span><p>{running ? 'Generating visualization…' : kind === 'heatmap' ? 'Surface the response' : 'See it in context'}</p><small>{running ? 'Waiting for the backend.' : 'Available after inspection.'}</small></div>}</div>
      </figure>)}
    </div><div className="workspace-note"><span aria-hidden="true">ⓘ</span><p>Heatmaps use per-image display scaling. Colors are not probabilities and should not be compared as absolute severity. Zoom changes the view only.</p>{result && <span className="temporary-tag">Temporary images</span>}</div>
  </section>
}
