import { useCallback, useEffect, useRef, useState } from 'react'
import { api, ApiClient, ApiError } from './api/client'
import type { Model, ModelInfo, Prediction } from './api/types'
import { UploadError, validateUpload } from './upload'
import type { SelectedImage } from './upload'

export type Connection = 'connecting' | 'ready' | 'unavailable' | 'artifact-unavailable'
export type Phase = 'idle' | 'validating' | 'selected' | 'running' | 'succeeded' | 'failed'
const cancelled = (e: unknown) => e instanceof DOMException && e.name === 'AbortError'
export function useInspection(client: ApiClient = api) {
  const [connection, setConnection] = useState<Connection>('connecting')
  const [connectionMessage, setConnectionMessage] = useState('Connecting to the inspection service…')
  const [models, setModels] = useState<Model[]>([]); const [categories, setCategories] = useState<string[]>([])
  const [model, setModel] = useState(''); const [category, setCategory] = useState('')
  const [info, setInfo] = useState<ModelInfo>(); const [infoError, setInfoError] = useState('')
  const [selected, setSelected] = useState<SelectedImage>(); const [result, setResult] = useState<Prediction>()
  const [phase, setPhase] = useState<Phase>('idle'); const [error, setError] = useState('')
  const [turnaround, setTurnaround] = useState<number>()
  const pending = useRef<AbortController | null>(null); const sequence = useRef(0)
  const health = useRef<AbortController | null>(null); const submissionStart = useRef(0)
  const previewStart = useRef(0); const previewMeasured = useRef(false)
  const runStart = useRef(false)
  const invalidate = useCallback(() => {
    sequence.current++; pending.current?.abort(); pending.current = null; runStart.current = false
    setResult(undefined); setError(''); setTurnaround(undefined)
    return sequence.current
  }, [])
  const connect = useCallback(async () => {
    health.current?.abort(); const controller = new AbortController(); health.current = controller
    setConnection('connecting'); setConnectionMessage('Connecting to the inspection service…')
    try {
      const [, discovered, options] = await Promise.all([client.live(controller.signal), client.models(controller.signal), client.categories(controller.signal)])
      if (controller.signal.aborted) return
      setModels(discovered); setCategories(options)
      setModel(m => discovered.some(x => x.id === m) ? m : discovered[0].id)
      setCategory(c => options.includes(c) ? c : options[0])
      await client.ready(controller.signal)
      if (!controller.signal.aborted) { setConnection('ready'); setConnectionMessage('Backend ready') }
    } catch (e) {
      if (controller.signal.aborted || cancelled(e)) return
      setConnection(e instanceof ApiError && e.code === 'model_unavailable' ? 'artifact-unavailable' : 'unavailable')
      setConnectionMessage(e instanceof ApiError ? e.message : 'The inspection service is unavailable.')
    }
  }, [client])
  const dispose = useCallback(() => { health.current?.abort(); pending.current?.abort(); sequence.current++ }, [])
  useEffect(() => { void connect(); return dispose }, [connect, dispose])
  useEffect(() => {
    setInfo(undefined); setInfoError('')
    if (!model || !category || connection !== 'ready') return
    const controller = new AbortController()
    void client.info(model, category, controller.signal).then(value => { if (!controller.signal.aborted) setInfo(value) }).catch(e => {
      if (!controller.signal.aborted && !cancelled(e)) setInfoError(e instanceof ApiError ? e.message : 'Model details are unavailable.')
    })
    return () => controller.abort()
  }, [client, model, category, connection])
  useEffect(() => () => { if (selected) URL.revokeObjectURL(selected.url) }, [selected])
  const choose = useCallback(async (files: File[]) => {
    const id = invalidate(); setSelected(undefined); previewMeasured.current = false; previewStart.current = performance.now()
    if (files.length !== 1) { setPhase('idle'); setError('Choose one image at a time.'); return }
    setPhase('validating')
    try {
      const value = await validateUpload(files[0])
      if (id !== sequence.current) { URL.revokeObjectURL(value.url); return }
      setSelected(value); setPhase('selected')
    } catch (e) {
      if (id !== sequence.current) return
      setError(e instanceof UploadError ? e.message : 'The image could not be read.'); setPhase('idle')
    }
  }, [invalidate])
  const previewReady = useCallback(() => {
    if (!previewMeasured.current && previewStart.current) {
      performance.measure('inspection:preview', { start: previewStart.current, end: performance.now() }); previewMeasured.current = true
    }
  }, [])
  const reset = useCallback(() => { invalidate(); setSelected(undefined); setPhase('idle'); previewStart.current = 0 }, [invalidate])
  const changeCategory = (value: string) => { invalidate(); setCategory(value); setPhase(selected ? 'selected' : 'idle') }
  const changeModel = (value: string) => { invalidate(); setModel(value); setPhase(selected ? 'selected' : 'idle') }
  const submit = async () => {
    if (!selected || !model || !category || connection !== 'ready' || runStart.current) return
    runStart.current = true; const id = ++sequence.current; const controller = new AbortController(); pending.current = controller
    setPhase('running'); setError(''); setResult(undefined); setTurnaround(undefined); submissionStart.current = performance.now()
    try {
      const value = await client.predict(selected.file, model, category, controller.signal)
      if (id !== sequence.current) return
      setResult(value); setPhase('succeeded')
      performance.measure('inspection:response', { start: submissionStart.current, end: performance.now() })
    } catch (e) {
      if (id !== sequence.current || cancelled(e)) return
      setPhase('failed'); setError(e instanceof ApiError ? e.message : 'Inspection failed. No result was accepted.')
      if (e instanceof ApiError && (e.code === 'connection_error' || e.code === 'model_unavailable')) {
        setConnection(e.code === 'model_unavailable' ? 'artifact-unavailable' : 'unavailable')
        setConnectionMessage(e.message)
      }
    } finally { if (id === sequence.current) { runStart.current = false; pending.current = null } }
  }
  const visualizationsReady = useCallback(() => {
    if (!submissionStart.current) return
    const elapsed = performance.now() - submissionStart.current
    setTurnaround(elapsed); performance.measure('inspection:complete', { start: submissionStart.current, end: performance.now() })
  }, [])
  return { connection, connectionMessage, connect, models, categories, model, category, info, infoError,
    selected, result, phase, error, turnaround, choose, previewReady, reset, changeCategory, changeModel, submit, visualizationsReady }
}
