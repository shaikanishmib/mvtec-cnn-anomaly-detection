import { useEffect } from 'react'
import { api, type ApiClient } from './api/client'
import { useInspection } from './useInspection'
import { Selectors, Upload } from './components/Controls'
import { PredictionSummary } from './components/PredictionSummary'
import { ImageWorkspace } from './components/ImageWorkspace'
import { Metadata } from './components/Metadata'

export default function App({ client = api }: { client?: ApiClient }) {
  const state = useInspection(client)
  useEffect(() => { performance.mark('app:mounted'); if (performance.getEntriesByName('app:start').length) performance.measure('app:initial-render', 'app:start', 'app:mounted') }, [])
  const activeModel = state.models.find(m => m.id === state.model)
  const busy = state.phase === 'running' || state.phase === 'validating'
  const status = state.connection === 'ready' ? 'Backend ready' : state.connection === 'connecting' ? 'Connecting' : state.connection === 'artifact-unavailable' ? 'Model unavailable' : 'Backend unavailable'
  const announcement = state.phase === 'running' ? 'Inspection running. Please wait.' : state.error || (state.result ? `Inspection complete. ${state.result.is_anomaly ? 'Research anomaly' : 'Research normal'}.` : state.connection === 'ready' ? 'Inspection service is ready. Choose an image to begin.' : state.connectionMessage)
  return <><a className="skip-link" href="#main">Skip to inspection</a><header className="app-header"><div className="header-inner"><div className="brand"><svg aria-hidden="true" viewBox="0 0 32 32"><path d="M4 12V4h8M20 4h8v8M28 20v8h-8M12 28H4v-8M10 16h12M16 10v12" /></svg><div><strong>Fieldwork<span> / </span></strong><small>INDUSTRIAL VISION</small></div></div><div className="header-status"><span className="prototype-tag">Research prototype</span><span className={`connection ${state.connection}`} data-testid="connection"><i aria-hidden="true" />{status}</span></div></div></header>
    <main id="main" className="app-shell"><div className="page-heading"><div><p className="eyebrow">INSPECTION CONSOLE <span>/ 01</span></p><h1>A closer look at quality.</h1><p>Inspect a surface. Explore the model response. Keep the evidence in view.</p></div><div className="active-model"><span className="mini-label">ACTIVE ADAPTER</span><strong>{activeModel?.family || 'Discovering backend models'}</strong><small>{activeModel?.version || 'No model selected'}</small></div></div>
    <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">{announcement}</p>
    {state.connection !== 'ready' && <div className={`connection-banner ${state.connection === 'connecting' ? '' : 'warning'}`}><div><strong>{state.connection === 'connecting' ? 'Preparing the inspection service' : status}</strong><p>{state.connectionMessage}{state.connection === 'connecting' ? ' The first model readiness check can take a moment.' : ''}</p></div><button className="button secondary small" type="button" onClick={() => void state.connect()} disabled={state.connection === 'connecting'}>Reconnect</button></div>}
    <div className="console-layout"><aside className="control-panel"><div className="panel-title"><span className="step">01</span><div><h2>Set up inspection</h2><p>Choose the right category.</p></div></div>
      <form onSubmit={e => { e.preventDefault(); void state.submit() }}><Selectors models={state.models} categories={state.categories} model={state.model} category={state.category} onModel={state.changeModel} onCategory={state.changeCategory} /><Upload selected={state.selected} phase={state.phase} onChoose={files => void state.choose(files)} />
        {state.error && <div className="error-message" role="alert"><strong>Inspection needs attention</strong><p>{state.error}</p></div>}
        <button className="button primary submit" type="submit" disabled={!state.selected || state.connection !== 'ready' || busy} aria-busy={state.phase === 'running'}>{state.phase === 'running' ? <><span className="loading-ring" aria-hidden="true" /> Inspecting…</> : <>Run inspection <span aria-hidden="true">↗</span></>}</button><button className="reset-button" type="button" onClick={state.reset} disabled={!state.selected && !state.error && !state.result && !busy}>Reset inspection</button>
      </form><div className="research-note"><span aria-hidden="true">ⓘ</span><div><strong>Evidence, not a verdict.</strong><p>A research anomaly is not a verified manufacturing defect. False alarms and missed defects remain possible, particularly across different textures and acquisition conditions.</p></div></div></aside>
      <div className="results-column"><PredictionSummary result={state.result} running={state.phase === 'running'} turnaround={state.turnaround} /><ImageWorkspace selected={state.selected} result={state.result} running={state.phase === 'running'} onPreview={state.previewReady} onReady={state.visualizationsReady} client={client} /><Metadata model={activeModel} info={state.info?.category === state.category ? state.info : undefined} result={state.result} error={state.infoError} /></div>
    </div><footer className="app-footer"><span>FIELDWORK · LOCAL RESEARCH WORKSPACE</span><span>Temporary results · No inspection history is persisted</span></footer></main></>
}
