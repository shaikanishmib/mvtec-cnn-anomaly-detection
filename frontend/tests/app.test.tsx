import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from '../src/App'
import { ApiClient } from '../src/api/client'
import { models, categories, info, prediction } from './fixtures'
const png = () => new File([new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 1])], 'fixture.png', { type: 'image/png' })
type ResponseFactory = (init: RequestInit) => Promise<Response>
function backend(predict?: ResponseFactory) {
  const fetcher = vi.fn((url: string, init: RequestInit = {}) => {
    if (url.endsWith('/api/v1/predict')) return predict ? predict(init) : Promise.resolve(new Response(JSON.stringify(prediction)))
    if (url.includes('/predictions/')) return Promise.resolve(new Response(new Uint8Array([137, 80, 78, 71]), { headers: { 'content-type': 'image/png' } }))
    const body = url.includes('/models') ? { models } : url.includes('/categories') ? { categories } : url.includes('model-info') ? { ...info, category: new URL(url, 'http://local').searchParams.get('category') } : { status: url.endsWith('/live') ? 'live' : 'ready', device: 'cpu' }
    return Promise.resolve(new Response(JSON.stringify(body)))
  })
  vi.stubGlobal('fetch', fetcher); return fetcher
}
async function mount(predict?: ResponseFactory) {
  const fetcher = backend(predict); const view = render(<App client={new ApiClient()} />)
  await screen.findByText('Backend ready'); return { fetcher, ...view }
}
async function select() { fireEvent.change(screen.getByLabelText('Inspection image'), { target: { files: [png()] } }); await screen.findByText('fixture.png') }
describe('inspection application', () => {
  it('renders a semantic empty workspace with no fabricated model result', async () => {
    await mount(); expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('A closer look at quality.')
    expect(screen.getByLabelText('Detection model')).toHaveValue('patchcore'); expect(screen.getByLabelText('Inspection category').querySelectorAll('option')).toHaveLength(15)
    expect(screen.queryByRole('option', { name: /PaDiM/ })).not.toBeInTheDocument(); expect(screen.queryByTestId('score')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Run inspection/ })).toBeDisabled()
  })
  it('shows connecting and unavailable health states and allows retry', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('unavailable')))
    render(<App client={new ApiClient()} />); expect(screen.getByTestId('connection')).toHaveTextContent('Connecting')
    await screen.findByRole('button', { name: 'Reconnect' }); await waitFor(() => expect(screen.getByTestId('connection')).toHaveTextContent('Backend unavailable'))
  })
  it('identifies unavailable model artifacts without showing private server details', async () => {
    const fetcher = backend(); const original = fetcher.getMockImplementation()!
    fetcher.mockImplementation((url, init) => url.endsWith('/ready') ? Promise.resolve(new Response(JSON.stringify({ error: { code: 'model_unavailable', message: 'private server details' } }), { status: 503 })) : original(url, init))
    render(<App client={new ApiClient()} />)
    await waitFor(() => expect(screen.getByTestId('connection')).toHaveTextContent('Model unavailable'))
    expect(screen.queryByText('private server details')).not.toBeInTheDocument()
  })
  it('reports a failed prediction without a misleading normal result', async () => {
    await mount(() => Promise.resolve(new Response(JSON.stringify({ error: { code: 'server_busy', message: 'busy' } }), { status: 503 })))
    await select(); await userEvent.click(screen.getByRole('button', { name: /Run inspection/ }))
    expect(await screen.findByRole('alert')).toHaveTextContent('worker is busy')
    expect(screen.queryByTestId('score')).not.toBeInTheDocument()
  })
  it('updates readiness after losing the backend during prediction', async () => {
    await mount(() => Promise.reject(new TypeError('network offline')))
    await select(); await userEvent.click(screen.getByRole('button', { name: /Run inspection/ }))
    await waitFor(() => expect(screen.getByTestId('connection')).toHaveTextContent('Backend unavailable'))
    expect(screen.getByRole('button', { name: /Run inspection/ })).toBeDisabled()
  })
  it('previews a selected file and releases its object URL on reset', async () => {
    await mount(); await select(); expect(screen.getByAltText('Original selected inspection image')).toHaveAttribute('src', expect.stringContaining('blob:'))
    await userEvent.click(screen.getByRole('button', { name: 'Reset inspection' })); expect(screen.queryByAltText('Original selected inspection image')).not.toBeInTheDocument()
    expect(URL.revokeObjectURL).toHaveBeenCalled()
  })
  it('shows a validation error without submitting an invalid file', async () => {
    const { fetcher } = await mount()
    fireEvent.change(screen.getByLabelText('Inspection image'), { target: { files: [new File(['bad'], 'x.exe', { type: 'image/png' })] } })
    expect(await screen.findByRole('alert')).toHaveTextContent('PNG or JPEG')
    expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/predict'))).toHaveLength(0)
  })
  it('accepts drop input and prevents multiple-file drops', async () => {
    await mount(); const drop = screen.getByText('Drop an image here').closest('.dropzone')!
    fireEvent.drop(drop, { dataTransfer: { files: [png()] } }); await screen.findByText('fixture.png')
    fireEvent.drop(drop, { dataTransfer: { files: [png(), png()] } }); expect(await screen.findByRole('alert')).toHaveTextContent('one image')
  })
  it('shows running state, blocks duplicate submission, then renders actual score fields', async () => {
    let resolve!: (response: Response) => void
    const { fetcher } = await mount(() => new Promise<Response>(done => { resolve = done }))
    await select(); await userEvent.click(screen.getByRole('button', { name: /Run inspection/ }))
    expect(screen.getByRole('button', { name: /Inspecting/ })).toBeDisabled(); expect(screen.getByRole('heading', { name: 'Inspecting your image' })).toBeInTheDocument()
    resolve(new Response(JSON.stringify(prediction)))
    expect(await screen.findByTestId('score')).toHaveTextContent(prediction.anomaly_score.toFixed(6)); expect(screen.getByTestId('threshold')).toHaveTextContent(prediction.threshold.toFixed(6))
    expect(screen.getByRole('heading', { name: /Research normal/ })).toBeInTheDocument(); expect(fetcher.mock.calls.filter(([url]) => url.endsWith('/predict'))).toHaveLength(1)
    expect(await screen.findByAltText(/anomaly heatmap/)).toBeInTheDocument(); expect(await screen.findByAltText(/response overlay/)).toBeInTheDocument()
  })
  it('labels anomalous results as research decisions, never probabilities', async () => {
    await mount(() => Promise.resolve(new Response(JSON.stringify({ ...prediction, anomaly_score: prediction.threshold + .4, is_anomaly: true }))))
    await select(); await userEvent.click(screen.getByRole('button', { name: /Run inspection/ })); expect(await screen.findByRole('heading', { name: /Research anomaly/ })).toBeInTheDocument()
    expect(screen.getByText('L2 distance, not a probability')).toBeInTheDocument(); expect(screen.queryByText(/confidence/i)).not.toBeInTheDocument()
  })
  it('rejects invalid API responses instead of showing success', async () => {
    await mount(() => Promise.resolve(new Response(JSON.stringify({ score: .99 }))))
    await select(); await userEvent.click(screen.getByRole('button', { name: /Run inspection/ })); expect(await screen.findByRole('alert')).toHaveTextContent('invalid response')
    expect(screen.queryByTestId('score')).not.toBeInTheDocument()
  })
  it('handles unavailable or expired visualizations', async () => {
    const { fetcher } = await mount(); fetcher.mockImplementation((url: string) => url.includes('/predictions/') ? Promise.resolve(new Response('{}', { status: 404 })) : Promise.resolve(new Response(JSON.stringify(prediction))))
    await select(); await userEvent.click(screen.getByRole('button', { name: /Run inspection/ }))
    expect(await screen.findByText('Heatmap unavailable')).toBeInTheDocument(); expect(await screen.findByText('Overlay unavailable')).toBeInTheDocument()
  })
  it('handles missing visualization URLs', async () => {
    await mount(() => Promise.resolve(new Response(JSON.stringify({ ...prediction, visualizations: { heatmap_url: null, overlay_url: null, expires_in_seconds: null, scaling: null } }))))
    await select(); await userEvent.click(screen.getByRole('button', { name: /Run inspection/ }))
    expect(await screen.findByText('No heatmap was returned.')).toBeInTheDocument()
  })
  it('clears stale results on category switching', async () => {
    await mount(); await select(); await userEvent.click(screen.getByRole('button', { name: /Run inspection/ })); await screen.findByTestId('score')
    await userEvent.selectOptions(screen.getByLabelText('Inspection category'), 'carpet')
    expect(screen.queryByTestId('score')).not.toBeInTheDocument(); expect(screen.getByRole('button', { name: /Run inspection/ })).toBeEnabled()
  })
  it('cancels pending requests on reset and ignores stale completion', async () => {
    let resolve!: (r: Response) => void; let signal: AbortSignal | null | undefined
    await mount(init => { signal = init.signal; return new Promise<Response>(done => { resolve = done }) })
    await select(); await userEvent.click(screen.getByRole('button', { name: /Run inspection/ })); await userEvent.click(screen.getByRole('button', { name: 'Reset inspection' }))
    expect(signal?.aborted).toBe(true); resolve(new Response(JSON.stringify(prediction)))
    await waitFor(() => expect(screen.queryByTestId('score')).not.toBeInTheDocument())
  })
  it('provides labeled keyboard-operable tabs and zoom controls', async () => {
    await mount(); const tab = screen.getByRole('tab', { name: 'Compare all' }); tab.focus(); await userEvent.keyboard('{ArrowRight}')
    expect(screen.getByRole('tab', { name: 'Original' })).toHaveAttribute('aria-selected', 'true'); expect(screen.getByLabelText('Image zoom')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Skip to inspection' })).toHaveAttribute('href', '#main')
  })
})
