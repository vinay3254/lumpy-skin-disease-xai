import type { ModelsResponse, PredictionResponse, ResultsResponse, SamplesResponse, ShapResponse } from './types'

const API_BASE = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '')
export function assetUrl(path: string): string {
  return /^(https?:|data:|blob:)/.test(path) ? path : `${API_BASE}/${path.replace(/^\//, '')}`
}
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, options)
  } catch {
    throw new Error('Cannot reach the local API. Check that the backend is running and try again.')
  }
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null)
    const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : null
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}). Please try again.`)
  }
  return response.json() as Promise<T>
}
function imageForm(file: File, run: string): FormData {
  const body = new FormData()
  body.append('file', file)
  if (run) body.append('run', run)
  return body
}
export const api = {
  models: () => request<ModelsResponse>('/api/models'),
  samples: () => request<SamplesResponse>('/api/samples'),
  results: () => request<ResultsResponse>('/api/results'),
  predict: (file: File, run: string) => request<PredictionResponse>('/api/predict', { method: 'POST', body: imageForm(file, run) }),
  shap: (file: File, run: string) => request<ShapResponse>('/api/shap', { method: 'POST', body: imageForm(file, run) }),
  sample: async (url: string) => {
    const response = await fetch(assetUrl(url))
    if (!response.ok) throw new Error('Could not load this sample. Please try another image.')
    const blob = await response.blob()
    return new File([blob], url.split('/').pop() || 'sample.jpg', { type: blob.type || 'image/jpeg' })
  },
}
