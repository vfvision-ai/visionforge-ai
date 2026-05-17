import type {
  TrainingJob, Experiment, ModelVersion, HealthStatus, TrainingSubmitPayload,
} from '@/types'

const BASE = '/api/v1'
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init?.headers as Record<string, string>),
  }
  if (API_KEY) headers['X-API-Key'] = API_KEY
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
    cache: 'no-store',
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

// ── Health ────────────────────────────────────────────────────────────────────
export const getHealth = () =>
  fetch('/health', { cache: 'no-store' })
    .then(r => r.ok ? r.json() as Promise<HealthStatus> : Promise.reject(new Error(`${r.status}`)))

// ── Experiments ───────────────────────────────────────────────────────────────
export const getExperiments = (skip = 0, limit = 100) =>
  request<{ experiments: Experiment[]; total: number }>(`/experiments/?skip=${skip}&limit=${limit}`)

export const createExperiment = (data: { name: string; description?: string; tags?: string[] }) =>
  request<Experiment>('/experiments/', { method: 'POST', body: JSON.stringify(data) })

// ── Training jobs ─────────────────────────────────────────────────────────────
export const getJobs = (params?: { status?: string; framework?: string; limit?: number }) => {
  const qs = new URLSearchParams()
  if (params?.status)    qs.set('status',    params.status)
  if (params?.framework) qs.set('framework', params.framework)
  if (params?.limit)     qs.set('limit',     String(params.limit))
  return request<{ jobs: TrainingJob[]; total: number }>(`/training/?${qs}`)
}

export const getJob = (id: string) =>
  request<TrainingJob>(`/training/${id}`)

export const submitJob = (data: TrainingSubmitPayload) =>
  request<TrainingJob>('/training/', { method: 'POST', body: JSON.stringify(data) })

export const cancelJob = (id: string) =>
  request<void>(`/training/${id}`, { method: 'DELETE' })

// ── Models ────────────────────────────────────────────────────────────────────
export const getModels = (skip = 0, limit = 100) =>
  request<{ models: ModelVersion[]; total: number }>(`/models/?skip=${skip}&limit=${limit}`)

export const promoteModel = (id: string) =>
  request<ModelVersion>(`/models/${id}/promote`, { method: 'POST', body: '{}' })
