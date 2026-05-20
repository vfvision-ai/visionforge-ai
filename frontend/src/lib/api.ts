import type {
  TrainingJob, Experiment, ModelVersion, HealthStatus, TrainingSubmitPayload, SystemInfo,
  User, AuthTokens, UserListResponse,
} from '@/types'

const BASE = '/api/v1'
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? ''

function getAccessToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem('vf_access_token')
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init?.headers as Record<string, string>),
  }
  if (API_KEY) headers['X-API-Key'] = API_KEY
  const token = getAccessToken()
  if (token) headers['Authorization'] = `Bearer ${token}`
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers,
    cache: 'no-store',
  })
  if (!res.ok) {
    // On 401, clear auth state and redirect to login so the user isn't stuck
    if (res.status === 401 && typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
      localStorage.removeItem('vf_access_token')
      localStorage.removeItem('vf_refresh_token')
      localStorage.removeItem('vf_user')
      window.location.href = '/login'
    }
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${text}`)
  }
  return res.json()
}

// ── Auth ──────────────────────────────────────────────────────────────────────
export const loginUser = (email: string, password: string) =>
  request<AuthTokens>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  })

export const registerUser = (email: string, full_name: string, password: string) =>
  request<User>('/auth/register', {
    method: 'POST',
    body: JSON.stringify({ email, full_name, password }),
  })

export const refreshAccessToken = (refresh_token: string) =>
  request<AuthTokens>('/auth/refresh', {
    method: 'POST',
    body: JSON.stringify({ refresh_token }),
  })

export const getMe = () => request<User>('/auth/me')

// ── Admin: user management ────────────────────────────────────────────────────
export const getAdminUsers = (skip = 0, limit = 100) =>
  request<UserListResponse>(`/auth/users?skip=${skip}&limit=${limit}`)

export const patchUser = (userId: string, patch: { is_active?: boolean; role?: string }) =>
  request<User>(`/auth/users/${userId}`, { method: 'PATCH', body: JSON.stringify(patch) })

export const deleteUser = (userId: string) =>
  request<void>(`/auth/users/${userId}`, { method: 'DELETE' })

// ── Health ────────────────────────────────────────────────────────────────────
export const getHealth = () =>
  fetch('/health', { cache: 'no-store' })
    .then(r => r.ok ? r.json() as Promise<HealthStatus> : Promise.reject(new Error(`${r.status}`)))

export const getSystemInfo = () =>
  fetch('/system-info', { cache: 'no-store' })
    .then(r => r.ok ? r.json() as Promise<SystemInfo> : Promise.reject(new Error(`${r.status}`)))

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

async function downloadWithAuth(url: string, filename: string): Promise<void> {
  const hdrs: Record<string, string> = {}
  if (API_KEY) hdrs['X-API-Key'] = API_KEY
  const token = getAccessToken()
  if (token) hdrs['Authorization'] = `Bearer ${token}`
  const res = await fetch(url, { headers: hdrs, cache: 'no-store' })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${text}`)
  }
  const blob = await res.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(a.href)
}

export function downloadModelFile(jobId: string) {
  downloadWithAuth(`/api/v1/training/${jobId}/download`, `model_${jobId.slice(0, 8)}.pt`)
}
export function downloadHistoryCSV(jobId: string) {
  downloadWithAuth(`/api/v1/training/${jobId}/history.csv`, `history_${jobId.slice(0, 8)}.csv`)
}
export function downloadResultsJSON(jobId: string) {
  downloadWithAuth(`/api/v1/training/${jobId}/results.json`, `results_${jobId.slice(0, 8)}.json`)
}

export async function generateTestSamples(jobId: string, numSamples: number, fmt: string): Promise<void> {
  const params = new URLSearchParams({ num_samples: String(numSamples), image_format: fmt })
  const hdrs: Record<string, string> = {}
  if (API_KEY) hdrs['X-API-Key'] = API_KEY
  const token = getAccessToken()
  if (token) hdrs['Authorization'] = `Bearer ${token}`
  const res = await fetch(`/api/v1/training/${jobId}/test-samples?${params}`, {
    method: 'POST',
    headers: hdrs,
    cache: 'no-store',
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status}: ${text}`)
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `test_samples_${jobId.slice(0, 8)}.zip`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// ── Models ────────────────────────────────────────────────────────────────────
export const getModels = (skip = 0, limit = 100) =>
  request<{ models: ModelVersion[]; total: number }>(`/models/?skip=${skip}&limit=${limit}`)

export const promoteModel  = (id: string) =>
  request<ModelVersion>(`/models/${id}/promote`, { method: 'POST', body: '{}' })
export const deleteModel   = (id: string) =>
  request<void>(`/models/${id}`, { method: 'DELETE' })
export const backfillModels = () =>
  request<{ backfilled: number; model_ids: string[] }>('/models/backfill', { method: 'POST', body: '{}' })
