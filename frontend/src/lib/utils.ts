import { type ClassValue, clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  })
}

export function formatDuration(started?: string | null, completed?: string | null): string {
  if (!started) return '—'
  const end = completed ? new Date(completed) : new Date()
  const secs = Math.round((end.getTime() - new Date(started).getTime()) / 1000)
  if (secs < 60)   return `${secs}s`
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`
}

export function pct(v: number | null | undefined): string {
  if (v == null) return '—'
  const n = v <= 1 ? v * 100 : v
  return `${n.toFixed(2)}%`
}
