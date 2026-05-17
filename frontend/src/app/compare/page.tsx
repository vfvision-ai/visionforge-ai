'use client'
import { useEffect, useState } from 'react'
import { GitCompare, RefreshCw, Check } from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import Badge from '@/components/Badge'
import { getJobs } from '@/lib/api'
import { formatDate, formatDuration, pct } from '@/lib/utils'
import type { TrainingJob } from '@/types'

function best(vals: (number | null | undefined)[], higher = true): number {
  const nums = vals.filter((v): v is number => v != null)
  if (!nums.length) return -Infinity
  return higher ? Math.max(...nums) : Math.min(...nums)
}

function metricOf(job: TrainingJob): number | null {
  const h = job.training_history
  if (!h?.length) return null
  const last = h[h.length - 1]
  if (job.task_type === 'segmentation') return last.val_miou ?? last.train_miou ?? null
  if (job.task_type === 'detection')    return last.val_miou ?? null  // use available
  return last.val_accuracy ?? last.train_accuracy ?? null
}

function lossOf(job: TrainingJob): number | null {
  const h = job.training_history
  if (!h?.length) return null
  const last = h[h.length - 1]
  return (last.val_loss ?? last.train_loss) as number | null
}

export default function ComparePage() {
  const [jobs,     setJobs]     = useState<TrainingJob[]>([])
  const [loading,  setLoading]  = useState(true)
  const [selected, setSelected] = useState<Set<string>>(new Set())

  async function load() {
    setLoading(true)
    try {
      const data = await getJobs({ status: 'completed', limit: 100 })
      setJobs(data.jobs)
    } catch { /* silent */ }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  function toggle(id: string) {
    setSelected(s => {
      const next = new Set(s)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const compared = jobs.filter(j => selected.has(j.id))

  const metrics  = compared.map(j => metricOf(j))
  const losses   = compared.map(j => lossOf(j))
  const durations = compared.map(j => {
    if (!j.started_at || !j.completed_at) return null
    return (new Date(j.completed_at).getTime() - new Date(j.started_at).getTime()) / 1000
  })

  const bestMetric   = best(metrics, true)
  const bestLoss     = best(losses, false)
  const bestDuration = best(durations, false)

  return (
    <div className="p-8 space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <GitCompare size={22} className="text-brand-400" /> Compare Jobs
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Select completed jobs below to compare their metrics side-by-side
          </p>
        </div>
        <Button variant="secondary" size="sm" icon={<RefreshCw size={14} />} onClick={load} loading={loading}>
          Refresh
        </Button>
      </div>

      {/* Job Selector */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Completed Jobs</h2>
        {loading ? (
          <p className="text-sm text-slate-500 py-4 text-center">Loading…</p>
        ) : jobs.length === 0 ? (
          <p className="text-sm text-slate-500 py-4 text-center">No completed jobs yet.</p>
        ) : (
          <div className="space-y-1 max-h-72 overflow-y-auto">
            {jobs.map(j => (
              <label key={j.id} className={`flex items-center gap-3 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                selected.has(j.id) ? 'bg-brand-500/10 border border-brand-500/30' : 'hover:bg-surface-700'
              }`}>
                <div className={`w-4 h-4 rounded border flex items-center justify-center shrink-0 ${
                  selected.has(j.id) ? 'bg-brand-500 border-brand-500' : 'border-surface-500'
                }`}>
                  {selected.has(j.id) && <Check size={10} className="text-white" />}
                </div>
                <input type="checkbox" className="sr-only" checked={selected.has(j.id)} onChange={() => toggle(j.id)} />
                <div className="flex-1 min-w-0">
                  <span className="text-sm text-white font-medium">{j.architecture}</span>
                  <span className="text-xs text-slate-500 ml-2">{j.dataset_name}</span>
                </div>
                <span className="text-xs text-slate-500 capitalize">{j.framework}</span>
                <span className="text-xs text-slate-600">{formatDate(j.created_at)}</span>
              </label>
            ))}
          </div>
        )}
      </Card>

      {/* Comparison Table */}
      {compared.length >= 1 && (
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">
            Comparison — {compared.length} job{compared.length !== 1 ? 's' : ''}
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-700 text-slate-500 text-left">
                  <th className="pb-3 pr-4 font-medium">Architecture</th>
                  <th className="pb-3 pr-4 font-medium">Framework</th>
                  <th className="pb-3 pr-4 font-medium">Dataset</th>
                  <th className="pb-3 pr-4 font-medium">Task</th>
                  <th className="pb-3 pr-4 font-medium">Val Metric</th>
                  <th className="pb-3 pr-4 font-medium">Val Loss</th>
                  <th className="pb-3 pr-4 font-medium">Duration</th>
                  <th className="pb-3 font-medium">Epochs</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-800">
                {compared.map((j, idx) => {
                  const m   = metrics[idx]
                  const l   = losses[idx]
                  const dur = durations[idx]
                  const hp  = j.hyperparams ?? {}
                  return (
                    <tr key={j.id} className="hover:bg-surface-800/40 transition-colors">
                      <td className="py-3 pr-4 text-white font-medium">{j.architecture}</td>
                      <td className="py-3 pr-4 text-slate-300 capitalize">{j.framework}</td>
                      <td className="py-3 pr-4 text-slate-300">{j.dataset_name}</td>
                      <td className="py-3 pr-4 text-slate-300 capitalize">{j.task_type}</td>
                      <td className={`py-3 pr-4 font-mono text-sm ${m === bestMetric ? 'text-green-400 font-bold' : 'text-slate-300'}`}>
                        {m != null ? pct(m) : '—'}
                        {m === bestMetric && m != null && <span className="ml-1 text-xs">🏆</span>}
                      </td>
                      <td className={`py-3 pr-4 font-mono text-sm ${l != null && l === bestLoss ? 'text-green-400 font-bold' : 'text-slate-300'}`}>
                        {l != null ? l.toFixed(4) : '—'}
                        {l != null && l === bestLoss && <span className="ml-1 text-xs">✓</span>}
                      </td>
                      <td className={`py-3 pr-4 font-mono text-sm ${dur != null && dur === bestDuration ? 'text-green-400 font-bold' : 'text-slate-300'}`}>
                        {formatDuration(j.started_at, j.completed_at)}
                        {dur != null && dur === bestDuration && <span className="ml-1 text-xs">⚡</span>}
                      </td>
                      <td className="py-3 text-slate-400">{String((hp.epochs as number) ?? '—')}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          <div className="mt-4 flex flex-wrap gap-4 text-xs text-slate-500 border-t border-surface-700 pt-3">
            <span>🏆 Best metric (higher)</span>
            <span>✓ Best loss (lower)</span>
            <span>⚡ Fastest training</span>
          </div>
        </Card>
      )}

      {selected.size === 0 && !loading && jobs.length > 0 && (
        <div className="text-center py-6 text-slate-600 text-sm">
          Select 2+ jobs above to see a comparison
        </div>
      )}
    </div>
  )
}
