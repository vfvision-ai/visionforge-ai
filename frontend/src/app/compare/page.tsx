'use client'
import { useEffect, useState } from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, Legend,
} from 'recharts'
import { GitCompare, RefreshCw, Check, Download } from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import { Select } from '@/components/FormControls'
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
  if (job.task_type === 'detection')    return last.val_map50 ?? null
  return last.val_accuracy ?? last.train_accuracy ?? null
}

function lossOf(job: TrainingJob): number | null {
  const h = job.training_history
  if (!h?.length) return (job.results?.best_loss as number | null) ?? null
  const last = h[h.length - 1]
  return (last.val_loss ?? last.train_loss) as number | null
}

const FW_TEXT: Record<string, string> = { pytorch: 'text-orange-400', tensorflow: 'text-yellow-400', sklearn: 'text-blue-400' }
const fwTextClass = (fw: string) => FW_TEXT[fw] ?? 'text-purple-400'

function extractResult(job: TrainingJob, ...keys: string[]): number | null {
  const r = job.results as Record<string, unknown> | null
  if (!r) return null
  for (const k of keys) { const v = r[k]; if (typeof v === 'number') return v }
  return null
}

const METRIC_OPTIONS = [
  { value: 'metric',    label: 'Val Metric (Acc / mIoU / mAP)' },
  { value: 'loss',      label: 'Val Loss' },
  { value: 'precision', label: 'Precision' },
  { value: 'recall',    label: 'Recall' },
  { value: 'f1',        label: 'F1 Score' },
  { value: 'duration',  label: 'Duration (s)' },
]
const CURVE_COLORS = ['#6366f1', '#22d3ee', '#f97316', '#22c55e', '#a855f7', '#f43f5e']

function resolveBar(job: TrainingJob, key: string, mVal: number | null, lVal: number | null, dur: number | null): number | null {
  if (key === 'metric')    return mVal != null ? parseFloat((mVal * 100).toFixed(1)) : null
  if (key === 'loss')      return lVal != null ? parseFloat(lVal.toFixed(4))         : null
  if (key === 'duration')  return dur  != null ? parseFloat(dur.toFixed(0))          : null
  if (key === 'precision') return extractResult(job, 'best_precision', 'final_precision', 'precision')
  if (key === 'recall')    return extractResult(job, 'best_recall',    'final_recall',    'recall')
  if (key === 'f1')        return extractResult(job, 'best_f1',        'final_f1',        'f1')
  return null
}

function doExportCSV(jobs: TrainingJob[], metrics: (number|null)[], losses: (number|null)[], durations: (number|null)[]) {
  const headers = ['id','architecture','framework','dataset','task','val_metric_%','val_loss','precision','recall','f1','epochs','duration_s','started_at']
  const rows = jobs.map((j, i) => [
    j.id.slice(0, 8), j.architecture, j.framework, j.dataset_name, j.task_type,
    metrics[i]  != null ? (metrics[i]! * 100).toFixed(2) : '',
    losses[i]   != null ? losses[i]!.toFixed(4)          : '',
    extractResult(j, 'best_precision', 'precision') ?? '',
    extractResult(j, 'best_recall',    'recall')    ?? '',
    extractResult(j, 'best_f1',        'f1')        ?? '',
    (j.hyperparams?.epochs ?? '').toString(),
    durations[i] != null ? durations[i]!.toFixed(1) : '',
    j.started_at ?? '',
  ])
  const csv = [headers, ...rows].map(r => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = `comparison_${Date.now()}.csv`
  document.body.appendChild(a); a.click(); document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export default function ComparePage() {
  const [jobs,     setJobs]     = useState<TrainingJob[]>([])
  const [loading,  setLoading]  = useState(true)
  const [selected,  setSelected]  = useState<Set<string>>(new Set())
  const [barMetric, setBarMetric] = useState('metric')

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

  const compared  = jobs.filter(j => selected.has(j.id))
  const metrics   = compared.map(j => metricOf(j))
  const losses    = compared.map(j => lossOf(j))
  const durations = compared.map(j => {
    if (!j.started_at || !j.completed_at) return null
    return (new Date(j.completed_at).getTime() - new Date(j.started_at).getTime()) / 1000
  })

  const bestMetric   = best(metrics, true)
  const bestLoss     = best(losses,    false)
  const bestDuration = best(durations, false)

  // Bar chart data
  const barData = compared.map((j, i) => ({
    name:  `${j.architecture.slice(0, 12)} / ${j.framework.slice(0, 2)}`,
    value: resolveBar(j, barMetric, metrics[i], losses[i], durations[i]) ?? 0,
    color: ({ pytorch: '#f97316', tensorflow: '#eab308', sklearn: '#3b82f6' } as Record<string,string>)[j.framework] ?? '#a855f7',
  }))

  // Training curves
  const maxEpochs = compared.reduce((m, j) => Math.max(m, j.training_history?.length ?? 0), 0)
  const curveData = Array.from({ length: maxEpochs }, (_, e) => {
    const pt: Record<string, number | string> = { epoch: e + 1 }
    compared.forEach((j, ci) => {
      const ep = j.training_history?.[e]
      if (!ep) return
      const v = j.task_type === 'segmentation' ? (ep.val_miou ?? ep.train_miou)
              : j.task_type === 'detection'    ? ep.val_map50
              : (ep.val_accuracy ?? ep.train_accuracy)
      if (v != null) pt[`job${ci}`] = parseFloat((v * 100).toFixed(2))
    })
    return pt
  })
  const hasCurves = maxEpochs > 0

  const mlabel = (j: TrainingJob) =>
    j.task_type === 'segmentation' ? 'mIoU' : j.task_type === 'detection' ? 'mAP' : 'Acc'

  return (
    <div className="p-8 space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <GitCompare size={22} className="text-brand-400" /> Compare Jobs
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Select completed jobs to compare metrics, view charts, and export
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
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-slate-300">
              Comparison — {compared.length} job{compared.length !== 1 ? 's' : ''}
            </h2>
            <Button size="sm" variant="secondary" icon={<Download size={13} />}
              onClick={() => doExportCSV(compared, metrics, losses, durations)}>
              Export CSV
            </Button>
          </div>
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
                  <th className="pb-3 pr-4 font-medium">Precision</th>
                  <th className="pb-3 pr-4 font-medium">Recall</th>
                  <th className="pb-3 pr-4 font-medium">F1</th>
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
                  const prec = extractResult(j, 'best_precision', 'final_precision', 'precision')
                  const rec  = extractResult(j, 'best_recall',    'final_recall',    'recall')
                  const f1   = extractResult(j, 'best_f1',        'final_f1',        'f1')
                  return (
                    <tr key={j.id} className="hover:bg-surface-800/40 transition-colors">
                      <td className="py-3 pr-4 text-white font-medium">{j.architecture}</td>
                      <td className={`py-3 pr-4 capitalize font-medium ${fwTextClass(j.framework)}`}>{j.framework}</td>
                      <td className="py-3 pr-4 text-slate-300">{j.dataset_name}</td>
                      <td className="py-3 pr-4 text-slate-300 capitalize">{j.task_type}</td>
                      <td className={`py-3 pr-4 font-mono text-sm ${m === bestMetric && m != null ? 'text-green-400 font-bold' : 'text-slate-300'}`}>
                        {m != null ? `${pct(m)} ${mlabel(j)}` : '—'}
                        {m === bestMetric && m != null && <span className="ml-1 text-xs">🏆</span>}
                      </td>
                      <td className={`py-3 pr-4 font-mono text-sm ${l != null && l === bestLoss ? 'text-green-400 font-bold' : 'text-slate-300'}`}>
                        {l != null ? l.toFixed(4) : '—'}
                        {l != null && l === bestLoss && <span className="ml-1 text-xs">✓</span>}
                      </td>
                      <td className="py-3 pr-4 font-mono text-sm text-slate-400">{prec != null ? `${(prec*100).toFixed(1)}%` : '—'}</td>
                      <td className="py-3 pr-4 font-mono text-sm text-slate-400">{rec  != null ? `${(rec *100).toFixed(1)}%` : '—'}</td>
                      <td className="py-3 pr-4 font-mono text-sm text-slate-400">{f1   != null ? `${(f1  *100).toFixed(1)}%` : '—'}</td>
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
            <span>🏆 Best metric (higher is better)</span>
            <span>✓ Best loss (lower is better)</span>
            <span>⚡ Fastest training</span>
          </div>
        </Card>
      )}

      {/* Charts — shown when 2+ jobs selected */}
      {compared.length >= 2 && (
        <>
          {/* Bar chart */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-slate-300">Metric Bar Chart</h2>
              <div className="w-60">
                <Select label="" value={barMetric} onChange={e => setBarMetric(e.target.value)} options={METRIC_OPTIONS} />
              </div>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} margin={{ left: 0, right: 16, top: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#64748b" tick={{ fontSize: 10 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                <Bar dataKey="value" isAnimationActive={false}>
                  {barData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          {/* Training curves */}
          {hasCurves && (
            <Card>
              <h2 className="text-sm font-semibold text-slate-300 mb-4">Training Curves (Val Metric %)</h2>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={curveData} margin={{ left: 0, right: 16, top: 4, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 10 }} tickFormatter={v => `${v}%`} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} formatter={(v: number) => [`${v}%`]} />
                  <Legend formatter={(value: string) => {
                    const ci = parseInt(value.replace('job', ''), 10)
                    const j = compared[ci]
                    return j ? `${j.architecture} (${j.framework})` : value
                  }} />
                  {compared.map((_, ci) => (
                    <Line key={ci} type="monotone" dataKey={`job${ci}`}
                      stroke={CURVE_COLORS[ci % CURVE_COLORS.length]}
                      dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </>
      )}

      {selected.size === 0 && !loading && jobs.length > 0 && (
        <div className="text-center py-6 text-slate-600 text-sm">
          Select 2+ jobs above to see a comparison
        </div>
      )}
    </div>
  )
}
