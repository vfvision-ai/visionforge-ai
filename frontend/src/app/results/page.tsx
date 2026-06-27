'use client'
import { useEffect, useState } from 'react'
import Link from 'next/link'
import { BarChart2, RefreshCw, ExternalLink, Download, ChevronUp, ChevronDown } from 'lucide-react'
import Card from '@/components/Card'
import Badge from '@/components/Badge'
import Button from '@/components/Button'
import { getJobs } from '@/lib/api'
import { formatDate, formatDuration } from '@/lib/utils'
import type { TrainingJob } from '@/types'

const FILTERS = ['all', 'running', 'completed', 'failed', 'pending'] as const

export default function ResultsPage() {
  const [jobs,    setJobs]    = useState<TrainingJob[]>([])
  const [filter,  setFilter]  = useState<typeof FILTERS[number]>('all')
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')
  const [sortKey, setSortKey] = useState('created_at')
  const [sortDir, setSortDir] = useState<'asc'|'desc'>('desc')

  async function load(quiet = false) {
    if (!quiet) { setLoading(true); setError('') }
    try {
      const data = await getJobs()
      setJobs(data.jobs)
    } catch (e: unknown) {
      if (!quiet) setError(e instanceof Error ? e.message : 'Failed to load jobs')
    } finally {
      if (!quiet) setLoading(false)
    }
  }

  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-refresh every 5s while any job is running/pending
  useEffect(() => {
    const active = jobs.some(j => j.status === 'running' || j.status === 'pending')
    if (!active) return
    const id = setInterval(() => load(true), 5000)
    return () => clearInterval(id)
  }, [jobs]) // eslint-disable-line react-hooks/exhaustive-deps

  const visible = filter === 'all' ? jobs : jobs.filter(j => j.status === filter)

  function sortVal(job: TrainingJob): number | string {
    if (sortKey === 'created_at')    return job.created_at ?? ''
    if (sortKey === 'dataset_name')  return job.dataset_name ?? ''
    if (sortKey === 'architecture')  return job.architecture ?? ''
    if (sortKey === 'framework')     return job.framework ?? ''
    if (sortKey === 'status')        return job.status ?? ''
    if (sortKey === 'accuracy') {
      return (job.results?.best_accuracy ?? job.results?.best_miou ?? job.results?.best_map ?? -1) as number
    }
    if (sortKey === 'duration') {
      if (!job.started_at || !job.completed_at) return -1
      return (new Date(job.completed_at).getTime() - new Date(job.started_at).getTime()) / 1000
    }
    return ''
  }
  const sorted = [...visible].sort((a, b) => {
    const av = sortVal(a), bv = sortVal(b)
    if (typeof av === 'number' && typeof bv === 'number') return sortDir === 'asc' ? av - bv : bv - av
    return sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av))
  })

  function toggleSort(key: string) {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  function exportCSV() {
    const headers = ['id','architecture','framework','dataset','task','status','accuracy_%','duration_s','started_at']
    const rows = sorted.map(j => [
      j.id.slice(0, 8), j.architecture, j.framework, j.dataset_name, j.task_type, j.status,
      j.results?.best_accuracy != null  ? ((j.results.best_accuracy as number)*100).toFixed(2)
      : j.results?.best_miou   != null  ? ((j.results.best_miou as number)*100).toFixed(2)
      : j.results?.best_map    != null  ? ((j.results.best_map as number)*100).toFixed(2) : '',
      (j.started_at && j.completed_at)
        ? ((new Date(j.completed_at).getTime() - new Date(j.started_at).getTime()) / 1000).toFixed(1) : '',
      j.started_at ?? '',
    ])
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `results_${Date.now()}.csv`
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  function SortIcon({ col }: { col: string }) {
    if (sortKey !== col) return <ChevronDown size={11} className="inline opacity-30" />
    return sortDir === 'asc' ? <ChevronUp size={11} className="inline" /> : <ChevronDown size={11} className="inline" />
  }

  return (
    <div className="p-8 space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <BarChart2 size={22} className="text-brand-400" />
            Results
          </h1>
          <p className="text-sm text-slate-500 mt-1">All training jobs</p>
        </div>
        <div className="flex items-center gap-2">
          {jobs.some(j => j.status === 'running' || j.status === 'pending') && (
            <span className="flex items-center gap-1.5 text-xs text-brand-400">
              <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse" />
              Auto-refreshing
            </span>
          )}
          <Button variant="secondary" size="sm" icon={<Download size={14} />} onClick={exportCSV} disabled={sorted.length === 0}>
            Export CSV
          </Button>
          <Button variant="secondary" size="sm" icon={<RefreshCw size={14} />} onClick={() => load()} loading={loading}>
            Refresh
          </Button>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-2">
        {FILTERS.map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors capitalize ${
              filter === f
                ? 'bg-brand-600 text-white'
                : 'bg-surface-800 text-slate-400 hover:text-white'
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">{error}</div>
      )}

      <Card>
        {visible.length === 0 && !loading ? (
          <div className="py-12 text-center text-slate-500 text-sm">
            {filter === 'all' ? 'No training jobs yet. Start one from the Training page.' : `No ${filter} jobs.`}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-700 text-slate-500 text-left">
                  <th className="pb-3 pr-4 font-medium">Job</th>
                  <th className="pb-3 pr-4 font-medium cursor-pointer hover:text-slate-300" onClick={() => toggleSort('dataset_name')}>Dataset <SortIcon col="dataset_name" /></th>
                  <th className="pb-3 pr-4 font-medium cursor-pointer hover:text-slate-300" onClick={() => toggleSort('architecture')}>Architecture <SortIcon col="architecture" /></th>
                  <th className="pb-3 pr-4 font-medium cursor-pointer hover:text-slate-300" onClick={() => toggleSort('framework')}>Framework <SortIcon col="framework" /></th>
                  <th className="pb-3 pr-4 font-medium cursor-pointer hover:text-slate-300" onClick={() => toggleSort('status')}>Status <SortIcon col="status" /></th>
                  <th className="pb-3 pr-4 font-medium">Task</th>
                  <th className="pb-3 pr-4 font-medium cursor-pointer hover:text-slate-300" onClick={() => toggleSort('accuracy')}>Metric <SortIcon col="accuracy" /></th>
                  <th className="pb-3 pr-4 font-medium cursor-pointer hover:text-slate-300" onClick={() => toggleSort('duration')}>Duration <SortIcon col="duration" /></th>
                  <th className="pb-3 pr-4 font-medium cursor-pointer hover:text-slate-300" onClick={() => toggleSort('created_at')}>Started <SortIcon col="created_at" /></th>
                  <th className="pb-3 font-medium" />
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-800">
                {sorted.map(job => (
                  <tr key={job.id} className="hover:bg-surface-800/40 transition-colors">
                    <td className="py-3 pr-4 text-white font-mono text-xs">{job.id.slice(0, 8)}…</td>
                    <td className="py-3 pr-4 text-slate-300">{job.dataset_name}</td>
                    <td className="py-3 pr-4 text-slate-300">{job.architecture}</td>
                    <td className="py-3 pr-4 text-slate-300 capitalize">{job.framework}</td>
                    <td className="py-3 pr-4"><Badge status={job.status} /></td>
                    <td className="py-3 pr-4">
                      {job.task_type === 'detection' ? (
                        <span className="px-2 py-0.5 rounded-full text-xs bg-orange-500/15 text-orange-300 font-medium">Detection</span>
                      ) : job.task_type === 'segmentation' ? (
                        <span className="px-2 py-0.5 rounded-full text-xs bg-purple-500/15 text-purple-300 font-medium">Segmentation</span>
                      ) : (
                        <span className="px-2 py-0.5 rounded-full text-xs bg-brand-500/15 text-brand-300 font-medium">Classification</span>
                      )}
                    </td>
                    <td className="py-3 pr-4 text-slate-300 font-mono text-xs">
                      {job.task_type === 'detection' && job.results?.best_map != null
                        ? <><span className="text-slate-500 text-xs mr-1">mAP@50</span>{((job.results.best_map as number) * 100).toFixed(1)}%</>
                        : job.task_type === 'segmentation' && job.results?.best_miou != null
                        ? <><span className="text-slate-500 text-xs mr-1">mIoU</span>{((job.results.best_miou as number) * 100).toFixed(1)}%</>
                        : job.results?.best_accuracy != null
                        ? <><span className="text-slate-500 text-xs mr-1">Acc</span>{((job.results.best_accuracy as number) * 100).toFixed(1)}%</>
                        : job.results?.best_miou != null
                        ? <><span className="text-slate-500 text-xs mr-1">mIoU</span>{((job.results.best_miou as number) * 100).toFixed(1)}%</>
                        : job.results?.best_map != null
                        ? <><span className="text-slate-500 text-xs mr-1">mAP</span>{((job.results.best_map as number) * 100).toFixed(1)}%</>
                        : <span className="text-slate-600">—</span>}
                    </td>
                    <td className="py-3 pr-4 text-slate-400">{formatDuration(job.started_at, job.completed_at)}</td>
                    <td className="py-3 pr-4 text-slate-500">{formatDate(job.created_at)}</td>
                    <td className="py-3">
                      <Link href={`/results/${job.id}`}>
                        <Button variant="ghost" size="sm" icon={<ExternalLink size={13} />}>View</Button>
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  )
}
