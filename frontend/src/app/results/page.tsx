'use client'
import { useEffect, useState } from 'react'
import Link from 'next/link'
import { BarChart2, RefreshCw, ExternalLink } from 'lucide-react'
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
                  <th className="pb-3 pr-4 font-medium">Dataset</th>
                  <th className="pb-3 pr-4 font-medium">Architecture</th>
                  <th className="pb-3 pr-4 font-medium">Framework</th>
                  <th className="pb-3 pr-4 font-medium">Status</th>
                  <th className="pb-3 pr-4 font-medium">Accuracy</th>
                  <th className="pb-3 pr-4 font-medium">Duration</th>
                  <th className="pb-3 pr-4 font-medium">Started</th>
                  <th className="pb-3 font-medium" />
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-800">
                {visible.map(job => (
                  <tr key={job.id} className="hover:bg-surface-800/40 transition-colors">
                    <td className="py-3 pr-4 text-white font-mono text-xs">{job.id.slice(0, 8)}…</td>
                    <td className="py-3 pr-4 text-slate-300">{job.dataset_name}</td>
                    <td className="py-3 pr-4 text-slate-300">{job.architecture}</td>
                    <td className="py-3 pr-4 text-slate-300 capitalize">{job.framework}</td>
                    <td className="py-3 pr-4"><Badge status={job.status} /></td>
                    <td className="py-3 pr-4 text-slate-300">
                      {job.results?.best_accuracy != null
                        ? `${((job.results.best_accuracy as number) * 100).toFixed(1)}%`
                        : job.results?.best_miou != null
                        ? `mIoU ${((job.results.best_miou as number) * 100).toFixed(1)}%`
                        : job.results?.best_map != null
                        ? `mAP ${((job.results.best_map as number) * 100).toFixed(1)}%`
                        : '—'}
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
