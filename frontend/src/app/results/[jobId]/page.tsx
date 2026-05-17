'use client'
import { useEffect, useRef, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { ArrowLeft, XCircle } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import Badge from '@/components/Badge'
import Button from '@/components/Button'
import Card from '@/components/Card'
import { getJob, cancelJob } from '@/lib/api'
import { formatDate, formatDuration, pct } from '@/lib/utils'
import type { TrainingJob } from '@/types'

function MetaItem({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="space-y-0.5">
      <p className="text-xs text-slate-500 uppercase tracking-wide">{label}</p>
      <p className="text-sm text-white">{value}</p>
    </div>
  )
}

export default function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const router     = useRouter()
  const [job,     setJob]     = useState<TrainingJob | null>(null)
  const [error,   setError]   = useState('')
  const [cancelling, setCancelling] = useState(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  async function load() {
    try {
      const data = await getJob(jobId)
      setJob(data)
      if (data.status !== 'running' && data.status !== 'pending') {
        if (intervalRef.current) clearInterval(intervalRef.current)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load job')
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }

  useEffect(() => {
    load()
    intervalRef.current = setInterval(load, 3000)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId])

  async function handleCancel() {
    setCancelling(true)
    try { await cancelJob(jobId) } catch { /* ignore */ }
    finally { setCancelling(false); load() }
  }

  if (error) return (
    <div className="p-8">
      <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">{error}</div>
    </div>
  )

  if (!job) return (
    <div className="p-8 flex items-center justify-center text-slate-500">Loading…</div>
  )

  const hp = job.hyperparams ?? {}
  const lossData = job.training_history?.map((m, i) => ({
    epoch: m.epoch ?? i + 1, train: m.train_loss, val: m.val_loss ?? null,
  })) ?? []

  const metricKey = job.task_type === 'segmentation' ? 'miou'
                  : job.task_type === 'detection'     ? 'map'
                  : 'accuracy'
  const metricLabel = job.task_type === 'segmentation' ? 'mIoU'
                    : job.task_type === 'detection'     ? 'mAP@50'
                    : 'Accuracy'
  const metricData = job.training_history?.map((m, i) => ({
    epoch: m.epoch ?? i + 1,
    train: metricKey === 'accuracy' ? m.train_accuracy : metricKey === 'miou' ? m.train_miou : null,
    val:   metricKey === 'accuracy' ? m.val_accuracy   : metricKey === 'miou' ? m.val_miou   : null,
  })) ?? []

  const bestMetric = (job.results?.best_accuracy ?? job.results?.best_miou ?? job.results?.best_map) as number | null | undefined

  return (
    <div className="p-8 space-y-6 max-w-4xl">
      {/* Breadcrumb */}
      <div className="flex items-center justify-between">
        <button onClick={() => router.push('/results')} className="flex items-center gap-1 text-slate-500 hover:text-white text-sm transition-colors">
          <ArrowLeft size={14} /> All Results
        </button>
        {(job.status === 'running' || job.status === 'pending') && (
          <Button variant="danger" size="sm" icon={<XCircle size={14} />} loading={cancelling} onClick={handleCancel}>
            Cancel Job
          </Button>
        )}
      </div>

      {/* Header */}
      <div className="flex items-start gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-3">
            <h1 className="text-xl font-bold text-white font-mono">{job.id.slice(0, 16)}…</h1>
            <Badge status={job.status} />
          </div>
          <p className="text-sm text-slate-500 mt-1">
            {job.architecture} · {job.framework} · {job.task_type}
          </p>
        </div>
      </div>

      {/* Error message */}
      {job.error_message && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">
          {job.error_message}
        </div>
      )}

      {/* Metadata */}
      <Card>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <MetaItem label="Dataset"   value={job.dataset_name} />
          <MetaItem label="Epochs"    value={hp.epochs != null ? String(hp.epochs) : '—'} />
          <MetaItem label="Duration"  value={formatDuration(job.started_at, job.completed_at)} />
          <MetaItem label="Started"   value={formatDate(job.created_at)} />
          <MetaItem label="Batch Size" value={hp.batch_size != null ? String(hp.batch_size) : '—'} />
          <MetaItem label="LR"        value={hp.lr != null ? String(hp.lr) : '—'} />
          <MetaItem label={metricLabel} value={bestMetric != null ? pct(bestMetric) : '—'} />
          <MetaItem label="Best Loss" value={job.results?.best_loss != null ? (job.results.best_loss as number).toFixed(4) : '—'} />
        </div>
      </Card>

      {/* Charts */}
      {lossData.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <h3 className="text-sm font-semibold text-slate-300 mb-4">Loss</h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                <Legend />
                <Line type="monotone" dataKey="train" name="Train" stroke="#6366f1" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="val"   name="Val"   stroke="#22d3ee" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {metricData.some(d => d.train != null) && (
            <Card>
              <h3 className="text-sm font-semibold text-slate-300 mb-4">{metricLabel}</h3>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={metricData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                  <Legend />
                  <Line type="monotone" dataKey="train" name="Train" stroke="#6366f1" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="val"   name="Val"   stroke="#22d3ee" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      )}

      {/* Raw results */}
      {job.results && Object.keys(job.results).length > 0 && (
        <Card>
          <h3 className="text-sm font-semibold text-slate-300 mb-4">Full Results</h3>
          <pre className="text-xs text-slate-400 font-mono bg-surface-900 rounded-lg p-4 overflow-x-auto">
            {JSON.stringify(job.results, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  )
}
