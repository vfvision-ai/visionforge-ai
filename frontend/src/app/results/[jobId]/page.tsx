'use client'
import { useEffect, useRef, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { ArrowLeft, XCircle, Loader2, Download, FileText, CheckCircle2, Package } from 'lucide-react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import Badge from '@/components/Badge'
import Button from '@/components/Button'
import Card from '@/components/Card'
import { getJob, cancelJob, downloadModelFile, downloadHistoryCSV, downloadResultsJSON, generateTestSamples } from '@/lib/api'
import { formatDate, formatDuration, pct } from '@/lib/utils'
import type { TrainingJob } from '@/types'

function MetaItem({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="space-y-0.5">
      <p className="text-xs text-slate-500 uppercase tracking-wide">{label}</p>
      <p className="text-sm text-white">{value ?? '—'}</p>
    </div>
  )
}

function ProgressBar({ value, max, label }: { value: number; max: number; label: string }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs text-slate-400">
        <span>{label}</span>
        <span>{value} / {max} epochs ({pct}%)</span>
      </div>
      <div className="h-2 bg-surface-700 rounded-full overflow-hidden">
        <div
          className="h-full bg-brand-500 rounded-full transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export default function JobDetailPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const router     = useRouter()
  const [job,     setJob]     = useState<TrainingJob | null>(null)
  const [error,   setError]   = useState('')
  const [cancelling, setCancelling] = useState(false)
  const [sampleCount,  setSampleCount]  = useState(50)
  const [sampleFmt,    setSampleFmt]    = useState<'png' | 'jpg'>('png')
  const [genSamples,   setGenSamples]   = useState(false)
  const [sampleError,  setSampleError]  = useState('')
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
  const totalEpochs   = (hp.epochs as number) ?? 0
  const doneEpochs    = job.training_history?.length ?? 0
  const isLive        = job.status === 'running' || job.status === 'pending'

  const lossData = job.training_history?.map((m, i) => ({
    epoch: m.epoch ?? i + 1, train: m.train_loss ?? null, val: m.val_loss ?? null,
  })).filter(d => d.train != null) ?? []

  const metricKey   = job.task_type === 'segmentation' ? 'miou'
                    : job.task_type === 'detection'     ? 'map'
                    : 'accuracy'
  const metricLabel = job.task_type === 'segmentation' ? 'mIoU'
                    : job.task_type === 'detection'     ? 'mAP@50'
                    : 'Accuracy'
  const metricData = job.training_history?.map((m, i) => ({
    epoch: m.epoch ?? i + 1,
    train: metricKey === 'accuracy' ? (m.train_accuracy ?? null) : metricKey === 'miou' ? (m.train_miou ?? null) : null,
    val:   metricKey === 'accuracy' ? (m.val_accuracy ?? null)   : metricKey === 'miou' ? (m.val_miou ?? null)   : null,
  })).filter(d => d.train != null || d.val != null) ?? []

  const lastMetrics   = job.training_history?.[job.training_history.length - 1]
  const bestMetric    = (job.results?.best_accuracy ?? job.results?.best_miou ?? job.results?.best_map) as number | null | undefined
  const latestMetric  = lastMetrics
    ? (metricKey === 'accuracy' ? (lastMetrics.val_accuracy ?? lastMetrics.train_accuracy)
     : metricKey === 'miou'     ? (lastMetrics.val_miou ?? lastMetrics.train_miou)
     : null) as number | null
    : null

  return (
    <div className="p-8 space-y-6 max-w-4xl">
      {/* Breadcrumb */}
      <div className="flex items-center justify-between">
        <button onClick={() => router.push('/results')} className="flex items-center gap-1 text-slate-500 hover:text-white text-sm transition-colors">
          <ArrowLeft size={14} /> All Results
        </button>
        {isLive && (
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
            {isLive && <Loader2 size={15} className="text-brand-400 animate-spin" />}
          </div>
          <p className="text-sm text-slate-500 mt-1">
            {job.architecture} · {job.framework} · {job.task_type}
          </p>
        </div>
      </div>

      {/* Completion banner */}
      {job.status === 'completed' && (
        <div className="flex items-center gap-4 p-4 rounded-xl bg-green-500/10 border border-green-500/20">
          <CheckCircle2 size={22} className="text-green-400 shrink-0" />
          <div className="flex-1">
            <p className="text-sm font-semibold text-green-300">Training complete!</p>
            <p className="text-xs text-green-400/70 mt-0.5">
              {bestMetric != null ? `Best ${metricLabel}: ${pct(bestMetric)}` : 'Model saved successfully'}
            </p>
          </div>
          <div className="flex gap-2 flex-wrap">
            {job.model_path && (
              <Button size="sm" icon={<Download size={13} />} onClick={() => downloadModelFile(jobId)}>
                Download Model
              </Button>
            )}
            <Button variant="secondary" size="sm" icon={<FileText size={13} />} onClick={() => downloadHistoryCSV(jobId)}>
              Export CSV
            </Button>
            <Button variant="secondary" size="sm" icon={<FileText size={13} />} onClick={() => downloadResultsJSON(jobId)}>
              Results JSON
            </Button>
          </div>
        </div>
      )}

      {/* Error message */}
      {job.error_message && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">
          {job.error_message}
        </div>
      )}

      {/* Live progress bar */}
      {isLive && totalEpochs > 0 && (
        <Card>
          <ProgressBar value={doneEpochs} max={totalEpochs} label="Training Progress" />
          {lastMetrics && (
            <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
              {lastMetrics.train_loss != null && (
                <div className="bg-surface-900 rounded-lg p-3 text-center">
                  <p className="text-xs text-slate-500">Train Loss</p>
                  <p className="text-sm font-mono text-white mt-0.5">{(lastMetrics.train_loss as number).toFixed(4)}</p>
                </div>
              )}
              {lastMetrics.val_loss != null && (
                <div className="bg-surface-900 rounded-lg p-3 text-center">
                  <p className="text-xs text-slate-500">Val Loss</p>
                  <p className="text-sm font-mono text-white mt-0.5">{(lastMetrics.val_loss as number).toFixed(4)}</p>
                </div>
              )}
              {latestMetric != null && (
                <div className="bg-surface-900 rounded-lg p-3 text-center">
                  <p className="text-xs text-slate-500">{metricLabel}</p>
                  <p className="text-sm font-mono text-brand-400 mt-0.5">{pct(latestMetric)}</p>
                </div>
              )}
              <div className="bg-surface-900 rounded-lg p-3 text-center">
                <p className="text-xs text-slate-500">Epoch</p>
                <p className="text-sm font-mono text-white mt-0.5">{doneEpochs} / {totalEpochs}</p>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Metadata */}
      <Card>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <MetaItem label="Dataset"    value={job.dataset_name} />
          <MetaItem label="Epochs"     value={totalEpochs > 0 ? String(totalEpochs) : undefined} />
          <MetaItem label="Duration"   value={formatDuration(job.started_at, job.completed_at)} />
          <MetaItem label="Started"    value={formatDate(job.created_at)} />
          <MetaItem label="Batch Size" value={(hp.batch_size as number) != null ? String(hp.batch_size) : undefined} />
          <MetaItem label="LR"         value={(hp.lr as number) != null ? String(hp.lr) : undefined} />
          <MetaItem label={metricLabel} value={bestMetric != null ? pct(bestMetric) : (latestMetric != null ? pct(latestMetric) : undefined)} />
          <MetaItem label="Best Loss"  value={job.results?.best_loss != null ? (job.results.best_loss as number).toFixed(4) : (lastMetrics?.val_loss != null ? (lastMetrics.val_loss as number).toFixed(4) : undefined)} />
        </div>
      </Card>

      {/* Charts — shown as soon as any history is available */}
      {lossData.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <h3 className="text-sm font-semibold text-slate-300 mb-4">Loss{isLive ? ' (live)' : ''}</h3>
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={lossData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} />
                <Legend />
                <Line type="monotone" dataKey="train" name="Train" stroke="#6366f1" dot={false} strokeWidth={2} isAnimationActive={false} />
                <Line type="monotone" dataKey="val"   name="Val"   stroke="#22d3ee" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {metricData.length > 0 && (
            <Card>
              <h3 className="text-sm font-semibold text-slate-300 mb-4">{metricLabel}{isLive ? ' (live)' : ''}</h3>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={metricData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="epoch" stroke="#64748b" tick={{ fontSize: 11 }} />
                  <YAxis stroke="#64748b" tick={{ fontSize: 11 }} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8 }} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
                  <Legend />
                  <Line type="monotone" dataKey="train" name="Train" stroke="#6366f1" dot={false} strokeWidth={2} isAnimationActive={false} />
                  <Line type="monotone" dataKey="val"   name="Val"   stroke="#22d3ee" dot={false} strokeWidth={2} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </div>
      )}

      {/* Pending/running placeholder when no history yet */}
      {isLive && lossData.length === 0 && (
        <Card>
          <div className="py-10 flex flex-col items-center gap-3 text-slate-500">
            <Loader2 size={28} className="animate-spin text-brand-500" />
            <p className="text-sm">Waiting for first epoch to complete…</p>
          </div>
        </Card>
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

      {/* Test Samples for Evaluation */}
      {job.status === 'completed' && (
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Package size={15} className="text-brand-400" />
            <h3 className="text-sm font-semibold text-slate-300">Test Samples for Evaluation</h3>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Extract labelled images from the training dataset. Downloads a <code className="text-slate-300">.zip</code> with
            images and a <code className="text-slate-300">labels.csv</code> manifest — ready for inference testing.
          </p>
          <div className="flex flex-wrap items-end gap-4">
            {/* Count */}
            <div>
              <label className="block text-xs text-slate-400 mb-1.5">
                Number of samples: <span className="text-brand-400 font-mono">{sampleCount}</span>
              </label>
              <input
                title="Number of test samples"
                type="range" min={1} max={500} step={1}
                value={sampleCount}
                onChange={e => setSampleCount(Number(e.target.value))}
                className="w-44 accent-brand-500"
              />
              <div className="flex justify-between text-xs text-slate-600 mt-0.5">
                <span>1</span><span>500</span>
              </div>
            </div>
            {/* Format */}
            <div>
              <p className="text-xs text-slate-400 mb-1.5">Image format</p>
              <div className="flex gap-1">
                {(['png', 'jpg'] as const).map(f => (
                  <button key={f} onClick={() => setSampleFmt(f)}
                    className={`px-3 py-1.5 rounded-md text-xs border transition-colors ${
                      sampleFmt === f
                        ? 'border-brand-500 bg-brand-500/10 text-white'
                        : 'border-surface-600 text-slate-400 hover:text-white'
                    }`}>
                    .{f.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            {/* Button */}
            <Button
              icon={genSamples ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
              loading={genSamples}
              onClick={async () => {
                setSampleError('')
                setGenSamples(true)
                try {
                  await generateTestSamples(jobId, sampleCount, sampleFmt)
                } catch (e: unknown) {
                  setSampleError(e instanceof Error ? e.message : 'Failed to generate samples')
                } finally {
                  setGenSamples(false)
                }
              }}
            >
              {genSamples ? 'Generating…' : `Generate & Download (${sampleCount} samples)`}
            </Button>
          </div>
          {sampleError && (
            <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400">
              {sampleError}
            </div>
          )}
        </Card>
      )}
    </div>
  )
}
