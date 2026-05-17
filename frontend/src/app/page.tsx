'use client'
import { useEffect, useRef, useState } from 'react'
import { LayoutDashboard, Cpu, Box, Activity, ArrowRight, Plus, CheckCircle2, XCircle, Loader2, Server } from 'lucide-react'
import Link from 'next/link'
import { getJobs, getModels, getExperiments, getHealth, getSystemInfo } from '@/lib/api'
import type { TrainingJob, HealthStatus, SystemInfo } from '@/types'
import { StatCard } from '@/components/Card'
import Card from '@/components/Card'
import Badge from '@/components/Badge'
import Button from '@/components/Button'
import { formatDate, formatDuration } from '@/lib/utils'

interface Toast { id: number; type: 'success' | 'error' | 'info'; message: string }

export default function DashboardPage() {
  const [jobs,    setJobs]    = useState<TrainingJob[]>([])
  const [counts,  setCounts]  = useState({ experiments: 0, models: 0 })
  const [health,  setHealth]  = useState<HealthStatus | null>(null)
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [toasts,  setToasts]  = useState<Toast[]>([])
  const prevStatuses = useRef<Record<string, string>>({})
  const toastId      = useRef(0)

  function addToast(type: Toast['type'], message: string) {
    const id = ++toastId.current
    setToasts(t => [...t, { id, type, message }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 5000)
  }

  async function load(quiet = false) {
    if (!quiet) setLoading(true)
    try {
      const [j, m, e, h] = await Promise.all([getJobs({ limit: 10 }), getModels(), getExperiments(), getHealth()])
      setJobs(j.jobs)
      setCounts({ experiments: e.total, models: m.total })
      setHealth(h)
      getSystemInfo().then(setSysInfo).catch(() => {})
      // Detect status transitions and notify
      j.jobs.forEach(job => {
        const prev = prevStatuses.current[job.id]
        if (prev && prev !== job.status) {
          if (job.status === 'completed')
            addToast('success', `✅ Training complete — ${job.architecture} on ${job.dataset_name}`)
          else if (job.status === 'failed')
            addToast('error', `❌ Training failed — ${job.architecture}: ${job.error_message ?? 'unknown error'}`)
          else if (job.status === 'running' && prev === 'pending')
            addToast('info', `🚀 Training started — ${job.architecture} on ${job.dataset_name}`)
        }
        prevStatuses.current[job.id] = job.status
      })
    } catch { /* silent */ } finally {
      if (!quiet) setLoading(false)
    }
  }

  // Initial load
  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-refresh every 5s while any job is active
  useEffect(() => {
    const active = jobs.some(j => j.status === 'running' || j.status === 'pending')
    if (!active) return
    const id = setInterval(() => load(true), 5000)
    return () => clearInterval(id)
  }, [jobs]) // eslint-disable-line react-hooks/exhaustive-deps

  const running   = jobs.filter(j => j.status === 'running').length
  const completed = jobs.filter(j => j.status === 'completed').length

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <LayoutDashboard size={22} className="text-brand-400" />
            Dashboard
          </h1>
          <p className="text-sm text-slate-500 mt-1">Welcome to VisionForge</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${health?.status === 'ok' ? 'bg-green-400' : 'bg-yellow-400'}`} />
          <span className="text-xs text-slate-500">
            {health ? `System ${health.status}` : 'Checking…'}
          </span>
        </div>
      </div>

      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 space-y-2 w-80">
        {toasts.map(t => (
          <div key={t.id} className={`flex items-start gap-3 px-4 py-3 rounded-lg shadow-lg text-sm backdrop-blur border transition-all ${
            t.type === 'success' ? 'bg-green-500/10 border-green-500/30 text-green-300' :
            t.type === 'error'   ? 'bg-red-500/10 border-red-500/30 text-red-300' :
                                   'bg-brand-500/10 border-brand-500/30 text-brand-300'
          }`}>
            {t.type === 'success' ? <CheckCircle2 size={16} className="mt-0.5 shrink-0" /> :
             t.type === 'error'   ? <XCircle size={16} className="mt-0.5 shrink-0" /> :
                                    <Loader2 size={16} className="mt-0.5 shrink-0 animate-spin" />}
            <span className="flex-1">{t.message}</span>
          </div>
        ))}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard label="Total Jobs"         value={loading ? '…' : jobs.length}          icon={<Cpu size={18} />}      color="text-brand-400" />
        <StatCard label="Running"            value={loading ? '…' : running}              icon={<Activity size={18} />} color="text-blue-400"  sub="active training" />
        <StatCard label="Completed"          value={loading ? '…' : completed}            icon={<Box size={18} />}      color="text-green-400" />
        <StatCard label="Saved Models"       value={loading ? '…' : counts.models}        icon={<Box size={18} />}      color="text-purple-400" />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { href: '/training', label: 'Start Training',   desc: 'Launch a new training job',      icon: <Cpu size={20} />,      color: 'text-brand-400' },
          { href: '/dataset',  label: 'Upload Dataset',   desc: 'Analyze your dataset',           icon: <Plus size={20} />,     color: 'text-green-400' },
          { href: '/inference',label: 'Run Inference',    desc: 'Test a trained model',           icon: <Activity size={20} />, color: 'text-yellow-400' },
        ].map(a => (
          <Link key={a.href} href={a.href}>
            <Card className="hover:border-brand-500/40 transition-all group">
              <div className={`mb-3 ${a.color}`}>{a.icon}</div>
              <p className="text-sm font-semibold text-white group-hover:text-brand-300 transition-colors">{a.label}</p>
              <p className="text-xs text-slate-500 mt-1">{a.desc}</p>
              <ArrowRight size={14} className="mt-3 text-slate-600 group-hover:text-brand-400 transition-colors" />
            </Card>
          </Link>
        ))}
      </div>

      {/* System Status */}
      {sysInfo && (
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
            <Server size={15} className="text-slate-400" /> System Status
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: 'PyTorch',    ok: !!sysInfo.pytorch_version,    detail: sysInfo.pytorch_version ?? 'not installed' },
              { label: 'CUDA',       ok: !!sysInfo.cuda_available,     detail: sysInfo.cuda_available ? (sysInfo.gpu_name ?? sysInfo.cuda_version ?? 'available') : 'CPU only' },
              { label: 'TensorFlow', ok: !!sysInfo.tensorflow_version, detail: sysInfo.tensorflow_version ?? 'not installed' },
              { label: 'Optuna',     ok: !!sysInfo.optuna_version,     detail: sysInfo.optuna_version ?? 'not installed' },
            ].map(s => (
              <div key={s.label} className="flex items-center gap-2 bg-surface-900 rounded-lg p-3">
                <span className={`w-2 h-2 rounded-full shrink-0 ${s.ok ? 'bg-green-400' : 'bg-slate-600'}`} />
                <div className="min-w-0">
                  <p className="text-xs font-medium text-white">{s.label}</p>
                  <p className="text-xs text-slate-500 truncate">{s.detail}</p>
                </div>
              </div>
            ))}
          </div>
          {sysInfo.ram_total_gb && (
            <div className="mt-3 text-xs text-slate-500">
              RAM: {sysInfo.ram_used_gb?.toFixed(1)} / {sysInfo.ram_total_gb.toFixed(1)} GB used
            </div>
          )}
        </Card>
      )}

      {/* Recent Jobs */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Recent Jobs</h2>
            {jobs.some(j => j.status === 'running' || j.status === 'pending') && (
              <span className="flex items-center gap-1.5 text-xs text-brand-400">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse" />
                Live
              </span>
            )}
          </div>
          <Link href="/results">
            <Button variant="ghost" size="sm">View all <ArrowRight size={12} /></Button>
          </Link>
        </div>
        <Card className="overflow-hidden p-0">
          {loading ? (
            <div className="p-8 text-center text-slate-500 text-sm">Loading…</div>
          ) : jobs.length === 0 ? (
            <div className="p-8 text-center text-slate-500 text-sm">
              No jobs yet. <Link href="/training" className="text-brand-400 hover:underline">Start training →</Link>
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead className="border-b border-surface-600">
                <tr className="text-xs text-slate-500 uppercase tracking-wider">
                  <th className="text-left px-5 py-3">Dataset</th>
                  <th className="text-left px-5 py-3">Architecture</th>
                  <th className="text-left px-5 py-3">Framework</th>
                  <th className="text-left px-5 py-3">Status</th>
                  <th className="text-left px-5 py-3">Duration</th>
                  <th className="text-left px-5 py-3">Started</th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job, i) => (
                  <tr key={job.id} className={`border-b border-surface-700 hover:bg-surface-700 transition-colors ${i === jobs.length - 1 ? 'border-b-0' : ''}`}>
                    <td className="px-5 py-3 text-white font-medium">{job.dataset_name}</td>
                    <td className="px-5 py-3 text-slate-400">{job.architecture}</td>
                    <td className="px-5 py-3 text-slate-400 capitalize">{job.framework}</td>
                    <td className="px-5 py-3"><Badge variant={job.status}>{job.status}</Badge></td>
                    <td className="px-5 py-3 text-slate-400 font-mono text-xs">{formatDuration(job.started_at, job.completed_at)}</td>
                    <td className="px-5 py-3 text-slate-500 text-xs">{formatDate(job.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
      </div>
    </div>
  )
}
