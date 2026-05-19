'use client'
import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { LayoutDashboard, Cpu, Activity, ArrowRight, Database, CheckCircle2, XCircle, Loader2, Server, Package, TrendingUp, ShieldCheck, Users } from 'lucide-react'
import Link from 'next/link'
import { getJobs, getModels, getExperiments, getHealth, getSystemInfo, getAdminUsers } from '@/lib/api'
import type { TrainingJob, HealthStatus, SystemInfo } from '@/types'
import { StatCard } from '@/components/Card'
import Card from '@/components/Card'
import Badge from '@/components/Badge'
import Button from '@/components/Button'
import { formatDate, formatDuration } from '@/lib/utils'
import { useAuth } from '@/contexts/AuthContext'

interface Toast { id: number; type: 'success' | 'error' | 'info'; message: string }

export default function DashboardPage() {
  const router = useRouter()
  const { user } = useAuth()
  const isAdmin = user?.role?.toLowerCase() === 'admin'
  const [jobs,       setJobs]       = useState<TrainingJob[]>([])
  const [counts,     setCounts]     = useState({ experiments: 0, models: 0 })
  const [health,     setHealth]     = useState<HealthStatus | null>(null)
  const [sysInfo,    setSysInfo]    = useState<SystemInfo | null>(null)
  const [sysLoading, setSysLoading] = useState(true)
  const [loading,    setLoading]    = useState(true)
  const [toasts,     setToasts]     = useState<Toast[]>([])
  const [userCount,  setUserCount]  = useState<number | null>(null)
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
      const [j, m, e, h] = await Promise.all([getJobs({ limit: 1000 }), getModels(), getExperiments(), getHealth()])
      setJobs(j.jobs)
      setCounts({ experiments: e.total, models: m.total })
      setHealth(h)
      setSysLoading(true)
      getSystemInfo().then(s => { setSysInfo(s); setSysLoading(false) }).catch(() => setSysLoading(false))
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

  // Fetch user count whenever admin status is confirmed (auth may resolve after mount)
  useEffect(() => {
    if (isAdmin) {
      getAdminUsers(0, 1).then(u => setUserCount(u.total)).catch(() => {})
    }
  }, [isAdmin]) // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-refresh every 5s while any job is active
  useEffect(() => {
    const active = jobs.some(j => j.status === 'running' || j.status === 'pending')
    if (!active) return
    const id = setInterval(() => load(true), 5000)
    return () => clearInterval(id)
  }, [jobs]) // eslint-disable-line react-hooks/exhaustive-deps

  const running   = jobs.filter(j => j.status === 'running').length
  const completed = jobs.filter(j => j.status === 'completed').length
  const failed    = jobs.filter(j => j.status === 'failed').length
  const bestAcc   = jobs
    .filter(j => j.status === 'completed' && j.results?.best_accuracy != null)
    .reduce((best, j) => Math.max(best, (j.results!.best_accuracy as number) * 100), 0)

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <LayoutDashboard size={22} className="text-brand-400" />
            Dashboard
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Welcome back,{' '}
            <span className="text-white">{user?.full_name ?? 'User'}</span>
            {isAdmin && (
              <span className="ml-2 inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-amber-900/50 text-amber-300 border border-amber-700">
                <ShieldCheck size={10} />
                Administrator
              </span>
            )}
          </p>
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

      {/* Admin Panel — only visible to admins */}
      {isAdmin && (
        <Card className="border-amber-700/40 bg-amber-950/10">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-amber-300 flex items-center gap-2">
              <ShieldCheck size={15} />
              Admin Overview — you can see all users&apos; data
            </h2>
            <Link href="/admin/users">
              <button className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium
                                 text-amber-300 border border-amber-700 hover:bg-amber-700/20 transition-colors">
                <Users size={12} />
                Manage Users
              </button>
            </Link>
          </div>
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="rounded-lg p-3 bg-black/20">
              <p className="text-xs text-amber-400/70">Registered Users</p>
              <p className="text-xl font-bold text-amber-300 mt-0.5">{userCount ?? '…'}</p>
            </div>
            <div className="rounded-lg p-3 bg-black/20">
              <p className="text-xs text-amber-400/70">Total Jobs (all users)</p>
              <p className="text-xl font-bold text-amber-300 mt-0.5">{loading ? '…' : jobs.length}</p>
            </div>
            <div className="rounded-lg p-3 bg-black/20">
              <p className="text-xs text-amber-400/70">Saved Models (all users)</p>
              <p className="text-xl font-bold text-amber-300 mt-0.5">{loading ? '…' : counts.models}</p>
            </div>
            <div className="rounded-lg p-3 bg-black/20">
              <p className="text-xs text-amber-400/70">Experiments (all users)</p>
              <p className="text-xl font-bold text-amber-300 mt-0.5">{loading ? '…' : counts.experiments}</p>
            </div>
          </div>
        </Card>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-4">
        <StatCard label="Total Jobs"    value={loading ? '…' : jobs.length}           icon={<Cpu size={18} />}          color="text-brand-400" />
        <StatCard label="Running"       value={loading ? '…' : running}               icon={<Activity size={18} />}     color="text-blue-400"  sub="active training" />
        <StatCard label="Completed"     value={loading ? '…' : completed}             icon={<CheckCircle2 size={18} />} color="text-green-400" />
        <StatCard label="Failed"        value={loading ? '…' : failed}                icon={<XCircle size={18} />}      color={failed > 0 ? 'text-red-400' : 'text-slate-600'} />
        <StatCard label="Saved Models"  value={loading ? '…' : counts.models}         icon={<Package size={18} />}      color="text-purple-400" />
        <StatCard label="Best Accuracy" value={loading ? '…' : bestAcc > 0 ? `${bestAcc.toFixed(1)}%` : '—'} icon={<TrendingUp size={18} />} color="text-yellow-400" />
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {[
          { href: '/training',  label: 'Start Training',   desc: 'Launch a new training job',           icon: <Cpu size={20} />,        color: 'text-brand-400' },
          { href: '/dataset',   label: 'Upload Dataset',   desc: 'Analyze your dataset',                icon: <Database size={20} />,   color: 'text-green-400' },
          { href: '/inference', label: 'Run Inference',    desc: 'Test a trained model',                icon: <Activity size={20} />,   color: 'text-yellow-400' },
          { href: '/models',    label: 'View Models',      desc: 'Browse saved model versions',         icon: <Package size={20} />,    color: 'text-purple-400' },
          { href: '/results',   label: 'View Results',     desc: 'Browse all training jobs',            icon: <TrendingUp size={20} />, color: 'text-sky-400' },
        ].map(a => (
          <Link key={a.href} href={a.href}>
            <Card className="hover:border-brand-500/40 transition-all group h-full">
              <div className={`mb-3 ${a.color}`}>{a.icon}</div>
              <p className="text-sm font-semibold text-white group-hover:text-brand-300 transition-colors">{a.label}</p>
              <p className="text-xs text-slate-500 mt-1">{a.desc}</p>
              <ArrowRight size={14} className="mt-3 text-slate-600 group-hover:text-brand-400 transition-colors" />
            </Card>
          </Link>
        ))}
      </div>

      {/* System Status */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
          <Server size={15} className="text-slate-400" /> System Status
        </h2>
        {sysLoading ? (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[0, 1, 2, 3].map(i => (
              <div key={i} className="flex items-center gap-2 bg-surface-900 rounded-lg p-3 animate-pulse">
                <span className="w-2 h-2 rounded-full bg-slate-700 shrink-0" />
                <div className="space-y-1.5 flex-1">
                  <div className="h-2.5 w-14 bg-slate-700 rounded" />
                  <div className="h-2.5 w-20 bg-slate-700 rounded" />
                </div>
              </div>
            ))}
          </div>
        ) : sysInfo ? (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                { label: 'PyTorch',    ok: !!sysInfo.pytorch,        detail: sysInfo.pytorch ?? 'not installed' },
                { label: 'CUDA',       ok: !!sysInfo.cuda_available,  detail: sysInfo.cuda_available ? (sysInfo.gpu_name ?? sysInfo.cuda_version ?? 'available') : 'CPU only' },
                { label: 'TensorFlow', ok: !!sysInfo.tensorflow,      detail: sysInfo.tensorflow ?? 'not installed' },
                { label: 'Optuna',     ok: !!sysInfo.optuna,          detail: sysInfo.optuna ?? 'not installed' },
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
              <div className="mt-3 flex items-center gap-3">
                <p className="text-xs text-slate-500 shrink-0">
                  RAM: {sysInfo.ram_used_gb?.toFixed(1)} / {sysInfo.ram_total_gb.toFixed(1)} GB
                </p>
                <div className="flex-1 h-1.5 bg-surface-700 rounded-full overflow-hidden max-w-[180px]">
                  <div
                    className="h-full bg-brand-500 rounded-full transition-all"
                    style={{ width: `${Math.min(100, ((sysInfo.ram_used_gb ?? 0) / sysInfo.ram_total_gb) * 100)}%` }}
                  />
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-xs text-slate-500">System info unavailable.</p>
        )}
      </Card>

        {/* Recent Jobs — latest 10 */}
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
                  <th className="text-left px-5 py-3">Accuracy</th>
                  <th className="text-left px-5 py-3">Duration</th>
                  <th className="text-left px-5 py-3">Started</th>
                </tr>
              </thead>
              <tbody>
                {jobs.slice(0, 10).map((job, i) => {
                  const acc = job.results?.best_accuracy ?? job.results?.best_miou ?? job.results?.best_map
                  return (
                    <tr
                      key={job.id}
                      onClick={() => router.push(`/results/${job.id}`)}
                      className={`border-b border-surface-700 hover:bg-surface-700 transition-colors cursor-pointer ${i === Math.min(jobs.length, 10) - 1 ? 'border-b-0' : ''}`}
                    >
                      <td className="px-5 py-3 text-white font-medium">{job.dataset_name}</td>
                      <td className="px-5 py-3 text-slate-400">{job.architecture}</td>
                      <td className="px-5 py-3 text-slate-400 capitalize">{job.framework}</td>
                      <td className="px-5 py-3"><Badge variant={job.status}>{job.status}</Badge></td>
                      <td className="px-5 py-3 font-mono text-xs">
                        {acc != null ? (
                          <span className="text-green-400">{((acc as number) * 100).toFixed(1)}%</span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                      <td className="px-5 py-3 text-slate-400 font-mono text-xs">{formatDuration(job.started_at, job.completed_at)}</td>
                      <td className="px-5 py-3 text-slate-500 text-xs">{formatDate(job.created_at)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </Card>
      </div>
    </div>
  )
}
