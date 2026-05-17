'use client'
import { useEffect, useState } from 'react'
import { LayoutDashboard, Cpu, Box, Activity, ArrowRight, Plus } from 'lucide-react'
import Link from 'next/link'
import { getJobs, getModels, getExperiments, getHealth } from '@/lib/api'
import type { TrainingJob, HealthStatus } from '@/types'
import { StatCard } from '@/components/Card'
import Card from '@/components/Card'
import Badge from '@/components/Badge'
import Button from '@/components/Button'
import { formatDate, formatDuration } from '@/lib/utils'

export default function DashboardPage() {
  const [jobs,    setJobs]    = useState<TrainingJob[]>([])
  const [counts,  setCounts]  = useState({ experiments: 0, models: 0 })
  const [health,  setHealth]  = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getJobs({ limit: 10 }), getModels(), getExperiments(), getHealth()])
      .then(([j, m, e, h]) => {
        setJobs(j.jobs)
        setCounts({ experiments: e.total, models: m.total })
        setHealth(h)
      })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

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

      {/* Recent Jobs */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Recent Jobs</h2>
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
