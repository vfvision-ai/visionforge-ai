'use client'
import { useEffect, useState } from 'react'
import { Package, CheckCircle2, RefreshCw, Download, Trash2 } from 'lucide-react'
import Card from '@/components/Card'
import Badge from '@/components/Badge'
import Button from '@/components/Button'
import { Select } from '@/components/FormControls'
import { getModels, promoteModel, deleteModel } from '@/lib/api'
import { formatDate } from '@/lib/utils'
import type { ModelVersion } from '@/types'

const FW_OPTS = [
  { value: '', label: 'All Frameworks' }, { value: 'pytorch', label: 'PyTorch' },
  { value: 'tensorflow', label: 'TensorFlow' }, { value: 'sklearn', label: 'Scikit-learn' },
]
const TASK_OPTS = [
  { value: '', label: 'All Tasks' }, { value: 'classification', label: 'Classification' },
  { value: 'detection', label: 'Detection' }, { value: 'segmentation', label: 'Segmentation' },
]

export default function ModelsPage() {
  const [models,    setModels]    = useState<ModelVersion[]>([])
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState('')
  const [promoting, setPromoting] = useState<string | null>(null)
  const [deleting,  setDeleting]  = useState<string | null>(null)
  const [fwFilter,  setFwFilter]  = useState('')
  const [taskFilter,setTaskFilter]= useState('')

  async function load() {
    setLoading(true); setError('')
    try { setModels((await getModels()).models) }
    catch (e: unknown) { setError(e instanceof Error ? e.message : 'Failed to load models') }
    finally { setLoading(false) }
  }

  async function handlePromote(id: string) {
    setPromoting(id)
    try { await promoteModel(id); await load() }
    catch (e: unknown) { setError(e instanceof Error ? e.message : 'Promote failed') }
    finally { setPromoting(null) }
  }

  async function handleDelete(m: ModelVersion) {
    if (!confirm(`Delete model "${m.name}"? This cannot be undone.`)) return
    setDeleting(m.id)
    try { await deleteModel(m.id); await load() }
    catch (e: unknown) { setError(e instanceof Error ? e.message : 'Delete failed') }
    finally { setDeleting(null) }
  }

  function handleDownload(id: string) {
    window.open(`/api/v1/models/${id}/download`, '_blank')
  }

  useEffect(() => { load() }, [])

  const visible = models
    .filter(m => !fwFilter   || m.framework === fwFilter)
    .filter(m => !taskFilter || m.task_type === taskFilter)

  return (
    <div className="p-8 space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Package size={22} className="text-brand-400" /> Models
          </h1>
          <p className="text-sm text-slate-500 mt-1">{models.length} saved model version{models.length !== 1 ? 's' : ''}</p>
        </div>
        <Button variant="secondary" size="sm" icon={<RefreshCw size={14} />} onClick={load} loading={loading}>Refresh</Button>
      </div>

      {error && <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">{error}</div>}

      {/* Filters */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-500">Filter:</span>
        <div className="w-44"><Select label="" value={fwFilter}   onChange={e => setFwFilter(e.target.value)}   options={FW_OPTS}   /></div>
        <div className="w-44"><Select label="" value={taskFilter} onChange={e => setTaskFilter(e.target.value)} options={TASK_OPTS} /></div>
        {(fwFilter || taskFilter) && (
          <button onClick={() => { setFwFilter(''); setTaskFilter('') }} className="text-xs text-slate-500 hover:text-white">Clear</button>
        )}
      </div>

      <Card>
        {visible.length === 0 && !loading ? (
          <div className="py-12 text-center text-slate-500 text-sm">
            {models.length === 0 ? 'No saved models yet. Models are saved automatically when training completes.' : 'No models match the current filters.'}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-700 text-slate-500 text-left">
                  <th className="pb-3 pr-4 font-medium">Name</th>
                  <th className="pb-3 pr-4 font-medium">Architecture</th>
                  <th className="pb-3 pr-4 font-medium">Framework</th>
                  <th className="pb-3 pr-4 font-medium">Task</th>
                  <th className="pb-3 pr-4 font-medium">Val Accuracy</th>
                  <th className="pb-3 pr-4 font-medium">Status</th>
                  <th className="pb-3 pr-4 font-medium">Created</th>
                  <th className="pb-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-800">
                {visible.map(m => (
                  <tr key={m.id} className="hover:bg-surface-800/40 transition-colors">
                    <td className="py-3 pr-4 text-white font-medium">{m.name}</td>
                    <td className="py-3 pr-4 text-slate-300">{m.architecture}</td>
                    <td className="py-3 pr-4 text-slate-300 capitalize">{m.framework}</td>
                    <td className="py-3 pr-4 text-slate-300 capitalize">{m.task_type}</td>
                    <td className="py-3 pr-4 text-slate-300">{m.val_accuracy != null ? `${(m.val_accuracy * 100).toFixed(1)}%` : '—'}</td>
                    <td className="py-3 pr-4">
                      {m.is_production
                        ? <span className="inline-flex items-center gap-1 text-xs text-green-400 font-medium"><CheckCircle2 size={13} /> Production</span>
                        : <Badge status="default" text="Saved" />}
                    </td>
                    <td className="py-3 pr-4 text-slate-500">{formatDate(m.created_at)}</td>
                    <td className="py-3">
                      <div className="flex items-center gap-1">
                        <button title="Download model file" onClick={() => handleDownload(m.id)}
                          className="p-1.5 rounded text-slate-500 hover:text-brand-400 hover:bg-brand-500/10 transition-colors">
                          <Download size={13} />
                        </button>
                        {!m.is_production && (
                          <Button variant="secondary" size="sm" loading={promoting === m.id} onClick={() => handlePromote(m.id)}>Promote</Button>
                        )}
                        <button title="Delete" onClick={() => handleDelete(m)} disabled={deleting === m.id}
                          className="p-1.5 rounded text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-40">
                          <Trash2 size={13} />
                        </button>
                      </div>
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

import Card from '@/components/Card'
import Badge from '@/components/Badge'
import Button from '@/components/Button'
import { getModels, promoteModel } from '@/lib/api'
import { formatDate } from '@/lib/utils'
import type { ModelVersion } from '@/types'

export default function ModelsPage() {
  const [models,  setModels]  = useState<ModelVersion[]>([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState('')
  const [promoting, setPromoting] = useState<string | null>(null)

  async function load() {
    setLoading(true); setError('')
    try { setModels((await getModels()).models) }
    catch (e: unknown) { setError(e instanceof Error ? e.message : 'Failed to load models') }
    finally { setLoading(false) }
  }

  async function handlePromote(id: string) {
    setPromoting(id)
    try { await promoteModel(id); await load() }
    catch (e: unknown) { setError(e instanceof Error ? e.message : 'Promote failed') }
    finally { setPromoting(null) }
  }

  useEffect(() => { load() }, [])

  return (
    <div className="p-8 space-y-6 max-w-5xl">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Package size={22} className="text-brand-400" />
            Models
          </h1>
          <p className="text-sm text-slate-500 mt-1">Saved model versions</p>
        </div>
        <Button variant="secondary" size="sm" icon={<RefreshCw size={14} />} onClick={load} loading={loading}>
          Refresh
        </Button>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">{error}</div>
      )}

      <Card>
        {models.length === 0 && !loading ? (
          <div className="py-12 text-center text-slate-500 text-sm">
            No saved models yet. Models are saved automatically when training completes.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-700 text-slate-500 text-left">
                  <th className="pb-3 pr-4 font-medium">Name</th>
                  <th className="pb-3 pr-4 font-medium">Architecture</th>
                  <th className="pb-3 pr-4 font-medium">Framework</th>
                  <th className="pb-3 pr-4 font-medium">Task</th>
                  <th className="pb-3 pr-4 font-medium">Accuracy</th>
                  <th className="pb-3 pr-4 font-medium">Status</th>
                  <th className="pb-3 pr-4 font-medium">Created</th>
                  <th className="pb-3 font-medium" />
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-800">
                {models.map(m => (
                  <tr key={m.id} className="hover:bg-surface-800/40 transition-colors">
                    <td className="py-3 pr-4 text-white font-medium">{m.name}</td>
                    <td className="py-3 pr-4 text-slate-300">{m.architecture}</td>
                    <td className="py-3 pr-4 text-slate-300 capitalize">{m.framework}</td>
                    <td className="py-3 pr-4 text-slate-300 capitalize">{m.task_type}</td>
                    <td className="py-3 pr-4 text-slate-300">
                      {m.val_accuracy != null ? `${(m.val_accuracy * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-3 pr-4">
                      {m.is_production
                        ? <span className="inline-flex items-center gap-1 text-xs text-green-400 font-medium"><CheckCircle2 size={13} /> Production</span>
                        : <Badge status="default" text="Saved" />}
                    </td>
                    <td className="py-3 pr-4 text-slate-500">{formatDate(m.created_at)}</td>
                    <td className="py-3">
                      {!m.is_production && (
                        <Button
                          variant="secondary" size="sm"
                          loading={promoting === m.id}
                          onClick={() => handlePromote(m.id)}
                        >
                          Promote
                        </Button>
                      )}
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
