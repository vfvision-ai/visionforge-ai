'use client'
import { useEffect, useState } from 'react'
import { Package, CheckCircle2, RefreshCw, Download, Trash2, ChevronUp, ChevronDown, FileDown, X } from 'lucide-react'
import Card from '@/components/Card'
import Badge from '@/components/Badge'
import Button from '@/components/Button'
import { Select, Input } from '@/components/FormControls'
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

type SortKey = 'name' | 'architecture' | 'framework' | 'task_type' | 'val_accuracy' | 'created_at'

function SortIcon({ col, sortKey, sortDir }: { col: SortKey; sortKey: SortKey; sortDir: 'asc'|'desc' }) {
  if (sortKey !== col) return <ChevronUp size={11} className="opacity-20 ml-0.5" />
  return sortDir === 'asc' ? <ChevronUp size={11} className="ml-0.5 text-brand-400" /> : <ChevronDown size={11} className="ml-0.5 text-brand-400" />
}

export default function ModelsPage() {
  const [models,    setModels]    = useState<ModelVersion[]>([])
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState('')
  const [promoting, setPromoting] = useState<string | null>(null)
  const [deleting,  setDeleting]  = useState<string | null>(null)
  const [fwFilter,  setFwFilter]  = useState('')
  const [taskFilter,setTaskFilter]= useState('')
  const [query,     setQuery]     = useState('')
  const [sortKey,   setSortKey]   = useState<SortKey>('created_at')
  const [sortDir,   setSortDir]   = useState<'asc'|'desc'>('desc')
  const [exporting, setExporting] = useState<string | null>(null)
  const [exportModel, setExportModel] = useState<ModelVersion | null>(null)
  const [exportFmt,   setExportFmt]   = useState('onnx')
  const [exportSize,  setExportSize]  = useState('224')

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

  async function handleExport(m: ModelVersion, fmt: string, inputSize: string) {
    setExporting(m.id)
    try {
      const res = await fetch(`/api/v1/models/${m.id}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: fmt, input_size: parseInt(inputSize, 10) }),
      })
      if (!res.ok) { const j = await res.json().catch(() => ({})); throw new Error((j as {detail?: string}).detail ?? 'Export failed') }
      const blob = await res.blob()
      const ext  = fmt === 'onnx' ? 'onnx' : 'pt'
      const url  = URL.createObjectURL(blob)
      const a    = document.createElement('a')
      a.href = url; a.download = `${m.name.replace(/\s+/g,'_')}_${fmt}.${ext}`
      document.body.appendChild(a); a.click(); document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setExportModel(null)
    } catch (e: unknown) { setError(e instanceof Error ? e.message : 'Export failed') }
    finally { setExporting(null) }
  }

  function toggleSort(col: SortKey) {
    if (sortKey === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(col); setSortDir('asc') }
  }

  useEffect(() => { load() }, [])

  const q = query.trim().toLowerCase()
  const filtered = models
    .filter(m => !fwFilter   || m.framework === fwFilter)
    .filter(m => !taskFilter || m.task_type === taskFilter)
    .filter(m => !q || m.name.toLowerCase().includes(q) || m.architecture.toLowerCase().includes(q))

  type AnyM = Record<string, unknown>
  const visible = [...filtered].sort((a, b) => {
    let av: unknown = (a as unknown as AnyM)[sortKey], bv: unknown = (b as unknown as AnyM)[sortKey]
    if (sortKey === 'val_accuracy') { av = (av as number) ?? -1; bv = (bv as number) ?? -1 }
    if (sortKey === 'created_at')  { av = av ?? ''; bv = bv ?? '' }
    if (av === bv) return 0
    const cmp = av! < bv! ? -1 : 1
    return sortDir === 'asc' ? cmp : -cmp
  })

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

      {/* Filters + Search */}
      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-xs text-slate-500">Filter:</span>
        <div className="w-44"><Select label="" value={fwFilter}   onChange={e => setFwFilter(e.target.value)}   options={FW_OPTS}   /></div>
        <div className="w-44"><Select label="" value={taskFilter} onChange={e => setTaskFilter(e.target.value)} options={TASK_OPTS} /></div>
        <div className="w-56"><Input label="" placeholder="Search name / architecture…" value={query} onChange={e => setQuery(e.target.value)} /></div>
        {(fwFilter || taskFilter || query) && (
          <button onClick={() => { setFwFilter(''); setTaskFilter(''); setQuery('') }} className="text-xs text-slate-500 hover:text-white">Clear</button>
        )}
        <span className="ml-auto text-xs text-slate-600">{visible.length} of {models.length}</span>
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
                <tr className="border-b border-surface-700 text-slate-500 text-left text-xs">
                  {(['name','architecture','framework','task_type','val_accuracy','created_at'] as SortKey[]).map(col => (
                    <th key={col} onClick={() => toggleSort(col)}
                      className="pb-3 pr-4 font-medium cursor-pointer hover:text-slate-300 select-none whitespace-nowrap">
                      <span className="inline-flex items-center gap-0.5">
                        {col === 'task_type' ? 'Task' : col === 'val_accuracy' ? 'Val Acc' : col === 'created_at' ? 'Created' : col.charAt(0).toUpperCase() + col.slice(1)}
                        <SortIcon col={col} sortKey={sortKey} sortDir={sortDir} />
                      </span>
                    </th>
                  ))}
                  <th className="pb-3 pr-4 font-medium">Status</th>
                  <th className="pb-3 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-surface-800">
                {visible.map(m => (
                  <tr key={m.id} className="hover:bg-surface-800/40 transition-colors">
                    <td className="py-3 pr-4 text-white font-medium text-sm">{m.name}</td>
                    <td className="py-3 pr-4 text-slate-300 text-sm">{m.architecture}</td>
                    <td className="py-3 pr-4 text-slate-300 capitalize text-sm">{m.framework}</td>
                    <td className="py-3 pr-4 text-slate-300 capitalize text-sm">{m.task_type}</td>
                    <td className="py-3 pr-4 text-slate-300 text-sm">{m.val_accuracy != null ? `${(m.val_accuracy * 100).toFixed(1)}%` : '—'}</td>
                    <td className="py-3 pr-4 text-slate-500 text-sm">{formatDate(m.created_at)}</td>
                    <td className="py-3 pr-4">
                      {m.is_production
                        ? <span className="inline-flex items-center gap-1 text-xs text-green-400 font-medium"><CheckCircle2 size={13} /> Production</span>
                        : <Badge status="default" text="Saved" />}
                    </td>
                    <td className="py-3">
                      <div className="flex items-center gap-1">
                        <button title="Download model file" onClick={() => handleDownload(m.id)}
                          className="p-1.5 rounded text-slate-500 hover:text-brand-400 hover:bg-brand-500/10 transition-colors">
                          <Download size={13} />
                        </button>
                        {m.framework === 'pytorch' && (
                          <button title="Export model (ONNX / TorchScript)" onClick={() => { setExportModel(m); setExportFmt('onnx'); setExportSize('224') }}
                            className="p-1.5 rounded text-slate-500 hover:text-purple-400 hover:bg-purple-500/10 transition-colors">
                            <FileDown size={13} />
                          </button>
                        )}
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

      {/* Export Modal */}
      {exportModel && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-surface-800 border border-surface-700 rounded-xl p-6 w-full max-w-sm shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">Export Model</h3>
              <button title="Close" onClick={() => setExportModel(null)} className="text-slate-500 hover:text-white"><X size={16} /></button>
            </div>
            <p className="text-xs text-slate-400">{exportModel.name} — {exportModel.architecture}</p>
            <div className="space-y-3">
              <Select label="Format" value={exportFmt} onChange={e => setExportFmt(e.target.value)}
                options={[{ value: 'onnx', label: 'ONNX (cross-platform)' }, { value: 'torchscript', label: 'TorchScript' }]} />
              <Input label="Input size (px)" type="number" value={exportSize} onChange={e => setExportSize(e.target.value)} />
            </div>
            <div className="flex gap-2 pt-1">
              <Button variant="secondary" size="sm" onClick={() => setExportModel(null)}>Cancel</Button>
              <Button variant="primary" size="sm" loading={exporting === exportModel.id}
                onClick={() => handleExport(exportModel, exportFmt, exportSize)}>
                Export &amp; Download
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
