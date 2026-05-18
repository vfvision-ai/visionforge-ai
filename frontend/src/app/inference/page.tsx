'use client'
import { useEffect, useRef, useState } from 'react'
import Image from 'next/image'
import {
  Upload, Zap, AlertCircle, BarChart2, Download, Layers,
  FileArchive, CheckCircle2, XCircle, Clock, Cpu, Target,
  ChevronRight, RefreshCw, Maximize2,
} from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import { Select } from '@/components/FormControls'
import { getModels } from '@/lib/api'
import type { ModelVersion } from '@/types'

interface Prediction { class_name: string; confidence: number; class_index?: number }
interface InferenceResult {
  predictions?: Prediction[]
  top_class?: string; top_confidence?: number
  class_name?: string; confidence?: number
  framework?: string
  [key: string]: unknown
}
interface BatchEntry {
  file: File; preview: string
  result: InferenceResult | null
  error?: string
  status: 'pending' | 'running' | 'done' | 'error'
}
type Mode = 'single' | 'batch' | 'zip'

const FW_COLOR: Record<string, string> = {
  pytorch: 'text-orange-400 bg-orange-400/10',
  tensorflow: 'text-yellow-400 bg-yellow-400/10',
  sklearn: 'text-green-400 bg-green-400/10',
}

function AccuracyRing({ pct, size = 120 }: { pct: number; size?: number }) {
  const r = (size - 16) / 2
  const circ = 2 * Math.PI * r
  const dash = (pct / 100) * circ
  const color = pct >= 75 ? '#22c55e' : pct >= 50 ? '#f59e0b' : '#ef4444'
  return (
    <svg width={size} height={size} className="-rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#1e293b" strokeWidth={10} />
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={10}
        strokeDasharray={`${dash} ${circ}`} strokeLinecap="round"
        style={{ transition: 'stroke-dasharray 0.8s ease' }} />
    </svg>
  )
}

export default function InferencePage() {
  const [models,        setModels]        = useState<ModelVersion[]>([])
  const [modelId,       setModelId]       = useState('')
  const [topK,          setTopK]          = useState(5)
  const [threshold,     setThreshold]     = useState(0.0)
  const [preview,       setPreview]       = useState<string | null>(null)
  const [file,          setFile]          = useState<File | null>(null)
  const [result,        setResult]        = useState<InferenceResult | null>(null)
  const [loading,       setLoading]       = useState(false)
  const [error,         setError]         = useState('')
  const [dragging,      setDragging]      = useState(false)
  const [batchEntries,  setBatchEntries]  = useState<BatchEntry[]>([])
  const [batchRunning,  setBatchRunning]  = useState(false)
  const [batchDone,     setBatchDone]     = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const batchRef = useRef<HTMLInputElement>(null)
  const zipRef   = useRef<HTMLInputElement>(null)
  const [mode,      setMode]      = useState<Mode>('single')
  const [zipFile,   setZipFile]   = useState<File | null>(null)
  const [zipRunning,setZipRunning]= useState(false)
  const [zipResult, setZipResult] = useState<{
    csvUrl: string; total: number; correct: number; accuracy: string; hasLabels: boolean
  } | null>(null)
  const [zipError, setZipError]   = useState('')

  const selectedModel = models.find(m => m.id === modelId) ?? null

  useEffect(() => {
    getModels().then(data => {
      const prod    = data.models.filter(m => m.is_production)
      const nonProd = data.models.filter(m => !m.is_production)
      const list    = [...prod, ...nonProd]
      setModels(list)
      if (list.length) setModelId(list[0].id)
    }).catch(() => {})
  }, [])

  function handleFile(f: File) {
    if (!f.type.startsWith('image/')) { setError('Please select an image file'); return }
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError('')
  }

  async function runInference(f: File): Promise<InferenceResult> {
    const form = new FormData()
    form.append('file', f)
    form.append('model_id', modelId)
    form.append('top_k', String(topK))
    const apiKey = process.env.NEXT_PUBLIC_API_KEY ?? ''
    const headers: Record<string, string> = {}
    if (apiKey) headers['X-API-Key'] = apiKey
    const res = await fetch('/api/v1/inference/', { method: 'POST', body: form, headers })
    if (!res.ok) {
      const j = await res.json().catch(() => ({}))
      throw new Error((j as { detail?: string }).detail || `HTTP ${res.status}`)
    }
    return res.json()
  }

  function topPred(r: InferenceResult): Prediction | null {
    if (Array.isArray(r.predictions) && r.predictions.length) return r.predictions[0] as Prediction
    if (r.top_class != null) return { class_name: String(r.top_class), confidence: Number(r.top_confidence ?? 0) }
    if (r.class_name != null) return { class_name: String(r.class_name), confidence: Number(r.confidence ?? 0) }
    return null
  }

  function addBatchFiles(files: FileList | File[]) {
    const newEntries: BatchEntry[] = Array.from(files)
      .filter(f => f.type.startsWith('image/'))
      .map(f => ({ file: f, preview: URL.createObjectURL(f), result: null, status: 'pending' as const }))
    setBatchEntries(prev => [...prev, ...newEntries])
  }

  async function runBatch() {
    if (!modelId || !batchEntries.length) return
    setBatchRunning(true); setBatchDone(0)
    // Mark all as running
    setBatchEntries(prev => prev.map(e => ({ ...e, status: 'running' as const })))
    let done = 0
    const updated = batchEntries.map(e => ({ ...e }))
    await Promise.all(updated.map(async (entry, i) => {
      try {
        const res = await runInference(entry.file)
        updated[i] = { ...entry, result: res, status: 'done', error: undefined }
      } catch (e: unknown) {
        updated[i] = { ...entry, error: e instanceof Error ? e.message : 'Failed', status: 'error' }
      }
      done++
      setBatchDone(done)
      setBatchEntries([...updated])
    }))
    setBatchRunning(false)
  }

  async function runZip() {
    if (!zipFile || !modelId) return
    setZipRunning(true); setZipError(''); setZipResult(null)
    try {
      const form = new FormData()
      form.append('file', zipFile)
      form.append('model_id', modelId)
      form.append('top_k', String(topK))
      const apiKey = process.env.NEXT_PUBLIC_API_KEY ?? ''
      const headers: Record<string, string> = {}
      if (apiKey) headers['X-API-Key'] = apiKey
      const res = await fetch('/api/v1/inference/zip', { method: 'POST', body: form, headers })
      if (!res.ok) {
        const j = await res.json().catch(() => ({}))
        throw new Error((j as { detail?: string }).detail || `HTTP ${res.status}`)
      }
      const data = await res.json() as {
        total: number; labelled: number; correct: number
        accuracy: string; has_labels: boolean
        csv_b64: string; filename: string
      }
      const csvBytes = Uint8Array.from(atob(data.csv_b64), c => c.charCodeAt(0))
      const csvUrl   = URL.createObjectURL(new Blob([csvBytes], { type: 'text/csv' }))
      setZipResult({ csvUrl, total: data.total, correct: data.correct, accuracy: data.accuracy, hasLabels: data.has_labels })
    } catch (e: unknown) { setZipError(e instanceof Error ? e.message : 'ZIP inference failed') }
    finally { setZipRunning(false) }
  }

  function downloadZipCSV() {
    if (!zipResult) return
    const a = document.createElement('a'); a.href = zipResult.csvUrl
    a.download = `inference_results_${Date.now()}.csv`
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
  }

  function exportBatchCSV() {
    const hdrs = ['filename', 'top_class', 'confidence_%', 'status']
    const rows = batchEntries.map(e => {
      const p = e.result ? topPred(e.result) : null
      return [e.file.name, p?.class_name ?? (e.error ?? ''), p != null ? (p.confidence * 100).toFixed(1) : '', e.status]
    })
    const csv  = [hdrs, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url  = URL.createObjectURL(blob)
    const a    = document.createElement('a')
    a.href = url; a.download = `inference_batch_${Date.now()}.csv`
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]; if (f) handleFile(f)
  }

  async function handleRun() {
    if (!file || !modelId) return
    setLoading(true); setError(''); setResult(null)
    try { setResult(await runInference(file)) }
    catch (e: unknown) { setError(e instanceof Error ? e.message : 'Inference failed') }
    finally { setLoading(false) }
  }

  const preds: Prediction[] = (() => {
    if (!result) return []
    if (Array.isArray(result.predictions)) return result.predictions as Prediction[]
    if (result.top_class != null) return [{ class_name: String(result.top_class), confidence: Number(result.top_confidence ?? 0) }]
    if (result.class_name != null) return [{ class_name: String(result.class_name), confidence: Number(result.confidence ?? 0) }]
    return []
  })()
  const filtered    = preds.filter(p => p.confidence >= threshold).slice(0, topK)
  const maxConf     = filtered.length ? Math.max(...filtered.map(p => p.confidence)) : 1
  const topPredResult = filtered[0] ?? null

  const batchSucceeded = batchEntries.filter(e => e.status === 'done').length
  const batchFailed    = batchEntries.filter(e => e.status === 'error').length
  const zipAccNum      = zipResult ? parseFloat(zipResult.accuracy) : NaN

  return (
    <div className="p-6 space-y-5 max-w-5xl">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Zap size={22} className="text-brand-400" /> Inference
          </h1>
          <p className="text-sm text-slate-500 mt-1">Run predictions on images using your trained models</p>
        </div>
        {/* Mode toggle */}
        <div className="flex gap-1 p-1 bg-surface-900 rounded-lg">
          {([
            ['single', 'Single', Zap],
            ['batch',  'Batch',  Layers],
            ['zip',    'ZIP',    FileArchive],
          ] as [Mode, string, React.ElementType][]).map(([m, label, Icon]) => (
            <button key={m}
              onClick={() => { setMode(m); setResult(null); setError(''); setZipResult(null); setZipError('') }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                mode === m ? 'bg-brand-600 text-white shadow-sm' : 'text-slate-400 hover:text-white'
              }`}>
              <Icon size={12} /> {label}
            </button>
          ))}
        </div>
      </div>

      {/* No models warning */}
      {models.length === 0 && (
        <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-sm text-yellow-400 flex items-start gap-2">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>No saved models yet. Complete a training job first, or click <strong>Repair Models</strong> on the <a href="/models" className="underline">Models page</a> to recover existing ones.</span>
        </div>
      )}

      {/* Top bar: model selector + settings */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Model selector */}
        <div className="lg:col-span-2">
          <Card>
            <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Model</h2>
            <Select label="" value={modelId} onChange={e => setModelId(e.target.value)}
              options={models.map(m => ({
                value: m.id,
                label: `${m.is_production ? 'â˜… ' : ''}${m.name} â€” ${m.framework} / ${m.architecture}${m.val_accuracy != null ? ` (${(m.val_accuracy * 100).toFixed(1)}%)` : ''}`,
              }))}
              disabled={models.length === 0} />
            {selectedModel && (
              <div className="mt-3 flex flex-wrap gap-2">
                <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${FW_COLOR[selectedModel.framework] ?? 'text-slate-400 bg-surface-700'}`}>
                  {selectedModel.framework}
                </span>
                <span className="text-xs text-slate-400 bg-surface-700 px-2 py-0.5 rounded-full flex items-center gap-1">
                  <Cpu size={10} /> {selectedModel.architecture}
                </span>
                {selectedModel.num_classes != null && (
                  <span className="text-xs text-slate-400 bg-surface-700 px-2 py-0.5 rounded-full flex items-center gap-1">
                    <Target size={10} /> {selectedModel.num_classes} classes
                  </span>
                )}
                {selectedModel.extra_metrics?.input_size != null && (
                  <span className="text-xs text-slate-400 bg-surface-700 px-2 py-0.5 rounded-full flex items-center gap-1">
                    <Maximize2 size={10} /> {String(selectedModel.extra_metrics.input_size)}
                  </span>
                )}
                {selectedModel.val_accuracy != null && (
                  <span className="text-xs text-green-400 bg-green-400/10 px-2 py-0.5 rounded-full">
                    {(selectedModel.val_accuracy * 100).toFixed(1)}% val acc
                  </span>
                )}
                {selectedModel.is_production && (
                  <span className="text-xs text-brand-300 bg-brand-500/10 px-2 py-0.5 rounded-full">â˜… Production</span>
                )}
              </div>
            )}
          </Card>
        </div>

        {/* Settings */}
        <Card>
          <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">Settings</h2>
          <div className="space-y-3">
            <div>
              <label className="text-xs text-slate-400 flex justify-between mb-1">
                Top-K <span className="text-brand-400 font-mono">{topK}</span>
              </label>
              <input type="range" title="Top-K" min={1} max={10} step={1} value={topK}
                onChange={e => setTopK(Number(e.target.value))} className="w-full accent-brand-500" />
            </div>
            <div>
              <label className="text-xs text-slate-400 flex justify-between mb-1">
                Min Confidence <span className="text-brand-400 font-mono">{(threshold * 100).toFixed(0)}%</span>
              </label>
              <input type="range" title="Min confidence" min={0} max={0.95} step={0.05} value={threshold}
                onChange={e => setThreshold(Number(e.target.value))} className="w-full accent-brand-500" />
            </div>
          </div>
        </Card>
      </div>

      {/* â”€â”€ SINGLE MODE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      {mode === 'single' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Upload */}
          <Card>
            <h2 className="text-sm font-semibold text-slate-300 mb-3 flex items-center gap-2">
              <Upload size={14} className="text-brand-400" /> Upload Image
            </h2>
            <div
              className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
                dragging ? 'border-brand-500 bg-brand-500/10 scale-[1.01]'
                         : 'border-surface-600 hover:border-brand-500/50 hover:bg-surface-700/30'
              }`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => inputRef.current?.click()}
            >
              {preview ? (
                <div className="flex flex-col items-center gap-2">
                  <div className="relative w-36 h-36">
                    <Image src={preview} alt="preview" fill className="object-contain rounded-lg" unoptimized />
                  </div>
                  <p className="text-xs text-slate-500 truncate max-w-full">{file?.name}</p>
                  <p className="text-xs text-brand-400 flex items-center gap-1"><RefreshCw size={10} /> Click to change</p>
                </div>
              ) : (
                <>
                  <Upload size={28} className="mx-auto text-slate-600 mb-2" />
                  <p className="text-sm text-slate-400">Drag &amp; drop or click to browse</p>
                  <p className="text-xs text-slate-600 mt-1">PNG Â· JPG Â· WEBP Â· BMP</p>
                </>
              )}
            </div>
            <input ref={inputRef} type="file" title="Select image" accept="image/*" className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />
            {error && (
              <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400 flex items-center gap-2">
                <XCircle size={14} /> {error}
              </div>
            )}
            <Button className="mt-4 w-full justify-center" size="lg" loading={loading}
              disabled={!file || !modelId} icon={<Zap size={15} />} onClick={handleRun}>
              Run Inference
            </Button>
          </Card>

          {/* Results */}
          <div className="space-y-4">
            {result && topPredResult ? (
              <>
                {/* Top prediction hero */}
                <div className="rounded-xl border border-brand-500/30 bg-brand-500/5 p-5 text-center">
                  <p className="text-xs text-slate-500 mb-1 uppercase tracking-wider">Top Prediction</p>
                  <p className="text-3xl font-bold text-white mb-1 capitalize">
                    {topPredResult.class_name}
                  </p>
                  <div className="flex items-center justify-center gap-2">
                    <div className="h-2 flex-1 max-w-32 bg-surface-700 rounded-full overflow-hidden">
                      <div className="h-full bg-brand-500 rounded-full transition-all duration-700"
                        style={{ width: `${(topPredResult.confidence * 100).toFixed(0)}%` }} />
                    </div>
                    <span className="text-brand-400 font-bold text-lg font-mono">
                      {(topPredResult.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Remaining predictions */}
                {filtered.length > 1 && (
                  <Card>
                    <h3 className="text-xs font-semibold text-slate-400 mb-3 flex items-center gap-1.5">
                      <BarChart2 size={12} /> All Predictions
                    </h3>
                    <div className="space-y-2.5">
                      {filtered.map((p, i) => (
                        <div key={i}>
                          <div className="flex justify-between text-xs mb-1">
                            <span className={`font-medium flex items-center gap-1 ${i === 0 ? 'text-brand-300' : 'text-slate-300'}`}>
                              {i === 0 && <CheckCircle2 size={11} className="text-brand-400" />}
                              {p.class_name}
                            </span>
                            <span className="font-mono text-slate-400">{(p.confidence * 100).toFixed(1)}%</span>
                          </div>
                          <div className="h-1.5 bg-surface-700 rounded-full overflow-hidden">
                            <div className={`h-full rounded-full transition-all duration-500 ${i === 0 ? 'bg-brand-500' : 'bg-surface-500'}`}
                              style={{ width: `${(p.confidence / maxConf) * 100}%` }} />
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}
              </>
            ) : result && !topPredResult ? (
              <Card>
                <p className="text-xs text-slate-500 mb-2">Raw response:</p>
                <pre className="text-xs text-slate-400 font-mono bg-surface-900 rounded-lg p-3 overflow-x-auto">{JSON.stringify(result, null, 2)}</pre>
              </Card>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center rounded-xl border-2 border-dashed border-surface-700 p-10">
                <BarChart2 size={32} className="text-slate-700 mb-3" />
                <p className="text-sm text-slate-500">Predictions will appear here</p>
                <p className="text-xs text-slate-600 mt-1">Upload an image and click Run Inference</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* â”€â”€ BATCH MODE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      {mode === 'batch' && (
        <>
          <Card>
            <div className="flex items-start justify-between mb-3">
              <h2 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                <Layers size={14} className="text-brand-400" /> Batch Upload
              </h2>
              {batchEntries.length > 0 && !batchRunning && (
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  {batchSucceeded > 0 && <span className="text-green-400 flex items-center gap-1"><CheckCircle2 size={11} /> {batchSucceeded} done</span>}
                  {batchFailed > 0   && <span className="text-red-400   flex items-center gap-1"><XCircle      size={11} /> {batchFailed} failed</span>}
                  {batchEntries.some(e => e.status === 'pending') && (
                    <span className="text-slate-500 flex items-center gap-1"><Clock size={11} /> {batchEntries.filter(e => e.status === 'pending').length} pending</span>
                  )}
                </div>
              )}
            </div>

            {/* Drop zone */}
            <div
              className="border-2 border-dashed border-surface-600 hover:border-brand-500/50 rounded-xl p-6 text-center cursor-pointer transition-colors"
              onClick={() => batchRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); if (e.dataTransfer.files.length) addBatchFiles(e.dataTransfer.files) }}
            >
              <Layers size={24} className="mx-auto text-slate-600 mb-2" />
              <p className="text-sm text-slate-400">Click or drag multiple images</p>
              <p className="text-xs text-slate-600 mt-1">
                {batchEntries.length > 0 ? `${batchEntries.length} image${batchEntries.length !== 1 ? 's' : ''} loaded â€” drop more to add` : 'PNG Â· JPG Â· WEBP Â· BMP'}
              </p>
            </div>
            <input ref={batchRef} type="file" title="Select images" accept="image/*" multiple className="hidden"
              onChange={e => { if (e.target.files) addBatchFiles(e.target.files) }} />

            {/* Progress bar while running */}
            {batchRunning && (
              <div className="mt-3">
                <div className="flex justify-between text-xs text-slate-500 mb-1">
                  <span>Processingâ€¦</span>
                  <span>{batchDone} / {batchEntries.length}</span>
                </div>
                <div className="h-1.5 bg-surface-700 rounded-full overflow-hidden">
                  <div className="h-full bg-brand-500 rounded-full transition-all duration-300"
                    style={{ width: `${batchEntries.length ? (batchDone / batchEntries.length * 100) : 0}%` }} />
                </div>
              </div>
            )}

            <div className="mt-4 flex gap-2 flex-wrap">
              <Button size="lg" icon={<Zap size={15} />} loading={batchRunning}
                disabled={!batchEntries.length || !modelId} onClick={runBatch}>
                {batchRunning ? `Runningâ€¦ (${batchDone}/${batchEntries.length})` : `Run All (${batchEntries.length})`}
              </Button>
              {batchEntries.some(e => e.result) && (
                <Button variant="secondary" icon={<Download size={14} />} onClick={exportBatchCSV}>Export CSV</Button>
              )}
              {batchEntries.length > 0 && (
                <Button variant="ghost" onClick={() => { setBatchEntries([]); setBatchDone(0) }}>Clear</Button>
              )}
            </div>
          </Card>

          {batchEntries.length > 0 && (
            <Card>
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-semibold text-slate-300">Results</h2>
                <div className="flex gap-3 text-xs">
                  <span className="text-slate-500">{batchEntries.length} images</span>
                  {batchSucceeded > 0 && <span className="text-green-400">{batchSucceeded} âœ“</span>}
                  {batchFailed   > 0 && <span className="text-red-400">{batchFailed} âœ—</span>}
                </div>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-5 gap-3">
                {batchEntries.map((entry, i) => {
                  const p = entry.result ? topPred(entry.result) : null
                  return (
                    <div key={i} className={`rounded-lg overflow-hidden border transition-colors ${
                      entry.status === 'done'    ? 'border-green-500/30 bg-surface-800'
                    : entry.status === 'error'   ? 'border-red-500/30   bg-surface-800'
                    : entry.status === 'running' ? 'border-brand-500/30 bg-brand-500/5'
                    :                              'border-surface-700   bg-surface-800'
                    }`}>
                      <div className="relative h-24 bg-surface-900">
                        <Image src={entry.preview} alt={entry.file.name} fill className="object-contain" unoptimized />
                        {/* Status badge */}
                        <div className="absolute top-1 right-1">
                          {entry.status === 'done'    && <CheckCircle2 size={14} className="text-green-400 drop-shadow" />}
                          {entry.status === 'error'   && <XCircle      size={14} className="text-red-400   drop-shadow" />}
                          {entry.status === 'running' && <RefreshCw    size={12} className="text-brand-400 animate-spin" />}
                          {entry.status === 'pending' && <Clock        size={12} className="text-slate-500" />}
                        </div>
                      </div>
                      <div className="p-2">
                        <p className="text-xs text-slate-400 truncate" title={entry.file.name}>{entry.file.name}</p>
                        {entry.error ? (
                          <p className="text-xs text-red-400 mt-1 truncate" title={entry.error}>{entry.error}</p>
                        ) : p ? (
                          <>
                            <p className="text-xs font-semibold text-white mt-0.5 truncate" title={p.class_name}>{p.class_name}</p>
                            <div className="mt-1 h-1 bg-surface-700 rounded-full">
                              <div className="h-full bg-brand-500 rounded-full" style={{ width: `${(p.confidence * 100).toFixed(0)}%` }} />
                            </div>
                            <p className="text-xs text-brand-400 mt-0.5">{(p.confidence * 100).toFixed(1)}%</p>
                          </>
                        ) : (
                          <p className="text-xs text-slate-600 mt-1">{entry.status === 'running' ? 'Runningâ€¦' : 'Pending'}</p>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            </Card>
          )}
        </>
      )}

      {/* â”€â”€ ZIP MODE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
      {mode === 'zip' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <h2 className="text-sm font-semibold text-slate-300 mb-1 flex items-center gap-2">
              <FileArchive size={14} className="text-brand-400" /> ZIP Batch Inference
            </h2>
            <p className="text-xs text-slate-500 mb-4">
              Upload the ZIP produced by <strong className="text-slate-300">Download Test Samples</strong> on the Results page.
              Include a <code className="text-brand-400 font-mono">labels.csv</code> (columns: <code className="text-slate-400 font-mono">image_name, label, label_name</code>) to get accuracy.
            </p>

            <div
              className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                zipFile ? 'border-brand-500/50 bg-brand-500/5' : 'border-surface-600 hover:border-brand-500/50 hover:bg-surface-700/20'
              }`}
              onClick={() => zipRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => {
                e.preventDefault()
                const f = e.dataTransfer.files[0]
                if (f?.name.endsWith('.zip')) { setZipFile(f); setZipResult(null); setZipError('') }
                else setZipError('Please drop a .zip file')
              }}
            >
              {zipFile ? (
                <div className="flex flex-col items-center gap-2">
                  <FileArchive size={32} className="text-brand-400" />
                  <p className="text-sm text-white font-medium">{zipFile.name}</p>
                  <p className="text-xs text-slate-500">{(zipFile.size / 1024).toFixed(0)} KB</p>
                  <p className="text-xs text-brand-400 flex items-center gap-1"><RefreshCw size={10} /> Click to change</p>
                </div>
              ) : (
                <>
                  <FileArchive size={28} className="mx-auto text-slate-600 mb-3" />
                  <p className="text-sm text-slate-400">Drag &amp; drop a .zip file, or click to browse</p>
                  <p className="text-xs text-slate-600 mt-1">Same format as test-samples download</p>
                </>
              )}
            </div>
            <input ref={zipRef} type="file" title="Select ZIP" accept=".zip,application/zip" className="hidden"
              onChange={e => { const f = e.target.files?.[0]; if (f) { setZipFile(f); setZipResult(null); setZipError('') } }} />

            {/* Loading state */}
            {zipRunning && (
              <div className="mt-4 p-3 rounded-lg bg-brand-500/10 border border-brand-500/20 flex items-center gap-3">
                <RefreshCw size={16} className="text-brand-400 animate-spin shrink-0" />
                <div>
                  <p className="text-sm text-brand-300 font-medium">Processing imagesâ€¦</p>
                  <p className="text-xs text-slate-500 mt-0.5">Running inference on each image in the ZIP</p>
                </div>
              </div>
            )}

            {zipError && (
              <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400 flex items-center gap-2">
                <XCircle size={14} /> {zipError}
              </div>
            )}
            <Button className="mt-4 w-full justify-center" size="lg" loading={zipRunning}
              disabled={!zipFile || !modelId} icon={<Zap size={15} />} onClick={runZip}>
              Run ZIP Inference
            </Button>
          </Card>

          {/* ZIP Results */}
          <div className="space-y-4">
            {zipResult ? (
              <>
                {/* Accuracy ring */}
                {zipResult.hasLabels && (
                  <Card className="flex flex-col items-center gap-2 py-6">
                    <p className="text-xs text-slate-500 uppercase tracking-wider">Overall Accuracy</p>
                    <div className="relative flex items-center justify-center">
                      <AccuracyRing pct={isNaN(zipAccNum) ? 0 : zipAccNum} size={130} />
                      <div className="absolute text-center">
                        <p className="text-2xl font-bold text-white">{zipResult.accuracy}</p>
                      </div>
                    </div>
                    <div className="flex gap-6 text-center mt-2">
                      <div><p className="text-xs text-slate-500">Total</p><p className="text-xl font-bold text-white">{zipResult.total}</p></div>
                      <div><p className="text-xs text-slate-500">Correct</p><p className="text-xl font-bold text-green-400">{zipResult.correct}</p></div>
                      <div><p className="text-xs text-slate-500">Wrong</p><p className="text-xl font-bold text-red-400">{zipResult.total - zipResult.correct}</p></div>
                    </div>
                  </Card>
                )}

                {/* No labels */}
                {!zipResult.hasLabels && (
                  <Card className="text-center py-8">
                    <FileArchive size={32} className="mx-auto text-slate-600 mb-3" />
                    <p className="text-sm text-white font-semibold">{zipResult.total} images processed</p>
                    <p className="text-xs text-slate-500 mt-1">No labels.csv found â€” add one to see accuracy</p>
                  </Card>
                )}

                {/* Download button */}
                <Button variant="secondary" className="w-full justify-center" icon={<Download size={14} />} onClick={downloadZipCSV}>
                  Download Results CSV
                </Button>

                {/* What&apos;s in the CSV */}
                <div className="p-3 rounded-lg bg-surface-900 border border-surface-700">
                  <p className="text-xs text-slate-500 mb-1.5 font-semibold">CSV columns:</p>
                  {['image_name', 'true_label', 'true_label_name', 'predicted_class', 'confidence_%', 'correct'].map(col => (
                    <span key={col} className="inline-block text-xs font-mono text-slate-400 bg-surface-700 rounded px-1.5 py-0.5 mr-1 mb-1">{col}</span>
                  ))}
                </div>
              </>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center rounded-xl border-2 border-dashed border-surface-700 p-10 min-h-[200px]">
                <ChevronRight size={28} className="text-slate-700 mb-3" />
                <p className="text-sm text-slate-500">Results will appear here</p>
                <p className="text-xs text-slate-600 mt-1">Upload a ZIP and run inference</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
