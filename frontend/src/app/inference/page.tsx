'use client'
import { useEffect, useRef, useState } from 'react'
import Image from 'next/image'
import { Upload, Zap, AlertCircle, BarChart2, Download, Layers, FileArchive } from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import { Select } from '@/components/FormControls'
import { getModels } from '@/lib/api'
import type { ModelVersion } from '@/types'

interface Prediction { class_name: string; confidence: number }
interface InferenceResult {
  predictions?: Prediction[]
  top_class?: string; top_confidence?: number
  class_name?: string; confidence?: number
  [key: string]: unknown
}
interface BatchEntry { file: File; preview: string; result: InferenceResult | null; error?: string }
type Mode = 'single' | 'batch' | 'zip'

export default function InferencePage() {
  const [models,   setModels]   = useState<ModelVersion[]>([])
  const [modelId,  setModelId]  = useState('')
  const [topK,     setTopK]     = useState(5)
  const [threshold,setThreshold]= useState(0.0)
  const [preview,  setPreview]  = useState<string | null>(null)
  const [file,     setFile]     = useState<File | null>(null)
  const [result,   setResult]   = useState<InferenceResult | null>(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')
  const [dragging, setDragging] = useState(false)
  const [batchMode, setBatchMode] = useState(false)
  const [batchEntries, setBatchEntries] = useState<BatchEntry[]>([])
  const [batchRunning, setBatchRunning] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const batchRef = useRef<HTMLInputElement>(null)
  const zipRef   = useRef<HTMLInputElement>(null)
  const [mode, setMode]             = useState<Mode>('single')
  const [zipFile,    setZipFile]    = useState<File | null>(null)
  const [zipRunning, setZipRunning] = useState(false)
  const [zipResult,  setZipResult]  = useState<{ csvUrl: string; total: number; correct: number; accuracy: string } | null>(null)
  const [zipError,   setZipError]   = useState('')

  useEffect(() => {
    getModels().then(data => {
      // Show all models; prefer production ones at the top
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
      .map(f => ({ file: f, preview: URL.createObjectURL(f), result: null }))
    setBatchEntries(prev => [...prev, ...newEntries])
  }

  async function runBatch() {
    if (!modelId || !batchEntries.length) return
    setBatchRunning(true)
    const updated = [...batchEntries]
    await Promise.all(updated.map(async (entry, i) => {
      try {
        updated[i] = { ...entry, result: await runInference(entry.file), error: undefined }
      } catch (e: unknown) {
        updated[i] = { ...entry, error: e instanceof Error ? e.message : 'Failed' }
      }
    }))
    setBatchEntries(updated)
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
      const blob = await res.blob()
      const csvUrl = URL.createObjectURL(blob)
      const total    = parseInt(res.headers.get('X-Total-Images') ?? '0', 10)
      const correct  = parseInt(res.headers.get('X-Correct')      ?? '0', 10)
      const accuracy = res.headers.get('X-Accuracy') ?? 'n/a'
      setZipResult({ csvUrl, total, correct, accuracy })
    } catch (e: unknown) { setZipError(e instanceof Error ? e.message : 'ZIP inference failed') }
    finally { setZipRunning(false) }
  }

  function downloadZipCSV() {
    if (!zipResult) return
    const a = document.createElement('a')
    a.href = zipResult.csvUrl
    a.download = `inference_results_${Date.now()}.csv`
    document.body.appendChild(a); a.click(); document.body.removeChild(a)
  }

  function exportBatchCSV() {
    const headers = ['filename', 'top_class', 'confidence_%']
    const rows = batchEntries.map(e => {
      const p = e.result ? topPred(e.result) : null
      return [e.file.name, p?.class_name ?? (e.error ?? ''), p != null ? (p.confidence * 100).toFixed(1) : '']
    })
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
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

  // Normalise to predictions array
  const preds: Prediction[] = (() => {
    if (!result) return []
    if (Array.isArray(result.predictions)) return result.predictions as Prediction[]
    if (result.top_class != null) return [{ class_name: String(result.top_class), confidence: Number(result.top_confidence ?? 0) }]
    if (result.class_name != null) return [{ class_name: String(result.class_name), confidence: Number(result.confidence ?? 0) }]
    return []
  })()
  const filtered = preds.filter(p => p.confidence >= threshold).slice(0, topK)
  const maxConf  = filtered.length ? Math.max(...filtered.map(p => p.confidence)) : 1
  const hasStructured = filtered.length > 0

  return (
    <div className="p-8 space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Zap size={22} className="text-brand-400" /> Inference
        </h1>
        <p className="text-sm text-slate-500 mt-1">Run prediction on an image using a trained model</p>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-1 p-1 bg-surface-900 rounded-lg w-fit">
        {([
          ['single', 'Single Image', Zap],
          ['batch',  'Batch Mode',   Layers],
          ['zip',    'ZIP Upload',   FileArchive],
        ] as [Mode, string, React.ElementType][]).map(([m, label, Icon]) => (
          <button key={m} onClick={() => { setMode(m); setResult(null); setError(''); setZipResult(null); setZipError('') }}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              mode === m ? 'bg-brand-600 text-white' : 'text-slate-400 hover:text-white'
            }`}>
            <Icon size={12} /> {label}
          </button>
        ))}
      </div>

      {models.length === 0 && (
        <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-sm text-yellow-400 flex items-start gap-2">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          <span>No saved models yet. Complete a training job to automatically register a model, then come back here to run inference. If you have existing jobs, click <strong>Repair Models</strong> on the <a href="/models" className="underline">Models page</a>.</span>
        </div>
      )}

      {/* Settings */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-4">Model &amp; Settings</h2>
        <div className="space-y-4">
          <Select label="Model" value={modelId} onChange={e => setModelId(e.target.value)}
            options={models.map(m => ({
              value: m.id,
              label: `${m.name}${m.is_production ? ' ★' : ''} | ${m.framework} | ${m.val_accuracy != null ? (m.val_accuracy * 100).toFixed(1) + '%' : 'n/a'}`,
            }))}
            disabled={models.length === 0} />
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1.5">
                Top-K Predictions: <span className="text-brand-400 font-mono">{topK}</span>
              </label>
              <input type="range" title="Top-K predictions" min={1} max={10} step={1} value={topK}
                onChange={e => setTopK(Number(e.target.value))} className="w-full accent-brand-500" />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1.5">
                Min Confidence: <span className="text-brand-400 font-mono">{(threshold*100).toFixed(0)}%</span>
              </label>
              <input type="range" title="Minimum confidence threshold" min={0} max={0.95} step={0.05} value={threshold}
                onChange={e => setThreshold(Number(e.target.value))} className="w-full accent-brand-500" />
            </div>
          </div>
        </div>
      </Card>

      {/* Upload — single mode only */}
      {mode === 'single' && (
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-4">Upload Image</h2>
        <div
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
            dragging ? 'border-brand-500 bg-brand-500/10' : 'border-surface-600 hover:border-surface-500'
          }`}
          onDragOver={e => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
        >
          {preview ? (
            <div className="flex flex-col items-center gap-3">
              <div className="relative w-40 h-40">
                <Image src={preview} alt="preview" fill className="object-contain rounded-lg" unoptimized />
              </div>
              <p className="text-xs text-slate-500">{file?.name}</p>
              <p className="text-xs text-brand-400">Click to change image</p>
            </div>
          ) : (
            <>
              <Upload size={32} className="mx-auto text-slate-600 mb-3" />
              <p className="text-sm text-slate-400">Drag &amp; drop an image, or click to browse</p>
              <p className="text-xs text-slate-600 mt-1">PNG, JPG, WEBP, BMP supported</p>
            </>
          )}
        </div>
        <input ref={inputRef} type="file" title="Select an image" accept="image/*" className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />
        {error && <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">{error}</div>}
        <Button className="mt-4 w-full justify-center" size="lg" loading={loading}
          disabled={!file || !modelId} icon={<Zap size={15} />} onClick={handleRun}>
          Run Inference
        </Button>
      </Card>
      )}

      {/* Batch Mode */}
      {mode === 'batch' && (
        <>
          <Card>
            <h2 className="text-sm font-semibold text-slate-300 mb-4">Batch Upload</h2>
            <div
              className="border-2 border-dashed border-surface-600 hover:border-surface-500 rounded-xl p-6 text-center cursor-pointer transition-colors"
              onClick={() => batchRef.current?.click()}
              onDragOver={e => e.preventDefault()}
              onDrop={e => { e.preventDefault(); if (e.dataTransfer.files.length) addBatchFiles(e.dataTransfer.files) }}
            >
              <Layers size={28} className="mx-auto text-slate-600 mb-2" />
              <p className="text-sm text-slate-400">Click or drag multiple images here</p>
              <p className="text-xs text-slate-600 mt-1">{batchEntries.length} image{batchEntries.length !== 1 ? 's' : ''} loaded</p>
            </div>
            <input ref={batchRef} type="file" title="Select images for batch inference" accept="image/*" multiple className="hidden"
              onChange={e => { if (e.target.files) addBatchFiles(e.target.files) }} />
            <div className="mt-4 flex gap-2">
              <Button size="lg" icon={<Zap size={15} />} loading={batchRunning}
                disabled={!batchEntries.length || !modelId} onClick={runBatch}>
                {batchRunning ? 'Running…' : `Run All (${batchEntries.length})`}
              </Button>
              {batchEntries.some(e => e.result) && (
                <Button variant="secondary" icon={<Download size={14} />} onClick={exportBatchCSV}>
                  Export CSV
                </Button>
              )}
              {batchEntries.length > 0 && (
                <Button variant="ghost" onClick={() => setBatchEntries([])}>Clear</Button>
              )}
            </div>
          </Card>

          {batchEntries.length > 0 && (
            <Card>
              <h2 className="text-sm font-semibold text-slate-300 mb-4">Results</h2>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                {batchEntries.map((entry, i) => {
                  const p = entry.result ? topPred(entry.result) : null
                  return (
                    <div key={i} className="bg-surface-800 rounded-lg overflow-hidden border border-surface-700">
                      <div className="relative h-28 bg-surface-900">
                        <Image src={entry.preview} alt={entry.file.name} fill className="object-contain" unoptimized />
                      </div>
                      <div className="p-2">
                        <p className="text-xs text-slate-400 truncate" title={entry.file.name}>{entry.file.name}</p>
                        {entry.error ? (
                          <p className="text-xs text-red-400 mt-1">{entry.error}</p>
                        ) : p ? (
                          <>
                            <p className="text-xs font-medium text-white mt-1 truncate">{p.class_name}</p>
                            <div className="mt-1 h-1.5 bg-surface-700 rounded-full">
                              <div className="h-full bg-brand-500 rounded-full" style={{ width: `${(p.confidence * 100).toFixed(0)}%` }} />
                            </div>
                            <p className="text-xs text-brand-400 mt-0.5">{(p.confidence * 100).toFixed(1)}%</p>
                          </>
                        ) : batchRunning ? (
                          <p className="text-xs text-slate-500 mt-1">Running…</p>
                        ) : (
                          <p className="text-xs text-slate-600 mt-1">Pending</p>
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

      {/* ZIP Mode */}
      {mode === 'zip' && (
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-1 flex items-center gap-2">
            <FileArchive size={15} className="text-brand-400" /> ZIP Batch Inference
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            Upload the same ZIP file produced by <strong className="text-slate-300">Download Test Samples</strong> on the
            Results page. The ZIP must contain images (jpg/png) and optionally a
            <code className="text-brand-400 mx-1 font-mono text-xs">labels.csv</code>
            with columns <code className="text-slate-400 font-mono text-xs">image_name, label, label_name</code>.
            You’ll receive a CSV with predictions and accuracy summary.
          </p>
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
              zipFile ? 'border-brand-500/50 bg-brand-500/5' : 'border-surface-600 hover:border-surface-500'
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
                <p className="text-xs text-slate-500">{(zipFile.size / 1024).toFixed(0)} KB — click to change</p>
              </div>
            ) : (
              <>
                <FileArchive size={32} className="mx-auto text-slate-600 mb-3" />
                <p className="text-sm text-slate-400">Drag &amp; drop a .zip file, or click to browse</p>
                <p className="text-xs text-slate-600 mt-1">Same format as test-samples download</p>
              </>
            )}
          </div>
          <input ref={zipRef} type="file" title="Select ZIP file" accept=".zip,application/zip" className="hidden"
            onChange={e => { const f = e.target.files?.[0]; if (f) { setZipFile(f); setZipResult(null); setZipError('') } }} />
          {zipError && <p className="mt-3 text-sm text-red-400">{zipError}</p>}
          <Button className="mt-4 w-full justify-center" size="lg" loading={zipRunning}
            disabled={!zipFile || !modelId} icon={<Zap size={15} />} onClick={runZip}>
            Run ZIP Inference
          </Button>

          {zipResult && (
            <div className="mt-5 p-4 rounded-xl bg-green-500/10 border border-green-500/20 space-y-3">
              <div className="grid grid-cols-3 gap-3 text-center">
                <div><p className="text-xs text-slate-500">Images</p><p className="text-lg font-bold text-white">{zipResult.total}</p></div>
                <div><p className="text-xs text-slate-500">Correct</p><p className="text-lg font-bold text-green-400">{zipResult.correct}</p></div>
                <div><p className="text-xs text-slate-500">Accuracy</p><p className="text-lg font-bold text-brand-400">{zipResult.accuracy}</p></div>
              </div>
              {zipResult.accuracy === 'n/a' && (
                <p className="text-xs text-slate-500 text-center">No labels.csv found — accuracy unavailable</p>
              )}
              <Button variant="secondary" className="w-full justify-center" icon={<Download size={14} />} onClick={downloadZipCSV}>
                Download Results CSV
              </Button>
            </div>
          )}
        </Card>
      )}

      {/* Single-mode Results */}
      {mode === 'single' && result && (
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4 flex items-center gap-2">
            <BarChart2 size={16} className="text-brand-400" /> Predictions
          </h2>
          {hasStructured ? (
            <div className="space-y-3">
              {filtered.map((p, i) => (
                <div key={i}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className={`font-medium ${i === 0 ? 'text-brand-300' : 'text-slate-300'}`}>
                      {i === 0 ? '🥇 ' : `${i+1}. `}{p.class_name}
                    </span>
                    <span className={`font-mono ${i === 0 ? 'text-brand-400' : 'text-slate-400'}`}>
                      {(p.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-2 bg-surface-700 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${ i === 0 ? 'bg-brand-500' : 'bg-surface-500'}`}
                      style={{ width: `${(p.confidence / maxConf) * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <>
              <p className="text-xs text-slate-500 mb-2">Raw API response:</p>
              <pre className="text-xs text-slate-400 font-mono bg-surface-900 rounded-lg p-4 overflow-x-auto">{JSON.stringify(result, null, 2)}</pre>
            </>
          )}
        </Card>
      )}
    </div>
  )
}

