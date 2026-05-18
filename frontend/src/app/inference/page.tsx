'use client'
import { useEffect, useRef, useState } from 'react'
import Image from 'next/image'
import { Upload, Zap, AlertCircle, BarChart2 } from 'lucide-react'
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
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getModels().then(data => {
      const prod = data.models.filter(m => m.is_production)
      const list = prod.length ? prod : data.models
      setModels(list)
      if (list.length) setModelId(list[0].id)
    }).catch(() => {})
  }, [])

  function handleFile(f: File) {
    if (!f.type.startsWith('image/')) { setError('Please select an image file'); return }
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError('')
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]; if (f) handleFile(f)
  }

  async function handleRun() {
    if (!file || !modelId) return
    setLoading(true); setError(''); setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
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
      setResult(await res.json())
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Inference failed')
    } finally { setLoading(false) }
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

      {models.length === 0 && (
        <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-sm text-yellow-400 flex items-start gap-2">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          No saved models available. Train a model first, then promote it or use any saved model.
        </div>
      )}

      {/* Settings */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-4">Model &amp; Settings</h2>
        <div className="space-y-4">
          <Select label="Model" value={modelId} onChange={e => setModelId(e.target.value)}
            options={models.map(m => ({ value: m.id, label: `${m.name} (${m.framework})` }))}
            disabled={models.length === 0} />
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-400 mb-1.5">
                Top-K Predictions: <span className="text-brand-400 font-mono">{topK}</span>
              </label>
              <input type="range" min={1} max={10} step={1} value={topK}
                onChange={e => setTopK(Number(e.target.value))} className="w-full accent-brand-500" />
            </div>
            <div>
              <label className="block text-xs text-slate-400 mb-1.5">
                Min Confidence: <span className="text-brand-400 font-mono">{(threshold*100).toFixed(0)}%</span>
              </label>
              <input type="range" min={0} max={0.95} step={0.05} value={threshold}
                onChange={e => setThreshold(Number(e.target.value))} className="w-full accent-brand-500" />
            </div>
          </div>
        </div>
      </Card>

      {/* Upload */}
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
        <input ref={inputRef} type="file" accept="image/*" className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />
        {error && <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">{error}</div>}
        <Button className="mt-4 w-full justify-center" size="lg" loading={loading}
          disabled={!file || !modelId} icon={<Zap size={15} />} onClick={handleRun}>
          Run Inference
        </Button>
      </Card>

      {/* Results */}
      {result && (
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

