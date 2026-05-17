'use client'
import { useEffect, useRef, useState } from 'react'
import Image from 'next/image'
import { Upload, Zap, AlertCircle } from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import { Select } from '@/components/FormControls'
import { getModels } from '@/lib/api'
import type { ModelVersion } from '@/types'

export default function InferencePage() {
  const [models,   setModels]   = useState<ModelVersion[]>([])
  const [modelId,  setModelId]  = useState('')
  const [preview,  setPreview]  = useState<string | null>(null)
  const [file,     setFile]     = useState<File | null>(null)
  const [result,   setResult]   = useState<Record<string, unknown> | null>(null)
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getModels().then(data => {
      const prod = data.models.filter(m => m.is_production)
      setModels(prod.length ? prod : data.models)
      if (data.models.length) setModelId(data.models[0].id)
    }).catch(() => {})
  }, [])

  function handleFile(f: File) {
    if (!f.type.startsWith('image/')) { setError('Please select an image file'); return }
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError('')
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault(); setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }

  async function handleRunInference() {
    if (!file || !modelId) return
    setLoading(true); setError(''); setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('model_id', modelId)
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
    } finally {
      setLoading(false)
    }
  }

  const modelOptions = models.map(m => ({ value: m.id, label: `${m.name} (${m.framework})` }))

  return (
    <div className="p-8 space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Zap size={22} className="text-brand-400" />
          Inference
        </h1>
        <p className="text-sm text-slate-500 mt-1">Run prediction on an image</p>
      </div>

      {models.length === 0 && (
        <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-sm text-yellow-400 flex items-start gap-2">
          <AlertCircle size={16} className="mt-0.5 shrink-0" />
          No production models available. Train a model and promote it first.
        </div>
      )}

      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-4">Select Model</h2>
        <Select
          label="Model"
          value={modelId}
          onChange={e => setModelId(e.target.value)}
          options={modelOptions}
          disabled={models.length === 0}
        />
      </Card>

      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-4">Upload Image</h2>

        {/* Drop zone */}
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
            </div>
          ) : (
            <>
              <Upload size={32} className="mx-auto text-slate-600 mb-3" />
              <p className="text-sm text-slate-400">Drag &amp; drop an image here, or click to browse</p>
              <p className="text-xs text-slate-600 mt-1">PNG, JPG, WEBP supported</p>
            </>
          )}
        </div>
        <input ref={inputRef} type="file" accept="image/*" className="hidden"
          onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />

        {error && (
          <div className="mt-3 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">{error}</div>
        )}

        <Button
          className="mt-4 w-full justify-center"
          size="lg"
          loading={loading}
          disabled={!file || !modelId}
          icon={<Zap size={15} />}
          onClick={handleRunInference}
        >
          Run Inference
        </Button>
      </Card>

      {result && (
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Result</h2>
          <pre className="text-xs text-slate-400 font-mono bg-surface-900 rounded-lg p-4 overflow-x-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        </Card>
      )}
    </div>
  )
}
