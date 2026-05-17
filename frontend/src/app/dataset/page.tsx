'use client'
import { useState } from 'react'
import { Database, Upload, CheckCircle2, Info } from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import { Input, Select } from '@/components/FormControls'

const BUILTIN = [
  { value: 'MNIST',         label: 'MNIST — 70k handwritten digits (28×28, grayscale)' },
  { value: 'Fashion-MNIST', label: 'Fashion-MNIST — 70k clothing items (28×28, grayscale)' },
  { value: 'CIFAR-10',      label: 'CIFAR-10 — 60k images, 10 classes (32×32, RGB)' },
  { value: 'CIFAR-100',     label: 'CIFAR-100 — 60k images, 100 classes (32×32, RGB)' },
]

const TASK_TYPES = [
  { value: 'classification', label: 'Classification' },
  { value: 'detection',      label: 'Object Detection' },
  { value: 'segmentation',   label: 'Segmentation' },
]

interface DatasetInfo {
  name: string
  task_type: string
  source: 'builtin' | 'custom'
}

export default function DatasetPage() {
  const [mode,     setMode]     = useState<'builtin' | 'custom'>('builtin')
  const [builtin,  setBuiltin]  = useState('MNIST')
  const [taskType, setTaskType] = useState('classification')
  const [path,     setPath]     = useState('')
  const [saved,    setSaved]    = useState<DatasetInfo | null>(null)

  function handleSave() {
    const info: DatasetInfo = {
      name:      mode === 'builtin' ? builtin : path,
      task_type: taskType,
      source:    mode,
    }
    sessionStorage.setItem('dataset_config', JSON.stringify(info))
    setSaved(info)
  }

  return (
    <div className="p-8 space-y-8 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Database size={22} className="text-green-400" />
          Dataset
        </h1>
        <p className="text-sm text-slate-500 mt-1">Configure the dataset for your next training run</p>
      </div>

      {/* Mode toggle */}
      <div className="flex gap-2">
        {(['builtin', 'custom'] as const).map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              mode === m
                ? 'bg-brand-600 text-white'
                : 'bg-surface-700 text-slate-400 hover:text-white'
            }`}
          >
            {m === 'builtin' ? '📦 Built-in Dataset' : '📁 Custom Dataset'}
          </button>
        ))}
      </div>

      <Card>
        {mode === 'builtin' ? (
          <div className="space-y-4">
            <Select
              label="Dataset"
              value={builtin}
              onChange={e => setBuiltin(e.target.value)}
              options={BUILTIN}
            />
            <Select
              label="Task Type"
              value={taskType}
              onChange={e => setTaskType(e.target.value)}
              options={TASK_TYPES}
            />

            {/* Info card */}
            <div className="flex gap-2 p-3 rounded-lg bg-brand-500/10 border border-brand-500/20 text-xs text-brand-300">
              <Info size={14} className="mt-0.5 shrink-0" />
              <span>Built-in datasets are downloaded automatically during training when selected.</span>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <Input
              label="Dataset Path (on server)"
              placeholder="/opt/ml-platform/data/my_dataset"
              value={path}
              onChange={e => setPath(e.target.value)}
            />
            <Select
              label="Task Type"
              value={taskType}
              onChange={e => setTaskType(e.target.value)}
              options={TASK_TYPES}
            />

            <div className="border-2 border-dashed border-surface-500 rounded-xl p-8 text-center">
              <Upload size={24} className="mx-auto text-slate-500 mb-2" />
              <p className="text-sm text-slate-500">Drag & drop dataset folder here</p>
              <p className="text-xs text-slate-600 mt-1">or enter the server path above</p>
            </div>

            <div className="space-y-2 text-xs text-slate-500">
              <p className="font-medium text-slate-400">Supported formats:</p>
              <ul className="space-y-1 pl-3">
                <li>• <span className="text-slate-300">Classification</span> — ImageFolder structure (class subfolders)</li>
                <li>• <span className="text-slate-300">Detection</span> — YOLO / COCO JSON / Pascal VOC XML</li>
                <li>• <span className="text-slate-300">Segmentation</span> — images/ + masks/ folders</li>
              </ul>
            </div>
          </div>
        )}

        <div className="mt-6">
          <Button onClick={handleSave} icon={<CheckCircle2 size={14} />}>
            Save Dataset Config
          </Button>
        </div>
      </Card>

      {saved && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-green-500/10 border border-green-500/20">
          <CheckCircle2 size={18} className="text-green-400 shrink-0" />
          <div>
            <p className="text-sm font-medium text-green-300">Dataset configured: <span className="font-bold">{saved.name}</span></p>
            <p className="text-xs text-green-400/70 mt-0.5">Task: {saved.task_type} · Source: {saved.source}</p>
          </div>
        </div>
      )}
    </div>
  )
}
