'use client'
import { useState } from 'react'
import { Database, Upload, CheckCircle2, Info } from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import { Input } from '@/components/FormControls'

interface BuiltinDataset { value: string; label: string; task: string[]; desc: string; classes: number; samples: string; size: string }
const BUILTIN_DATASETS: BuiltinDataset[] = [
  { value: 'MNIST',           label: 'MNIST',                    task: ['classification'], desc: 'Handwritten digits 0–9',                    classes: 10,  samples: '70k',        size: '28×28 grayscale' },
  { value: 'Fashion-MNIST',   label: 'Fashion-MNIST',            task: ['classification'], desc: 'Zalando clothing items',                    classes: 10,  samples: '70k',        size: '28×28 grayscale' },
  { value: 'CIFAR-10',        label: 'CIFAR-10',                 task: ['classification'], desc: 'Natural images, 10 classes',                classes: 10,  samples: '60k',        size: '32×32 RGB'       },
  { value: 'CIFAR-100',       label: 'CIFAR-100',                task: ['classification'], desc: 'Natural images, 100 fine-grained classes',  classes: 100, samples: '60k',        size: '32×32 RGB'       },
  { value: 'VOC2012',         label: 'VOC 2012 Segmentation',    task: ['segmentation'],  desc: 'PASCAL VOC semantic segmentation masks',   classes: 21,  samples: '11k',        size: 'Variable RGB'    },
  { value: 'Oxford-IIIT-Pet', label: 'Oxford-IIIT Pet (Seg)',    task: ['segmentation'],  desc: '37 pet breeds with pixel-level masks',     classes: 37,  samples: '7.4k',       size: 'Variable RGB'    },
  { value: 'COCO-Detection',  label: 'COCO (Detection)',         task: ['detection'],     desc: 'Common Objects in Context, 80 categories', classes: 80,  samples: '118k train', size: 'Variable RGB'    },
  { value: 'VOC2012-Det',     label: 'VOC 2012 Detection',       task: ['detection'],     desc: 'PASCAL VOC bbox detection',                classes: 20,  samples: '11k',        size: 'Variable RGB'    },
]
const TASKS = [
  { value: 'classification', label: '🔍 Classification' },
  { value: 'detection',      label: '📦 Object Detection' },
  { value: 'segmentation',   label: '🎨 Segmentation' },
]

interface DatasetConfig { name: string; task_type: string; source: 'builtin' | 'custom' }

export default function DatasetPage() {
  const [mode,     setMode]     = useState<'builtin' | 'custom'>('builtin')
  const [taskType, setTaskType] = useState('classification')
  const [builtin,  setBuiltin]  = useState('MNIST')
  const [path,     setPath]     = useState('')
  const [saved,    setSaved]    = useState<DatasetConfig | null>(null)

  const filtered = BUILTIN_DATASETS.filter(d => d.task.includes(taskType))
  const selectedInfo = BUILTIN_DATASETS.find(d => d.value === builtin)

  function onTaskChange(t: string) {
    setTaskType(t)
    const first = BUILTIN_DATASETS.find(d => d.task.includes(t))
    if (first) setBuiltin(first.value)
  }

  function handleSave() {
    const cfg: DatasetConfig = { name: mode === 'builtin' ? builtin : path, task_type: taskType, source: mode }
    sessionStorage.setItem('dataset_config', JSON.stringify(cfg))
    setSaved(cfg)
  }

  return (
    <div className="p-8 space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Database size={22} className="text-green-400" /> Dataset
        </h1>
        <p className="text-sm text-slate-500 mt-1">Configure the dataset for your next training run</p>
      </div>

      {/* Step 1: Task Type */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Step 1 — Select Task Type</h2>
        <div className="flex gap-2 flex-wrap">
          {TASKS.map(t => (
            <button key={t.value} onClick={() => onTaskChange(t.value)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                taskType === t.value ? 'bg-brand-600 text-white' : 'bg-surface-700 text-slate-400 hover:text-white'
              }`}>{t.label}</button>
          ))}
        </div>
      </Card>

      {/* Step 2: Source */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Step 2 — Choose Dataset Source</h2>
        <div className="flex gap-2 mb-4">
          {(['builtin', 'custom'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                mode === m ? 'bg-brand-600 text-white' : 'bg-surface-700 text-slate-400 hover:text-white'
              }`}>{m === 'builtin' ? '📦 Built-in' : '📁 Custom Path'}</button>
          ))}
        </div>

        {mode === 'builtin' ? (
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-2">
              {filtered.map(d => (
                <button key={d.value} onClick={() => setBuiltin(d.value)}
                  className={`flex items-start gap-3 p-3 rounded-lg border text-left transition-all ${
                    builtin === d.value
                      ? 'border-brand-500 bg-brand-500/10'
                      : 'border-surface-600 hover:border-surface-500 bg-surface-800'
                  }`}>
                  <div className="flex-1">
                    <div className="text-sm font-medium text-white">{d.label}</div>
                    <div className="text-xs text-slate-400 mt-0.5">{d.desc}</div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="text-xs text-slate-500">{d.classes} cls</div>
                    <div className="text-xs text-slate-500">{d.samples}</div>
                    <div className="text-xs font-mono text-slate-600">{d.size}</div>
                  </div>
                </button>
              ))}
            </div>
            {selectedInfo && (
              <div className="flex gap-2 p-3 rounded-lg bg-brand-500/10 border border-brand-500/20 text-xs text-brand-300">
                <Info size={14} className="mt-0.5 shrink-0" />
                Built-in datasets are downloaded automatically during training — no files to manage.
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-4">
            <Input label="Dataset Path (on server)" placeholder="/opt/ml-platform/data/my_dataset"
              value={path} onChange={e => setPath(e.target.value)} />
            <div className="border-2 border-dashed border-surface-500 rounded-xl p-8 text-center">
              <Upload size={24} className="mx-auto text-slate-500 mb-2" />
              <p className="text-sm text-slate-500">Drag &amp; drop dataset folder reference</p>
              <p className="text-xs text-slate-600 mt-1">or enter the server-side path above</p>
            </div>
            <div className="space-y-1.5 text-xs text-slate-500">
              <p className="font-medium text-slate-400">Expected directory structure:</p>
              <ul className="space-y-1 pl-3">
                <li>• <span className="text-slate-300">Classification</span> — ImageFolder (class subfolders)</li>
                <li>• <span className="text-slate-300">Detection</span> — YOLO / COCO JSON / Pascal VOC XML</li>
                <li>• <span className="text-slate-300">Segmentation</span> — images/ + masks/ folders</li>
              </ul>
            </div>
          </div>
        )}

        <div className="mt-6">
          <Button onClick={handleSave} icon={<CheckCircle2 size={14} />}
            disabled={mode === 'builtin' ? !builtin : !path}>
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
            <p className="text-xs text-green-400/50 mt-0.5">→ Go to Training to launch a job with this dataset</p>
          </div>
        </div>
      )}
    </div>
  )
}


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
