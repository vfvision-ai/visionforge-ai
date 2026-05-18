'use client'
import { useState } from 'react'
import { Database, Upload, CheckCircle2, Info, ExternalLink } from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import { Input, Select } from '@/components/FormControls'

// ── Types ─────────────────────────────────────────────────────────────────────
type Framework = 'pytorch' | 'tensorflow' | 'sklearn'
type TaskType  = 'classification' | 'detection' | 'segmentation'
type Source    = 'pytorch' | 'tensorflow' | 'huggingface' | 'custom'

interface DatasetEntry {
  value: string; label: string; desc: string
  classes: number; samples: string; size: string
  task: TaskType[]; icon: string
}

interface HFEntry {
  name: string; label: string; desc: string
  samples: string; classes: string; size: string
  task: TaskType[]; icon: string; verified: boolean
}

interface DatasetConfig {
  name: string; task_type: TaskType; framework: Framework; source: Source
}

// ── Static data ───────────────────────────────────────────────────────────────
const FRAMEWORKS: { value: Framework; label: string; icon: string }[] = [
  { value: 'pytorch',     label: 'PyTorch',           icon: '🔥' },
  { value: 'tensorflow',  label: 'TensorFlow / Keras', icon: '🧠' },
  { value: 'sklearn',     label: 'Scikit-learn',       icon: '⚙️' },
]

const TASKS: { value: TaskType; label: string }[] = [
  { value: 'classification', label: '🔍 Classification'   },
  { value: 'detection',      label: '📦 Object Detection' },
  { value: 'segmentation',   label: '🎨 Segmentation'     },
]

const PYTORCH_DATASETS: DatasetEntry[] = [
  { value: 'MNIST',         label: 'MNIST',                   icon: '🔢', task: ['classification'], desc: 'Handwritten digits 0–9',                    classes: 10,  samples: '70k',   size: '28×28 grayscale' },
  { value: 'Fashion-MNIST', label: 'Fashion-MNIST',           icon: '👕', task: ['classification'], desc: 'Zalando clothing items',                    classes: 10,  samples: '70k',   size: '28×28 grayscale' },
  { value: 'CIFAR-10',      label: 'CIFAR-10',                icon: '🎯', task: ['classification'], desc: 'Natural images, 10 classes',                classes: 10,  samples: '60k',   size: '32×32 RGB'       },
  { value: 'CIFAR-100',     label: 'CIFAR-100',               icon: '🎨', task: ['classification'], desc: 'Natural images, 100 fine-grained classes',  classes: 100, samples: '60k',   size: '32×32 RGB'       },
  { value: 'VOC2012',       label: 'VOC 2012 Segmentation',   icon: '🎨', task: ['segmentation'],   desc: 'PASCAL VOC semantic segmentation masks',   classes: 21,  samples: '11k',   size: 'Variable RGB'    },
  { value: 'Oxford-IIIT-Pet', label: 'Oxford-IIIT Pet',       icon: '🐕', task: ['segmentation'],   desc: '37 pet breeds with pixel-level masks',     classes: 37,  samples: '7.4k',  size: 'Variable RGB'    },
  { value: 'COCO-Detection',  label: 'COCO Detection',        icon: '📦', task: ['detection'],      desc: 'Common Objects in Context, 80 categories', classes: 80,  samples: '118k',  size: 'Variable RGB'    },
  { value: 'VOC2012-Det',   label: 'VOC 2012 Detection',      icon: '📦', task: ['detection'],      desc: 'PASCAL VOC bounding-box detection',        classes: 20,  samples: '11k',   size: 'Variable RGB'    },
]

const TF_DATASETS: DatasetEntry[] = [
  { value: 'MNIST',         label: 'MNIST',                   icon: '🔢', task: ['classification'], desc: 'Handwritten digits 0–9 (tf.keras.datasets)', classes: 10,  samples: '70k',  size: '28×28 grayscale' },
  { value: 'Fashion-MNIST', label: 'Fashion-MNIST',           icon: '👕', task: ['classification'], desc: 'Clothing items (tf.keras.datasets)',          classes: 10,  samples: '70k',  size: '28×28 grayscale' },
  { value: 'CIFAR-10',      label: 'CIFAR-10',                icon: '🎯', task: ['classification'], desc: 'Natural images, 10 classes (tf.keras)',       classes: 10,  samples: '60k',  size: '32×32 RGB'       },
  { value: 'CIFAR-100',     label: 'CIFAR-100',               icon: '🎨', task: ['classification'], desc: 'Natural images, 100 classes (tf.keras)',      classes: 100, samples: '60k',  size: '32×32 RGB'       },
  { value: 'Oxford-IIIT-Pet', label: 'Oxford-IIIT Pet (Seg)', icon: '🐕', task: ['segmentation'],   desc: 'Pet segmentation masks (tensorflow_datasets)', classes: 3,  samples: '7.4k', size: '128×128 RGB'     },
]

const HF_DATASETS: HFEntry[] = [
  // Classification
  { name: 'cifar10',                            label: 'CIFAR-10',             icon: '🚗', task: ['classification'], desc: 'Classic image classification — vehicles, animals, objects', samples: '60K', classes: '10', size: '32×32',     verified: true  },
  { name: 'fashion_mnist',                      label: 'Fashion Items',        icon: '👕', task: ['classification'], desc: 'Fashion and clothing items classification',                 samples: '70K', classes: '10', size: '28×28',     verified: true  },
  { name: 'food101',                            label: 'Food-101',             icon: '🍕', task: ['classification'], desc: 'Food and cuisine classification',                           samples: '101K',classes: '101',size: 'Variable',  verified: true  },
  { name: 'cats_vs_dogs',                       label: 'Cats vs Dogs',         icon: '🐱', task: ['classification'], desc: 'Binary classification — cats vs dogs',                      samples: '23K', classes: '2',  size: 'Variable',  verified: true  },
  { name: 'keremberke/indoor-scene-classification', label: 'Indoor Scenes',   icon: '🏠', task: ['classification'], desc: 'Indoor scene recognition, 67 categories',                   samples: '15K', classes: '67', size: 'Variable',  verified: true  },
  // Segmentation
  { name: 'oxford_iiit_pet',                    label: 'Oxford-IIIT Pet',      icon: '🐾', task: ['segmentation'],   desc: 'Pet breeds with pixel-level segmentation masks',            samples: '7.4K',classes: '37', size: 'Variable',  verified: true  },
  { name: 'scene_parse_150',                    label: 'ADE20K Scene Parsing', icon: '🏙️', task: ['segmentation'],   desc: 'Scene parsing, 150 object categories',                     samples: '22K', classes: '150',size: 'Variable',  verified: true  },
  // Detection
  { name: 'detection-datasets/coco',            label: 'COCO',                 icon: '📦', task: ['detection'],      desc: 'Common Objects in Context, standard detection benchmark',  samples: '118K',classes: '80', size: 'Variable',  verified: true  },
  { name: 'keremberke/vehicle-detection',       label: 'Vehicle Detection',    icon: '🚗', task: ['detection'],      desc: 'Detect vehicles — cars, bikes, trucks',                    samples: '10K', classes: '4',  size: 'Variable',  verified: true  },
  { name: 'keremberke/hard-hat-detection',      label: 'Hard Hat Detection',   icon: '⛑️',  task: ['detection'],      desc: 'Safety hard-hat detection for PPE compliance',             samples: '5K',  classes: '2',  size: 'Variable',  verified: true  },
]

// ── Helpers ───────────────────────────────────────────────────────────────────
function sources(fw: Framework): { value: Source; label: string }[] {
  const base: { value: Source; label: string }[] = []
  if (fw === 'pytorch')    base.push({ value: 'pytorch',    label: '🔥 Built-in PyTorch Datasets'     })
  if (fw === 'tensorflow') base.push({ value: 'tensorflow', label: '🧠 Built-in TensorFlow Datasets'  })
  base.push({ value: 'huggingface', label: '🤗 HuggingFace Dataset Hub' })
  base.push({ value: 'custom',      label: '📁 Custom / Server Path'    })
  return base
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function DatasetPage() {
  const [framework, setFramework] = useState<Framework>('pytorch')
  const [taskType,  setTaskType]  = useState<TaskType>('classification')
  const [source,    setSource]    = useState<Source>('pytorch')
  const [builtin,   setBuiltin]   = useState('MNIST')
  const [hfDataset, setHfDataset] = useState('cifar10')
  const [path,      setPath]      = useState('')
  const [saved,     setSaved]     = useState<DatasetConfig | null>(null)

  function onFrameworkChange(fw: Framework) {
    setFramework(fw)
    // reset source to the first available for the new framework
    const first = sources(fw)[0].value
    setSource(first)
  }

  function onTaskChange(t: TaskType) {
    setTaskType(t)
    // keep builtin valid
    const pool = source === 'tensorflow' ? TF_DATASETS : PYTORCH_DATASETS
    const first = pool.find(d => d.task.includes(t))
    if (first) setBuiltin(first.value)
    const firstHf = HF_DATASETS.find(d => d.task.includes(t))
    if (firstHf) setHfDataset(firstHf.name)
  }

  function onSourceChange(s: Source) {
    setSource(s)
    const pool = s === 'tensorflow' ? TF_DATASETS : PYTORCH_DATASETS
    const first = pool.find(d => d.task.includes(taskType))
    if (first) setBuiltin(first.value)
  }

  function handleSave() {
    let name = ''
    if (source === 'pytorch' || source === 'tensorflow') name = builtin
    else if (source === 'huggingface') name = hfDataset
    else name = path

    const cfg: DatasetConfig = { name, task_type: taskType, framework, source }
    sessionStorage.setItem('dataset_config', JSON.stringify(cfg))
    setSaved(cfg)
  }

  const isValid = source === 'custom' ? !!path : source === 'huggingface' ? !!hfDataset : !!builtin

  const builtinPool  = (source === 'tensorflow' ? TF_DATASETS : PYTORCH_DATASETS).filter(d => d.task.includes(taskType))
  const hfPool       = HF_DATASETS.filter(d => d.task.includes(taskType))
  const selectedBuiltin = builtinPool.find(d => d.value === builtin)
  const selectedHF      = hfPool.find(d => d.name === hfDataset)
  const sourceOptions   = sources(framework)

  return (
    <div className="p-8 space-y-6 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Database size={22} className="text-green-400" /> Dataset
        </h1>
        <p className="text-sm text-slate-500 mt-1">Configure the dataset for your next training run</p>
      </div>

      {/* Step 1: Framework */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Step 1 — Select Framework</h2>
        <div className="flex gap-2 flex-wrap">
          {FRAMEWORKS.map(fw => (
            <button key={fw.value} onClick={() => onFrameworkChange(fw.value)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                framework === fw.value ? 'bg-brand-600 text-white' : 'bg-surface-700 text-slate-400 hover:text-white'
              }`}>{fw.icon} {fw.label}</button>
          ))}
        </div>
      </Card>

      {/* Step 2: Task Type */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Step 2 — Select Task Type</h2>
        <div className="flex gap-2 flex-wrap">
          {TASKS.map(t => (
            <button key={t.value} onClick={() => onTaskChange(t.value)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                taskType === t.value ? 'bg-brand-600 text-white' : 'bg-surface-700 text-slate-400 hover:text-white'
              }`}>{t.label}</button>
          ))}
        </div>
      </Card>

      {/* Step 3: Source + Dataset */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Step 3 — Choose Dataset Source</h2>
        <div className="flex gap-2 flex-wrap mb-5">
          {sourceOptions.map(s => (
            <button key={s.value} onClick={() => onSourceChange(s.value)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                source === s.value ? 'bg-brand-600 text-white' : 'bg-surface-700 text-slate-400 hover:text-white'
              }`}>{s.label}</button>
          ))}
        </div>

        {/* Built-in (PyTorch or TF) */}
        {(source === 'pytorch' || source === 'tensorflow') && (
          <div className="space-y-3">
            {builtinPool.length === 0 ? (
              <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-sm text-yellow-400">
                No built-in datasets for this task type. Try HuggingFace or a custom path.
              </div>
            ) : (
              <>
                <div className="grid grid-cols-1 gap-2">
                  {builtinPool.map(d => (
                    <button key={d.value} onClick={() => setBuiltin(d.value)}
                      className={`flex items-start gap-3 p-3 rounded-lg border text-left transition-all ${
                        builtin === d.value
                          ? 'border-brand-500 bg-brand-500/10'
                          : 'border-surface-600 hover:border-surface-500 bg-surface-800'
                      }`}>
                      <span className="text-xl mt-0.5">{d.icon}</span>
                      <div className="flex-1 min-w-0">
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
                {selectedBuiltin && (
                  <div className="flex gap-2 p-3 rounded-lg bg-brand-500/10 border border-brand-500/20 text-xs text-brand-300">
                    <Info size={14} className="mt-0.5 shrink-0" />
                    Downloaded automatically during training — no manual setup needed.
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* HuggingFace */}
        {source === 'huggingface' && (
          <div className="space-y-3">
            {hfPool.length === 0 ? (
              <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-sm text-yellow-400">
                No curated HuggingFace datasets for this task type. Use a custom HF dataset ID below.
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-2">
                {hfPool.map(d => (
                  <button key={d.name} onClick={() => setHfDataset(d.name)}
                    className={`flex items-start gap-3 p-3 rounded-lg border text-left transition-all ${
                      hfDataset === d.name
                        ? 'border-brand-500 bg-brand-500/10'
                        : 'border-surface-600 hover:border-surface-500 bg-surface-800'
                    }`}>
                    <span className="text-xl mt-0.5">{d.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-white">{d.label}</span>
                        {d.verified && <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">✓ verified</span>}
                      </div>
                      <div className="text-xs text-slate-400 mt-0.5">{d.desc}</div>
                      <div className="text-xs font-mono text-slate-600 mt-0.5">{d.name}</div>
                    </div>
                    <div className="text-right shrink-0">
                      <div className="text-xs text-slate-500">{d.classes} cls</div>
                      <div className="text-xs text-slate-500">{d.samples}</div>
                      <div className="text-xs font-mono text-slate-600">{d.size}</div>
                    </div>
                  </button>
                ))}
              </div>
            )}
            <div className="space-y-2">
              <Input
                label="Or enter a custom HuggingFace dataset ID"
                placeholder="e.g. imagenet-1k, nielsr/cifar10"
                value={hfPool.find(d => d.name === hfDataset) ? '' : hfDataset}
                onChange={e => setHfDataset(e.target.value)}
              />
              <a href="https://huggingface.co/datasets" target="_blank" rel="noreferrer"
                className="inline-flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300">
                <ExternalLink size={11} /> Browse all datasets on HuggingFace Hub
              </a>
            </div>
          </div>
        )}

        {/* Custom path */}
        {source === 'custom' && (
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
          <Button onClick={handleSave} icon={<CheckCircle2 size={14} />} disabled={!isValid}>
            Save Dataset Config
          </Button>
        </div>
      </Card>

      {saved && (
        <div className="flex items-center gap-3 p-4 rounded-xl bg-green-500/10 border border-green-500/20">
          <CheckCircle2 size={18} className="text-green-400 shrink-0" />
          <div>
            <p className="text-sm font-medium text-green-300">Dataset configured: <span className="font-bold">{saved.name}</span></p>
            <p className="text-xs text-green-400/70 mt-0.5">
              Task: {saved.task_type} · Framework: {saved.framework} · Source: {saved.source}
            </p>
            <p className="text-xs text-green-400/50 mt-0.5">→ Go to Training to launch a job with this dataset</p>
          </div>
        </div>
      )}
    </div>
  )
}


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
