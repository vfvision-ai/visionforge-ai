'use client'
import { useState } from 'react'
import { Database, Upload, CheckCircle2, Info, ExternalLink, Key, Eye, EyeOff } from 'lucide-react'
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
  task: TaskType[]; icon: string; badge?: string
}

interface DatasetConfig {
  name: string; task_type: TaskType; framework: Framework; source: Source
  hf_subset?: string; hf_token?: string
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
  // Classification
  { name: 'cifar10',                               label: 'CIFAR-10',             icon: '🚗', badge: 'popular',  task: ['classification'], desc: 'Classic benchmark — vehicles, animals, objects', samples: '60K',  classes: '10',  size: '32×32'    },
  { name: 'fashion_mnist',                         label: 'Fashion-MNIST',        icon: '👕', badge: 'popular',  task: ['classification'], desc: 'Clothing classification, 10 categories',          samples: '70K',  classes: '10',  size: '28×28'    },
  { name: 'food101',                               label: 'Food-101',             icon: '🍕',                    task: ['classification'], desc: '101 food types from restaurant photos',           samples: '101K', classes: '101', size: 'Variable' },
  { name: 'cats_vs_dogs',                          label: 'Cats vs Dogs',         icon: '🐱',                    task: ['classification'], desc: 'Binary classification — cat or dog',              samples: '23K',  classes: '2',   size: 'Variable' },
  { name: 'keremberke/indoor-scene-classification',label: 'Indoor Scenes',        icon: '🏠',                    task: ['classification'], desc: '67 indoor scene categories',                     samples: '15K',  classes: '67',  size: 'Variable' },
  // Segmentation
  { name: 'oxford_iiit_pet',                       label: 'Oxford-IIIT Pet',      icon: '🐾', badge: 'verified', task: ['segmentation'],   desc: '37 breeds + pixel-level trimap segmentation',    samples: '7.4K', classes: '37',  size: 'Variable' },
  { name: 'scene_parse_150',                       label: 'ADE20K Scene Parsing', icon: '🏙️',                    task: ['segmentation'],   desc: 'Scene parsing — 150 object categories',          samples: '22K',  classes: '150', size: 'Variable' },
  // Detection
  { name: 'detection-datasets/coco',               label: 'MS COCO',              icon: '📦', badge: 'popular',  task: ['detection'],      desc: 'Common Objects in Context, 80 categories',       samples: '118K', classes: '80',  size: 'Variable' },
  { name: 'keremberke/vehicle-detection',          label: 'Vehicle Detection',    icon: '🚗',                    task: ['detection'],      desc: 'Cars, bikes, trucks bounding boxes',             samples: '10K',  classes: '4',   size: 'Variable' },
  { name: 'keremberke/face-mask-detection',        label: 'Face Mask Detection',  icon: '😷',                    task: ['detection'],      desc: 'Mask / no-mask / improper — PPE safety',         samples: '5K',   classes: '3',   size: 'Variable' },
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
  const [hfMode,    setHfMode]    = useState<'curated' | 'manual'>('curated')
  const [hfManual,  setHfManual]  = useState('')
  const [hfSubset,  setHfSubset]  = useState('')
  const [hfAuth,    setHfAuth]    = useState<'none' | 'token' | 'env'>('none')
  const [hfToken,   setHfToken]   = useState('')
  const [hfEnvVar,  setHfEnvVar]  = useState('HF_TOKEN')
  const [showToken, setShowToken] = useState(false)
  const [path,      setPath]      = useState('')
  const [saved,     setSaved]     = useState<DatasetConfig | null>(null)

  function onFrameworkChange(fw: Framework) {
    setFramework(fw)
    const first = sources(fw)[0].value
    setSource(first)
  }

  function onTaskChange(t: TaskType) {
    setTaskType(t)
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

  const hfName = hfMode === 'curated' ? hfDataset : hfManual

  function handleSave() {
    let name = ''
    if (source === 'pytorch' || source === 'tensorflow') name = builtin
    else if (source === 'huggingface') name = hfName
    else name = path

    const cfg: DatasetConfig = {
      name, task_type: taskType, framework, source,
      hf_subset: hfSubset || undefined,
      hf_token:  hfAuth === 'token' ? hfToken
               : hfAuth === 'env'   ? `env:${hfEnvVar}`
               : undefined,
    }
    sessionStorage.setItem('dataset_config', JSON.stringify(cfg))
    setSaved(cfg)
  }

  const isValid = source === 'custom' ? !!path : source === 'huggingface' ? !!hfName : !!builtin

  const builtinPool  = (source === 'tensorflow' ? TF_DATASETS : PYTORCH_DATASETS).filter(d => d.task.includes(taskType))
  const hfPool       = HF_DATASETS.filter(d => d.task.includes(taskType))
  const selectedBuiltin = builtinPool.find(d => d.value === builtin)
  const selectedHF      = hfPool.find(d => d.name === hfDataset)
  const sourceOptions   = sources(framework)

  function badgeClass(badge?: string) {
    if (badge === 'popular')  return 'bg-orange-500/20 text-orange-300'
    if (badge === 'verified') return 'bg-green-500/20  text-green-300'
    return 'bg-brand-500/20 text-brand-300'
  }

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
          <div className="space-y-4">
            {/* Mode tabs */}
            <div className="flex gap-1 p-1 bg-surface-900 rounded-lg w-fit">
              {(['curated', 'manual'] as const).map(m => (
                <button key={m} onClick={() => setHfMode(m)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    hfMode === m ? 'bg-brand-600 text-white' : 'text-slate-400 hover:text-white'
                  }`}>
                  {m === 'curated' ? '📋 Curated (recommended)' : '✏️ Manual Entry'}
                </button>
              ))}
            </div>

            {/* Curated list */}
            {hfMode === 'curated' && (
              <div className="space-y-2">
                {hfPool.length === 0 ? (
                  <div className="p-4 rounded-lg bg-yellow-500/10 border border-yellow-500/20 text-sm text-yellow-400">
                    No curated datasets for <strong>{taskType}</strong>. Switch to Manual Entry.
                  </div>
                ) : hfPool.map(d => (
                  <button key={d.name} onClick={() => setHfDataset(d.name)}
                    className={`w-full flex items-start gap-3 p-3 rounded-lg border text-left transition-all ${
                      hfDataset === d.name
                        ? 'border-brand-500 bg-brand-500/10'
                        : 'border-surface-600 hover:border-surface-500 bg-surface-800'
                    }`}>
                    <span className="text-xl mt-0.5">{d.icon}</span>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm font-medium text-white">{d.label}</span>
                        {d.badge && (
                          <span className={`text-xs px-1.5 py-0.5 rounded font-medium ${badgeClass(d.badge)}`}>{d.badge}</span>
                        )}
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

            {/* Manual entry */}
            {hfMode === 'manual' && (
              <div className="space-y-3">
                <Input label="Dataset ID" placeholder="e.g. imagenet-1k, nielsr/cifar10-demo"
                  value={hfManual} onChange={e => setHfManual(e.target.value)} />
                <a href="https://huggingface.co/datasets" target="_blank" rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-brand-400 hover:text-brand-300">
                  <ExternalLink size={11} /> Browse all datasets on HuggingFace Hub
                </a>
              </div>
            )}

            {/* Subset / split field for both modes */}
            <Input label="Subset / Split override (optional)" placeholder="e.g. train, train[:5000], default"
              value={hfSubset} onChange={e => setHfSubset(e.target.value)} />

            {/* Auth */}
            <div className="border-t border-surface-700 pt-4 space-y-3">
              <div className="flex items-center gap-2">
                <Key size={13} className="text-slate-400" />
                <span className="text-xs font-medium text-slate-300">Authentication (for private or gated datasets)</span>
              </div>
              <div className="flex gap-2 flex-wrap">
                {(['none', 'token', 'env'] as const).map(a => (
                  <button key={a} onClick={() => setHfAuth(a)}
                    className={`px-3 py-1.5 rounded-md text-xs border transition-colors ${
                      hfAuth === a
                        ? 'border-brand-500 bg-brand-500/10 text-white'
                        : 'border-surface-600 text-slate-400 hover:text-white'
                    }`}>
                    {a === 'none' ? 'None (public)' : a === 'token' ? '🔑 HF Token' : '📌 Env Variable'}
                  </button>
                ))}
              </div>
              {hfAuth === 'token' && (
                <div className="relative">
                  <input
                    type={showToken ? 'text' : 'password'}
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxx"
                    value={hfToken}
                    onChange={e => setHfToken(e.target.value)}
                    className="w-full bg-surface-800 border border-surface-600 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-brand-500 pr-10"
                  />
                  <button type="button" onClick={() => setShowToken(s => !s)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white">
                    {showToken ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
              )}
              {hfAuth === 'env' && (
                <Input label="Environment variable name" placeholder="HF_TOKEN"
                  value={hfEnvVar} onChange={e => setHfEnvVar(e.target.value)} />
              )}
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
            Save &amp; Continue to Training
          </Button>
        </div>
      </Card>

      {saved && (
        <div className="flex items-start gap-3 p-4 rounded-xl bg-green-500/10 border border-green-500/20">
          <CheckCircle2 size={18} className="text-green-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-green-300">Dataset configured: <span className="font-bold">{saved.name}</span></p>
            <p className="text-xs text-green-400/70 mt-0.5">
              Task: {saved.task_type} · Framework: {saved.framework} · Source: {saved.source}
              {saved.hf_subset && <> · Subset: <span className="font-mono">{saved.hf_subset}</span></>}
              {saved.hf_token  && <> · Auth: {saved.hf_token.startsWith('env:') ? `env (${saved.hf_token.slice(4)})` : '🔑 token set'}</>}
            </p>
            <a href="/training" className="inline-block text-xs text-green-400 hover:text-green-300 mt-1">
              🚀 Go to Training →
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
