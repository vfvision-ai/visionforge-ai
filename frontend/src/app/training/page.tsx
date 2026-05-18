'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Cpu, Play, ChevronDown, ChevronUp } from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import { Input, Select } from '@/components/FormControls'
import { submitJob } from '@/lib/api'
import type { TrainingSubmitPayload } from '@/types'

const FRAMEWORKS = [
  { value: 'pytorch',     label: 'PyTorch' },
  { value: 'tensorflow',  label: 'TensorFlow' },
  { value: 'sklearn',     label: 'Scikit-learn' },
]
const TASK_TYPES = [
  { value: 'classification', label: 'Classification' },
  { value: 'detection',      label: 'Object Detection' },
  { value: 'segmentation',   label: 'Segmentation' },
]
const ARCHITECTURES = {
  classification: [
    { value: 'adaptive_cnn',    label: 'AdaptiveCNN (custom)' },
    { value: 'resnet18',        label: 'ResNet-18' },
    { value: 'resnet50',        label: 'ResNet-50' },
    { value: 'mobilenet_v3',    label: 'MobileNet V3' },
    { value: 'efficientnet_b0', label: 'EfficientNet B0' },
  ],
  detection: [
    { value: 'fasterrcnn',      label: 'Faster R-CNN (MobileNet FPN)' },
    { value: 'fcos',            label: 'FCOS ResNet-50 FPN' },
    { value: 'simple_detector', label: 'Simple Detector (baseline)' },
  ],
  segmentation: [
    { value: 'unet',            label: 'UNet' },
    { value: 'deeplabv3',       label: 'DeepLabV3 (ResNet-50)' },
    { value: 'fcn',             label: 'FCN (ResNet-50)' },
  ],
}
const LR_OPTIONS = [
  { value: '0.1',     label: '0.1' },
  { value: '0.01',    label: '0.01' },
  { value: '0.005',   label: '0.005' },
  { value: '0.001',   label: '0.001 (default)' },
  { value: '0.0005',  label: '0.0005' },
  { value: '0.0001',  label: '0.0001' },
  { value: '0.00001', label: '0.00001' },
]
const BATCH_OPTIONS = [
  { value: '8',   label: '8' },
  { value: '16',  label: '16' },
  { value: '32',  label: '32 (default)' },
  { value: '64',  label: '64' },
  { value: '128', label: '128' },
  { value: '256', label: '256' },
]

interface Preset { label: string; icon: string; epochs: number; lr: number; batch: number; hpo: boolean; desc: string }
const PRESETS: Preset[] = [
  { label: 'Debug',     icon: '🐢', epochs: 3,   lr: 0.001,  batch: 32, hpo: false, desc: '3 epochs, quick sanity check' },
  { label: 'Quick',    icon: '🚀', epochs: 10,  lr: 0.001,  batch: 32, hpo: false, desc: '10 epochs, fast exploration' },
  { label: 'Standard', icon: '⭐', epochs: 50,  lr: 0.001,  batch: 32, hpo: false, desc: '50 epochs, balanced run' },
  { label: 'Full',     icon: '💪', epochs: 150, lr: 0.0005, batch: 64, hpo: true,  desc: '150 epochs + HPO, best accuracy' },
]

interface FormState {
  experiment_name: string
  framework: string; task_type: string; dataset_name: string
  architecture: string; epochs: number; learning_rate: number
  batch_size: number; optimize_hyperparams: boolean; n_trials: number
  early_stopping: boolean; patience: number; min_delta: number
}

export default function TrainingPage() {
  const router = useRouter()
  const [form, setForm] = useState<FormState>({
    experiment_name: `exp_${new Date().toISOString().slice(0,10).replace(/-/g,'')}`,
    framework: 'pytorch', task_type: 'classification',
    dataset_name: 'MNIST', architecture: 'adaptive_cnn',
    epochs: 20, learning_rate: 0.001, batch_size: 32,
    optimize_hyperparams: false, n_trials: 20,
    early_stopping: false, patience: 10, min_delta: 0.001,
  })
  const [activePreset,  setActivePreset]  = useState<string | null>(null)
  const [showAdvanced,  setShowAdvanced]  = useState(false)
  const [showSummary,   setShowSummary]   = useState(false)
  const [error,         setError]         = useState('')
  const [loading,       setLoading]       = useState(false)

  useEffect(() => {
    try {
      const cfg = sessionStorage.getItem('dataset_config')
      if (cfg) {
        const d = JSON.parse(cfg)
        setForm(f => ({
          ...f,
          dataset_name: d.name      || f.dataset_name,
          task_type:    d.task_type || f.task_type,
          framework:    d.framework || f.framework,
        }))
      }
    } catch { /* ignore */ }
  }, [])

  const archs = ARCHITECTURES[form.task_type as keyof typeof ARCHITECTURES] || ARCHITECTURES.classification

  function set<K extends keyof FormState>(key: K, val: FormState[K]) {
    setForm(f => ({ ...f, [key]: val }))
  }

  function applyPreset(p: Preset) {
    setForm(f => ({ ...f, epochs: p.epochs, learning_rate: p.lr, batch_size: p.batch, optimize_hyperparams: p.hpo }))
    setActivePreset(p.label)
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault(); setError(''); setLoading(true)
    try {
      const payload: TrainingSubmitPayload = {
        framework:            form.framework as TrainingSubmitPayload['framework'],
        task_type:            form.task_type as TrainingSubmitPayload['task_type'],
        dataset_name:         form.dataset_name,
        architecture:         form.architecture,
        epochs:               form.epochs,
        learning_rate:        form.learning_rate,
        batch_size:           form.batch_size,
        optimize_hyperparams: form.optimize_hyperparams,
        early_stopping:       form.early_stopping,
        patience:             form.patience,
        min_delta:            form.min_delta,
        n_trials:             form.n_trials,
        experiment_name:      form.experiment_name,
        dataset_config:       {},
      }
      const job = await submitJob(payload)
      router.push(`/results/${job.id}`)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Submission failed')
    } finally { setLoading(false) }
  }

  const estMinutes = Math.max(1, Math.round(form.epochs * form.batch_size / 500))

  return (
    <div className="p-8 space-y-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Cpu size={22} className="text-brand-400" /> Training
        </h1>
        <p className="text-sm text-slate-500 mt-1">Configure and launch a training job</p>
      </div>

      {/* Quick Presets */}
      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-3">Quick Presets</h2>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {PRESETS.map(p => (
            <button key={p.label} type="button" onClick={() => applyPreset(p)}
              className={`p-3 rounded-lg border text-left transition-all ${
                activePreset === p.label
                  ? 'border-brand-500 bg-brand-500/10'
                  : 'border-surface-600 hover:border-surface-500 bg-surface-800'
              }`}>
              <div className="text-xl mb-1">{p.icon}</div>
              <div className="text-xs font-semibold text-white">{p.label}</div>
              <div className="text-xs text-slate-500 mt-0.5">{p.desc}</div>
            </button>
          ))}
        </div>
      </Card>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Experiment & Dataset */}
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Experiment &amp; Dataset</h2>
          <div className="grid grid-cols-2 gap-4">
            <Input label="Experiment Name" value={form.experiment_name}
              onChange={e => set('experiment_name', e.target.value)} placeholder="my_experiment" />
            <Input label="Dataset Name / Path" value={form.dataset_name}
              onChange={e => set('dataset_name', e.target.value)}
              placeholder="MNIST, CIFAR-10, /path/to/data" required />
            <Select label="Task Type" value={form.task_type}
              onChange={e => { set('task_type', e.target.value); set('architecture', ARCHITECTURES[e.target.value as keyof typeof ARCHITECTURES]?.[0]?.value || 'adaptive_cnn') }}
              options={TASK_TYPES} />
            <Select label="Framework" value={form.framework}
              onChange={e => set('framework', e.target.value)} options={FRAMEWORKS} />
          </div>
        </Card>

        {/* Model */}
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Model Architecture</h2>
          <Select label="Architecture" value={form.architecture}
            onChange={e => set('architecture', e.target.value)} options={archs} />
        </Card>

        {/* Hyperparameters */}
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Hyperparameters</h2>
          <div className="grid grid-cols-3 gap-4 items-end">
            <div>
              <label className="block text-xs text-slate-400 mb-1.5">Epochs</label>
              <input type="range" min={1} max={200} step={1} value={form.epochs}
                onChange={e => set('epochs', Number(e.target.value))}
                className="w-full accent-brand-500" />
              <div className="text-xs text-brand-400 mt-1 font-mono text-center">{form.epochs}</div>
            </div>
            <Select label="Learning Rate" value={String(form.learning_rate)}
              onChange={e => set('learning_rate', Number(e.target.value))} options={LR_OPTIONS} />
            <Select label="Batch Size" value={String(form.batch_size)}
              onChange={e => set('batch_size', Number(e.target.value))} options={BATCH_OPTIONS} />
          </div>
          <div className="mt-4 space-y-3">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" className="w-4 h-4 accent-brand-500"
                checked={form.optimize_hyperparams}
                onChange={e => set('optimize_hyperparams', e.target.checked)} />
              <span className="text-sm text-slate-300">Auto-optimize hyperparameters (Optuna)</span>
            </label>
            {form.optimize_hyperparams && (
              <div className="ml-6">
                <label className="block text-xs text-slate-400 mb-1.5">
                  Optimization Trials: <span className="text-brand-400 font-mono">{form.n_trials}</span>
                </label>
                <input type="range" min={5} max={50} step={5} value={form.n_trials}
                  onChange={e => set('n_trials', Number(e.target.value))}
                  className="w-48 accent-brand-500" />
              </div>
            )}
          </div>
        </Card>

        {/* Advanced — Early Stopping */}
        <Card>
          <button type="button"
            className="flex items-center justify-between w-full text-sm font-semibold text-slate-300"
            onClick={() => setShowAdvanced(s => !s)}>
            <span>Advanced Options (Early Stopping)</span>
            {showAdvanced ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {showAdvanced && (
            <div className="mt-4 space-y-4 border-t border-surface-700 pt-4">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="w-4 h-4 accent-brand-500"
                  checked={form.early_stopping}
                  onChange={e => set('early_stopping', e.target.checked)} />
                <span className="text-sm text-slate-300">Enable Early Stopping</span>
              </label>
              {form.early_stopping && (
                <div className="ml-6 grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1.5">
                      Patience: <span className="text-brand-400 font-mono">{form.patience}</span>
                    </label>
                    <input type="range" min={3} max={30} step={1} value={form.patience}
                      onChange={e => set('patience', Number(e.target.value))}
                      className="w-full accent-brand-500" />
                  </div>
                  <Input label="Min Improvement (δ)" type="number" step="any" min={0}
                    value={form.min_delta} onChange={e => set('min_delta', Number(e.target.value))} />
                </div>
              )}
            </div>
          )}
        </Card>

        {/* Config Summary */}
        <Card>
          <button type="button"
            className="flex items-center justify-between w-full text-sm font-semibold text-slate-300"
            onClick={() => setShowSummary(s => !s)}>
            <span>Configuration Summary</span>
            {showSummary ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {showSummary && (
            <div className="mt-4 grid grid-cols-2 gap-x-8 gap-y-1.5 border-t border-surface-700 pt-4 text-xs">
              {([
                ['Experiment', form.experiment_name], ['Dataset', form.dataset_name],
                ['Task', form.task_type],             ['Framework', form.framework],
                ['Architecture', form.architecture],  ['Epochs', form.epochs],
                ['Learning Rate', form.learning_rate],['Batch Size', form.batch_size],
                ['HPO', form.optimize_hyperparams ? `Yes (${form.n_trials} trials)` : 'No'],
                ['Early Stopping', form.early_stopping ? `Yes (patience=${form.patience})` : 'No'],
                ['Est. Time', `~${estMinutes} min`],
              ] as [string, string | number][]).map(([k, v]) => (
                <div key={k} className="flex justify-between border-b border-surface-800 pb-1">
                  <span className="text-slate-500">{k}</span>
                  <span className="text-slate-200 font-mono">{String(v)}</span>
                </div>
              ))}
            </div>
          )}
        </Card>

        {error && (
          <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">{error}</div>
        )}

        <Button type="submit" size="lg" loading={loading} icon={<Play size={15} />} className="w-full justify-center">
          {loading ? 'Submitting…' : 'Launch Training Job'}
        </Button>
      </form>
    </div>
  )
}
