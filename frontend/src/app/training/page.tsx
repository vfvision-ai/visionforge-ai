'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { Cpu, Play } from 'lucide-react'
import Card from '@/components/Card'
import Button from '@/components/Button'
import { Input, Select } from '@/components/FormControls'
import { submitJob } from '@/lib/api'
import type { TrainingSubmitPayload } from '@/types'

const FRAMEWORKS    = [{ value: 'pytorch', label: 'PyTorch' }, { value: 'tensorflow', label: 'TensorFlow' }, { value: 'sklearn', label: 'Scikit-learn' }]
const TASK_TYPES    = [{ value: 'classification', label: 'Classification' }, { value: 'detection', label: 'Object Detection' }, { value: 'segmentation', label: 'Segmentation' }]
const ARCHITECTURES = {
  classification: [
    { value: 'adaptive_cnn',   label: 'AdaptiveCNN (custom)'  },
    { value: 'resnet18',       label: 'ResNet-18'             },
    { value: 'resnet50',       label: 'ResNet-50'             },
    { value: 'mobilenet_v3',   label: 'MobileNet V3'          },
    { value: 'efficientnet_b0',label: 'EfficientNet B0'       },
  ],
  detection: [
    { value: 'fasterrcnn',     label: 'Faster R-CNN (MobileNet FPN)' },
    { value: 'fcos',           label: 'FCOS ResNet-50 FPN'           },
    { value: 'simple_detector',label: 'Simple Detector (baseline)'   },
  ],
  segmentation: [
    { value: 'unet',           label: 'UNet'                  },
    { value: 'deeplabv3',      label: 'DeepLabV3 (ResNet-50)' },
    { value: 'fcn',            label: 'FCN (ResNet-50)'        },
  ],
}

export default function TrainingPage() {
  const router = useRouter()

  const [form, setForm] = useState<{
    framework: string; task_type: string; dataset_name: string
    architecture: string; epochs: number; learning_rate: number
    batch_size: number; optimize_hyperparams: boolean
  }>({
    framework: 'pytorch', task_type: 'classification',
    dataset_name: 'MNIST', architecture: 'adaptive_cnn',
    epochs: 20, learning_rate: 0.001, batch_size: 32,
    optimize_hyperparams: false,
  })
  const [error,   setError]   = useState('')
  const [loading, setLoading] = useState(false)

  // load dataset config from previous page
  useEffect(() => {
    try {
      const cfg = sessionStorage.getItem('dataset_config')
      if (cfg) {
        const d = JSON.parse(cfg)
        setForm(f => ({ ...f, dataset_name: d.name, task_type: d.task_type || f.task_type }))
      }
    } catch { /* ignore */ }
  }, [])

  const archs = ARCHITECTURES[form.task_type as keyof typeof ARCHITECTURES] || ARCHITECTURES.classification

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      const payload: TrainingSubmitPayload = {
        framework:           form.framework as TrainingSubmitPayload['framework'],
        task_type:           form.task_type as TrainingSubmitPayload['task_type'],
        dataset_name:        form.dataset_name,
        architecture:        form.architecture,
        epochs:              form.epochs,
        learning_rate:       form.learning_rate,
        batch_size:          form.batch_size,
        optimize_hyperparams:form.optimize_hyperparams,
        dataset_config:      {},
      }
      const job = await submitJob(payload)
      router.push(`/results/${job.id}`)
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Submission failed')
    } finally {
      setLoading(false)
    }
  }

  function set<K extends keyof typeof form>(key: K, val: typeof form[K]) {
    setForm(f => ({ ...f, [key]: val }))
  }

  return (
    <div className="p-8 space-y-8 max-w-2xl">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Cpu size={22} className="text-brand-400" />
          Training
        </h1>
        <p className="text-sm text-slate-500 mt-1">Configure and launch a training job</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Dataset &amp; Task</h2>
          <div className="grid grid-cols-2 gap-4">
            <Input
              label="Dataset Name / Path"
              value={form.dataset_name}
              onChange={e => set('dataset_name', e.target.value)}
              placeholder="MNIST, CIFAR-10, /path/to/data"
              required
            />
            <Select
              label="Task Type"
              value={form.task_type}
              onChange={e => { set('task_type', e.target.value); set('architecture', ARCHITECTURES[e.target.value as keyof typeof ARCHITECTURES]?.[0]?.value || 'adaptive_cnn') }}
              options={TASK_TYPES}
            />
          </div>
        </Card>

        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Model</h2>
          <div className="grid grid-cols-2 gap-4">
            <Select
              label="Framework"
              value={form.framework}
              onChange={e => set('framework', e.target.value)}
              options={FRAMEWORKS}
            />
            <Select
              label="Architecture"
              value={form.architecture}
              onChange={e => set('architecture', e.target.value)}
              options={archs}
            />
          </div>
        </Card>

        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Hyperparameters</h2>
          <div className="grid grid-cols-3 gap-4">
            <Input
              label="Epochs"
              type="number" min={1} max={500}
              value={form.epochs}
              onChange={e => set('epochs', Number(e.target.value))}
            />
            <Input
              label="Learning Rate"
              type="number" step="any" min={0.000001} max={1}
              value={form.learning_rate}
              onChange={e => set('learning_rate', Number(e.target.value))}
            />
            <Input
              label="Batch Size"
              type="number" min={1} max={512}
              value={form.batch_size}
              onChange={e => set('batch_size', Number(e.target.value))}
            />
          </div>

          <label className="flex items-center gap-2 mt-4 cursor-pointer">
            <input
              type="checkbox"
              className="w-4 h-4 accent-brand-500"
              checked={form.optimize_hyperparams}
              onChange={e => set('optimize_hyperparams', e.target.checked)}
            />
            <span className="text-sm text-slate-400">Auto-optimize hyperparameters (Optuna)</span>
          </label>
        </Card>

        {error && (
          <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">
            {error}
          </div>
        )}

        <Button type="submit" size="lg" loading={loading} icon={<Play size={15} />} className="w-full justify-center">
          {loading ? 'Submitting…' : 'Launch Training Job'}
        </Button>
      </form>
    </div>
  )
}
