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
interface ArchEntry { value: string; label: string; params: string; speed: number; desc: string; recommended?: boolean }
const ARCHITECTURES: Record<string, Record<string, ArchEntry[]>> = {
  pytorch: {
    classification: [
      { value: 'Simple CNN',       label: 'Simple CNN',         params: '50K',   speed: 5, desc: 'Lightweight baseline — great for grayscale 28×28 images' },
      { value: 'efficientnet_b0',  label: 'EfficientNet-B0',    params: '5.3M',  speed: 4, desc: 'Best accuracy/speed tradeoff, ImageNet pretrained', recommended: true },
      { value: 'efficientnet_b3',  label: 'EfficientNet-B3',    params: '12M',   speed: 3, desc: 'Higher accuracy, 300×300 resolution' },
      { value: 'resnet50',         label: 'ResNet-50',          params: '25.6M', speed: 3, desc: 'Classic deep residual network, robust and pretrained' },
      { value: 'convnext_tiny',    label: 'ConvNeXt-Tiny',      params: '28.6M', speed: 3, desc: 'Modern CNN design, outperforms ResNet on most tasks' },
      { value: 'vit_base',         label: 'Vision Transformer',  params: '86.4M', speed: 2, desc: 'Transformer-based, highest accuracy on large datasets' },
    ],
    detection: [
      { value: 'yolov8n',          label: 'YOLOv8-Nano',        params: '3.2M',  speed: 5, desc: 'Fastest YOLO, suited for real-time applications', recommended: true },
      { value: 'yolov8s',          label: 'YOLOv8-Small',       params: '11.2M', speed: 4, desc: 'Balanced speed and accuracy' },
      { value: 'yolov8m',          label: 'YOLOv8-Medium',      params: '25.9M', speed: 3, desc: 'Higher accuracy YOLO variant' },
      { value: 'faster_rcnn_r50',  label: 'Faster R-CNN',       params: '41.8M', speed: 2, desc: 'Two-stage detector, high precision' },
      { value: 'detr_r50',         label: 'DETR',               params: '41.3M', speed: 2, desc: 'Transformer end-to-end detection, no NMS' },
    ],
    segmentation: [
      { value: 'segformer_b0',        label: 'SegFormer-B0',      params: '3.7M',  speed: 4, desc: 'Lightweight transformer segmentation', recommended: true },
      { value: 'unet_resnet34',       label: 'U-Net ResNet-34',   params: '24.4M', speed: 3, desc: 'Classic encoder-decoder with deep skip connections' },
      { value: 'segformer_b2',        label: 'SegFormer-B2',      params: '27.3M', speed: 3, desc: 'Stronger transformer, 88% mIoU on ADE20K' },
      { value: 'deeplabv3_resnet50',  label: 'DeepLabV3+',        params: '39.6M', speed: 2, desc: 'Atrous convolutions for dense prediction, 87% mIoU' },
    ],
  },
  tensorflow: {
    classification: [
      { value: 'Sequential CNN', label: 'Sequential CNN',  params: '~200K', speed: 5, desc: 'Simple 3-conv-block Keras model, no pretrained weights' },
      { value: 'EfficientNetB0', label: 'EfficientNet-B0', params: '5.3M',  speed: 4, desc: 'Best Keras accuracy/speed, pretrained on ImageNet', recommended: true },
      { value: 'MobileNetV2',    label: 'MobileNetV2',     params: '3.4M',  speed: 5, desc: 'Mobile-optimised, excellent for edge deployment' },
      { value: 'ResNet50',       label: 'ResNet-50',       params: '25.6M', speed: 3, desc: 'Deep residual network, battle-tested backbone' },
      { value: 'VGG16',          label: 'VGG-16',          params: '138M',  speed: 1, desc: 'Deep conv architecture — high memory usage' },
      { value: 'InceptionV3',    label: 'Inception V3',    params: '23.9M', speed: 2, desc: 'Multi-scale inception modules' },
    ],
    detection: [
      { value: 'SSD-MobileNet',   label: 'SSD MobileNet',  params: '~10M',  speed: 5, desc: 'Single-shot detector, mobile-friendly', recommended: true },
      { value: 'YOLO-TensorFlow', label: 'YOLO (TF)',      params: '~25M',  speed: 4, desc: 'YOLO port for TensorFlow ecosystem' },
      { value: 'CenterNet',       label: 'CenterNet',      params: '~20M',  speed: 3, desc: 'Anchor-free detection via heatmap' },
      { value: 'Faster R-CNN',    label: 'Faster R-CNN',   params: '~41M',  speed: 2, desc: 'Two-stage detector for high accuracy' },
    ],
    segmentation: [
      { value: 'U-Net',      label: 'U-Net',      params: '~31M', speed: 3, desc: 'Encoder-decoder with skip connections — widely used in medical imaging', recommended: true },
      { value: 'FCN',        label: 'FCN',        params: '~15M', speed: 4, desc: 'Fully Convolutional Network, fast baseline' },
      { value: 'DeepLabV3+', label: 'DeepLabV3+', params: '~40M', speed: 2, desc: 'Atrous spatial pyramid pooling' },
      { value: 'Mask R-CNN', label: 'Mask R-CNN', params: '~45M', speed: 1, desc: 'Instance segmentation — per-object masks' },
    ],
  },
  sklearn: {
    classification: [
      { value: 'Random Forest',          label: 'Random Forest',          params: 'n/a', speed: 4, desc: 'Ensemble of decision trees, handles noise and overfitting well', recommended: true },
      { value: 'Extra Trees',            label: 'Extra Trees',            params: 'n/a', speed: 5, desc: 'Faster random forest with extra randomisation' },
      { value: 'Gradient Boosting',      label: 'Gradient Boosting',      params: 'n/a', speed: 3, desc: 'Sequential boosting for high accuracy on flat features' },
      { value: 'Support Vector Machine', label: 'SVM (RBF kernel)',       params: 'n/a', speed: 2, desc: 'Effective for medium-size high-dimensional feature sets' },
      { value: 'Linear SVM',             label: 'Linear SVM',             params: 'n/a', speed: 4, desc: 'Fast SVM for very high-dimensional features' },
      { value: 'Logistic Regression',    label: 'Logistic Regression',    params: 'n/a', speed: 5, desc: 'Simple linear classifier, interpretable baseline' },
      { value: 'K-Nearest Neighbors',    label: 'K-Nearest Neighbors',    params: 'n/a', speed: 2, desc: 'Distance-based, no training phase, lazy learner' },
      { value: 'Naive Bayes',            label: 'Gaussian Naive Bayes',   params: 'n/a', speed: 5, desc: 'Probabilistic, very fast inference' },
      { value: 'Decision Tree',          label: 'Decision Tree',          params: 'n/a', speed: 5, desc: 'Interpretable tree-based model' },
      { value: 'Multi-layer Perceptron', label: 'MLP Neural Network',     params: 'n/a', speed: 3, desc: 'Shallow fully-connected net via sklearn' },
      { value: 'AdaBoost',               label: 'AdaBoost',               params: 'n/a', speed: 3, desc: 'Adaptive boosting ensemble' },
      { value: 'Ridge Classifier',       label: 'Ridge Classifier',       params: 'n/a', speed: 5, desc: 'L2-regularised linear model, fast and robust' },
    ],
    detection: [
      { value: 'HOG + SVM',                              label: 'HOG + SVM',                 params: 'n/a', speed: 3, desc: 'Histogram of Oriented Gradients features + SVM sliding window', recommended: true },
      { value: 'Feature Extraction + Random Forest',     label: 'Features + Random Forest',  params: 'n/a', speed: 4, desc: 'Handcrafted image features + ensemble classifier' },
      { value: 'Local Binary Pattern + SVM',             label: 'LBP + SVM',                 params: 'n/a', speed: 3, desc: 'Texture descriptor for robust feature extraction' },
      { value: 'SIFT + KMeans + SVM',                    label: 'SIFT Bag-of-Words',         params: 'n/a', speed: 2, desc: 'SIFT keypoints → BoW histogram → SVM pipeline' },
      { value: 'Histogram Features + Gradient Boosting', label: 'Histogram + Gradient Boost', params: 'n/a', speed: 3, desc: 'Colour histograms + gradient boosted trees' },
    ],
    segmentation: [
      { value: 'K-Means Clustering',  label: 'K-Means Clustering',    params: 'n/a', speed: 4, desc: 'Unsupervised pixel clustering by colour similarity', recommended: true },
      { value: 'Watershed + Features', label: 'Watershed',            params: 'n/a', speed: 3, desc: 'Morphological watershed + feature extraction' },
      { value: 'Gaussian Mixture',     label: 'Gaussian Mixture Model',params: 'n/a', speed: 3, desc: 'Probabilistic pixel grouping via EM algorithm' },
      { value: 'DBSCAN + Features',    label: 'DBSCAN',               params: 'n/a', speed: 2, desc: 'Density-based spatial clustering of features' },
      { value: 'Mean Shift',           label: 'Mean Shift',           params: 'n/a', speed: 1, desc: 'Mode-seeking non-parametric clustering' },
      { value: 'Spectral Clustering',  label: 'Spectral Clustering',  params: 'n/a', speed: 1, desc: 'Graph-based spectral method for complex shapes' },
    ],
  },
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
    dataset_name: 'MNIST', architecture: 'Simple CNN',
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

  const archsByFramework = ARCHITECTURES[form.framework] ?? ARCHITECTURES.pytorch
  const archs = archsByFramework[form.task_type] ?? archsByFramework.classification
  const selectedArch = archs.find(a => a.value === form.architecture)

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
              onChange={e => {
                const newTask = e.target.value
                set('task_type', newTask)
                const newArchs = (ARCHITECTURES[form.framework] ?? ARCHITECTURES.pytorch)[newTask] ?? ARCHITECTURES.pytorch.classification
                set('architecture', newArchs[0]?.value ?? '')
              }} options={TASK_TYPES} />
            <Select label="Framework" value={form.framework}
              onChange={e => {
                const newFw = e.target.value
                set('framework', newFw)
                const newArchs = (ARCHITECTURES[newFw] ?? ARCHITECTURES.pytorch)[form.task_type] ?? ARCHITECTURES.pytorch.classification
                set('architecture', newArchs[0]?.value ?? '')
              }} options={FRAMEWORKS} />
          </div>
        </Card>

        {/* Model */}
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Model Architecture</h2>
          <Select label="Architecture" value={form.architecture}
            onChange={e => set('architecture', e.target.value)}
            options={archs.map(a => ({
              value: a.value,
              label: `${a.label}${a.recommended ? ' ⭐' : ''} — ${a.params}`,
            }))} />
          {selectedArch && (
            <div className="mt-3 p-3 rounded-lg bg-surface-800 border border-surface-700 flex items-start gap-4 text-xs">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-medium text-white">{selectedArch.label}</span>
                  {selectedArch.recommended && (
                    <span className="px-1.5 py-0.5 rounded text-xs bg-brand-500/20 text-brand-300">recommended</span>
                  )}
                </div>
                <p className="text-slate-400">{selectedArch.desc}</p>
              </div>
              <div className="shrink-0 text-right space-y-1">
                <div>
                  <span className="text-slate-500">Params </span>
                  <span className="font-mono text-white">{selectedArch.params}</span>
                </div>
                <div>
                  <span className="text-slate-500">Speed </span>
                  <span className="text-yellow-400" title={`${selectedArch.speed}/5`}>{'⚡'.repeat(selectedArch.speed)}{'·'.repeat(5 - selectedArch.speed)}</span>
                </div>
              </div>
            </div>
          )}
        </Card>

        {/* Hyperparameters */}
        <Card>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">Hyperparameters</h2>
          <div className="grid grid-cols-3 gap-4 items-end">
            <div>
              <label className="block text-xs text-slate-400 mb-1.5">Epochs</label>
              <input type="range" title="Epochs" min={1} max={200} step={1} value={form.epochs}
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
                <input type="range" title="Number of HPO trials" min={5} max={50} step={5} value={form.n_trials}
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
                    <input type="range" title="Early stopping patience" min={3} max={30} step={1} value={form.patience}
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
