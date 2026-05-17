export type JobStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
export type Framework = 'pytorch' | 'tensorflow' | 'sklearn'
export type TaskType  = 'classification' | 'detection' | 'segmentation'

export interface TrainingJob {
  id: string
  experiment_id: string | null
  task_type: TaskType
  framework: Framework
  dataset_name: string
  architecture: string
  status: JobStatus
  celery_task_id: string | null
  error_message: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
  output_dir: string | null
  model_path: string | null
  results: Record<string, number | string | boolean | null> | null
  training_history: EpochMetric[] | null
  hyperparams?: Record<string, unknown> | null
  duration_seconds?: number | null
}

export interface EpochMetric {
  epoch?: number
  train_loss?: number
  val_loss?: number
  train_accuracy?: number
  val_accuracy?: number
  train_miou?: number
  val_miou?: number
  val_dice?: number
  val_map50?: number
}

export interface Experiment {
  id: string
  name: string
  description: string
  tags: string[]
  created_at: string
  updated_at: string
}

export interface ModelVersion {
  id: string
  job_id: string
  name: string
  architecture: string
  framework: Framework
  task_type: TaskType
  num_classes: number | null
  model_path: string
  onnx_path: string | null
  val_accuracy: number | null
  val_loss: number | null
  is_production: boolean
  created_at: string
}

export interface HealthStatus {
  status: 'ok' | 'degraded'
  version: string
  database: string
  broker: string
  timestamp: string
}

export interface TrainingSubmitPayload {
  task_type: TaskType
  framework: Framework
  dataset_name: string
  architecture: string
  dataset_config: Record<string, unknown>
  epochs: number
  learning_rate: number
  batch_size: number
  optimize_hyperparams: boolean
  experiment_id?: string
}
