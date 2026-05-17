'use client'
import { Settings } from 'lucide-react'
import Card from '@/components/Card'

const CONFIG_SECTIONS = [
  {
    title: 'API / Backend',
    items: [
      { key: 'API_URL',           default: 'http://api:8000',         desc: 'Internal URL of the FastAPI backend (used by Next.js server-side rewrites)' },
      { key: 'NEXT_PUBLIC_WS_URL',default: 'ws://localhost:8000',     desc: 'WebSocket URL for real-time training updates (if implemented)' },
    ],
  },
  {
    title: 'Database',
    items: [
      { key: 'DATABASE_URL',      default: 'sqlite:///./ml_platform.db', desc: 'SQLAlchemy database connection string' },
    ],
  },
  {
    title: 'Task Queue',
    items: [
      { key: 'REDIS_URL',         default: 'redis://redis:6379/0',    desc: 'Redis connection URL for Celery task queue' },
      { key: 'CELERY_BROKER_URL', default: 'redis://redis:6379/0',    desc: 'Celery broker URL' },
      { key: 'CELERY_RESULT_BACKEND', default: 'redis://redis:6379/0', desc: 'Celery result backend URL' },
    ],
  },
  {
    title: 'Storage & Paths',
    items: [
      { key: 'MODEL_SAVE_DIR',    default: './models',                desc: 'Directory where trained models are saved' },
      { key: 'EXPERIMENT_DIR',    default: './experiments',           desc: 'Directory for experiment results and logs' },
      { key: 'UPLOAD_DIR',        default: './uploads',               desc: 'Directory for uploaded dataset files' },
    ],
  },
  {
    title: 'GPU & Hardware',
    items: [
      { key: 'NVIDIA_VISIBLE_DEVICES', default: 'all',               desc: 'GPU devices visible to containers (Docker compose gpu profile)' },
      { key: 'CUDA_VISIBLE_DEVICES',   default: '0',                 desc: 'CUDA device index for training workers' },
    ],
  },
]

export default function SettingsPage() {
  return (
    <div className="p-8 space-y-6 max-w-3xl">
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Settings size={22} className="text-brand-400" />
          Settings
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Configuration is managed via environment variables. Set these in your <code className="text-brand-400 font-mono text-xs">.env</code> file or Docker compose environment block.
        </p>
      </div>

      {CONFIG_SECTIONS.map(section => (
        <Card key={section.title}>
          <h2 className="text-sm font-semibold text-slate-300 mb-4">{section.title}</h2>
          <div className="divide-y divide-surface-700">
            {section.items.map(item => (
              <div key={item.key} className="py-3 grid grid-cols-5 gap-4">
                <div className="col-span-2">
                  <code className="text-xs text-brand-400 font-mono">{item.key}</code>
                  <p className="text-xs text-slate-600 mt-0.5">default: <span className="text-slate-500 font-mono">{item.default}</span></p>
                </div>
                <p className="col-span-3 text-sm text-slate-400">{item.desc}</p>
              </div>
            ))}
          </div>
        </Card>
      ))}

      <Card>
        <h2 className="text-sm font-semibold text-slate-300 mb-2">Running Services</h2>
        <p className="text-sm text-slate-500">
          The platform runs as Docker containers defined in <code className="text-brand-400 font-mono text-xs">docker-compose.yml</code>.
          Use the <code className="text-brand-400 font-mono text-xs">production</code> profile for all services:
        </p>
        <pre className="mt-3 text-xs text-slate-400 font-mono bg-surface-900 rounded-lg p-4 overflow-x-auto">
{`docker compose --profile production up -d --build
docker compose logs -f api worker
docker compose ps`}
        </pre>
      </Card>
    </div>
  )
}
