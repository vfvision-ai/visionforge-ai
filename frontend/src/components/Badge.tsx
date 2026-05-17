import { cn } from '@/lib/utils'
import type { JobStatus } from '@/types'

interface BadgeProps {
  status?: JobStatus | 'default'
  text?: string
  children?: React.ReactNode
  variant?: JobStatus | 'default'
  className?: string
}

const variants: Record<string, string> = {
  pending:   'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20',
  running:   'bg-blue-500/10   text-blue-400   border border-blue-500/20 animate-pulse',
  completed: 'bg-green-500/10  text-green-400  border border-green-500/20',
  failed:    'bg-red-500/10    text-red-400    border border-red-500/20',
  cancelled: 'bg-slate-500/10  text-slate-400  border border-slate-500/20',
  default:   'bg-brand-500/10  text-brand-400  border border-brand-500/20',
}

export default function Badge({ status, text, children, variant, className }: BadgeProps) {
  const key = status ?? variant ?? 'default'
  const label = children ?? text ?? key
  return (
    <span className={cn('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium capitalize', variants[key] ?? variants.default, className)}>
      {label}
    </span>
  )
}
