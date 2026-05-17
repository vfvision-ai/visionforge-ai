import { cn } from '@/lib/utils'

interface CardProps {
  children: React.ReactNode
  className?: string
  onClick?: () => void
}

export default function Card({ children, className, onClick }: CardProps) {
  return (
    <div
      onClick={onClick}
      className={cn(
        'rounded-xl border border-surface-600 bg-surface-800 p-5',
        onClick && 'cursor-pointer hover:border-brand-500/50 transition-colors',
        className,
      )}
    >
      {children}
    </div>
  )
}

interface StatCardProps {
  label: string
  value: string | number
  sub?: string
  icon: React.ReactNode
  color?: string
}

export function StatCard({ label, value, sub, icon, color = 'text-brand-400' }: StatCardProps) {
  return (
    <Card>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-1">{label}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
          {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
        </div>
        <div className={cn('p-2 rounded-lg bg-surface-700', color)}>{icon}</div>
      </div>
    </Card>
  )
}
