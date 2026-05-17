import { cn } from '@/lib/utils'
import { Loader2 } from 'lucide-react'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
  icon?: React.ReactNode
}

const variants = {
  primary:   'bg-brand-600 hover:bg-brand-700 text-white',
  secondary: 'bg-surface-600 hover:bg-surface-500 text-slate-200 border border-surface-500',
  danger:    'bg-red-600 hover:bg-red-700 text-white',
  ghost:     'hover:bg-surface-600 text-slate-400 hover:text-white',
}
const sizes = {
  sm:  'px-3 py-1.5 text-xs rounded-lg',
  md:  'px-4 py-2 text-sm rounded-lg',
  lg:  'px-5 py-2.5 text-sm rounded-xl',
}

export default function Button({
  variant = 'primary', size = 'md', loading, icon, children, className, disabled, ...props
}: ButtonProps) {
  return (
    <button
      {...props}
      disabled={disabled || loading}
      className={cn(
        'inline-flex items-center gap-2 font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed',
        variants[variant], sizes[size], className,
      )}
    >
      {loading ? <Loader2 size={14} className="animate-spin" /> : icon}
      {children}
    </button>
  )
}
