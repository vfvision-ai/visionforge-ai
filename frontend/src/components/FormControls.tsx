import { cn } from '@/lib/utils'

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
}

export function Input({ label, error, className, ...props }: InputProps) {
  return (
    <div className="space-y-1">
      {label && <label className="block text-xs text-slate-400 font-medium">{label}</label>}
      <input
        {...props}
        className={cn(
          'w-full px-3 py-2 text-sm rounded-lg bg-surface-700 border border-surface-500',
          'text-white placeholder-slate-500',
          'focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent',
          'disabled:opacity-50',
          error && 'border-red-500',
          className,
        )}
      />
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  )
}

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  options: { value: string; label: string }[]
}

export function Select({ label, options, className, ...props }: SelectProps) {
  return (
    <div className="space-y-1">
      {label && <label className="block text-xs text-slate-400 font-medium">{label}</label>}
      <select
        {...props}
        className={cn(
          'w-full px-3 py-2 text-sm rounded-lg bg-surface-700 border border-surface-500',
          'text-white focus:outline-none focus:ring-2 focus:ring-brand-500 focus:border-transparent',
          'disabled:opacity-50',
          className,
        )}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </div>
  )
}
