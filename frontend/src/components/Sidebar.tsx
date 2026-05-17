'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  LayoutDashboard, Database, Cpu, BarChart3,
  Box, Zap, Settings, Github,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const NAV = [
  { href: '/',           label: 'Dashboard',  icon: LayoutDashboard },
  { href: '/dataset',    label: 'Dataset',    icon: Database         },
  { href: '/training',   label: 'Training',   icon: Cpu              },
  { href: '/results',    label: 'Results',    icon: BarChart3        },
  { href: '/models',     label: 'Models',     icon: Box              },
  { href: '/inference',  label: 'Inference',  icon: Zap              },
  { href: '/settings',   label: 'Settings',   icon: Settings         },
]

export default function Sidebar() {
  const path = usePathname()

  return (
    <aside className="flex flex-col w-56 min-h-screen bg-surface-800 border-r border-surface-600">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-surface-600">
        <div className="w-8 h-8 rounded-lg bg-brand-500 flex items-center justify-center text-white font-bold text-sm">V</div>
        <span className="text-white font-semibold tracking-wide">VisionForge</span>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-4 space-y-0.5 px-2">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === '/' ? path === '/' : path.startsWith(href)
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors',
                active
                  ? 'bg-brand-600 text-white'
                  : 'text-slate-400 hover:text-white hover:bg-surface-600',
              )}
            >
              <Icon size={16} />
              {label}
            </Link>
          )
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-surface-600">
        <a
          href="https://github.com/vfvision-ai/visionforge-ai"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-xs text-slate-500 hover:text-slate-300 transition-colors"
        >
          <Github size={14} />
          vfvision-ai/visionforge-ai
        </a>
      </div>
    </aside>
  )
}
