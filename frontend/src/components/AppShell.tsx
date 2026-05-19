'use client'

import { usePathname } from 'next/navigation'
import Sidebar from '@/components/Sidebar'
import ProtectedRoute from '@/components/ProtectedRoute'

const PUBLIC_PATHS = ['/login', '/register']

export default function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname()
  const isPublic = PUBLIC_PATHS.some(p => pathname.startsWith(p))

  if (isPublic) {
    // Public pages — no sidebar, no auth guard, just render content
    return <>{children}</>
  }

  return (
    <ProtectedRoute>
      <Sidebar />
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </ProtectedRoute>
  )
}
