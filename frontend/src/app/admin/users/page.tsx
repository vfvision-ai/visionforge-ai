'use client'

import { useEffect, useState, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/AuthContext'
import { getAdminUsers, patchUser, deleteUser } from '@/lib/api'
import type { User } from '@/types'
import {
  ShieldCheck, ShieldOff, UserCheck, UserX, Trash2, RefreshCw, Search,
} from 'lucide-react'

type Toast = { id: number; type: 'success' | 'error'; message: string }

export default function AdminUsersPage() {
  const { user: me } = useAuth()
  const router = useRouter()

  const [users,   setUsers]   = useState<User[]>([])
  const [total,   setTotal]   = useState(0)
  const [loading, setLoading] = useState(true)
  const [search,  setSearch]  = useState('')
  const [toasts,  setToasts]  = useState<Toast[]>([])
  const toastId = { current: 0 }

  function toast(type: Toast['type'], message: string) {
    const id = ++toastId.current
    setToasts(t => [...t, { id, type, message }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 4000)
  }

  const load = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true)
    try {
      const data = await getAdminUsers(0, 200)
      setUsers(data.users)
      setTotal(data.total)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Failed to load users')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    // Guard: redirect non-admins
    if (me && me.role !== 'admin') { router.replace('/'); return }
    load()
  }, [me, router, load])

  async function handlePatch(userId: string, patch: { is_active?: boolean; role?: string }) {
    try {
      const updated = await patchUser(userId, patch)
      setUsers(u => u.map(x => x.id === userId ? updated : x))
      toast('success', 'User updated.')
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Update failed')
    }
  }

  async function handleDelete(target: User) {
    if (!confirm(`Delete ${target.email}? This cannot be undone.`)) return
    try {
      await deleteUser(target.id)
      setUsers(u => u.filter(x => x.id !== target.id))
      setTotal(t => t - 1)
      toast('success', `Deleted ${target.email}`)
    } catch (e) {
      toast('error', e instanceof Error ? e.message : 'Delete failed')
    }
  }

  const filtered = users.filter(u =>
    u.email.toLowerCase().includes(search.toLowerCase()) ||
    u.full_name.toLowerCase().includes(search.toLowerCase()),
  )

  return (
    <div className="p-6 min-h-screen" style={{ background: '#0d0f14' }}>
      {/* Toasts */}
      <div className="fixed top-4 right-4 z-50 space-y-2">
        {toasts.map(t => (
          <div key={t.id} className={`px-4 py-2.5 rounded-lg text-sm text-white shadow-lg ${
            t.type === 'success' ? 'bg-green-700' : 'bg-red-700'
          }`}>{t.message}</div>
        ))}
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold text-white">User Management</h1>
          <p className="text-sm text-slate-400 mt-0.5">{total} registered account{total !== 1 ? 's' : ''}</p>
        </div>
        <button
          onClick={() => load(true)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400
                     hover:text-white hover:bg-slate-700 transition-colors"
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>

      {/* Search */}
      <div className="relative mb-4">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
        <input
          type="text"
          placeholder="Search by name or email…"
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full max-w-sm pl-8 pr-3 py-2 rounded-lg border border-slate-700 text-sm
                     text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 transition-colors"
          style={{ background: '#141720' }}
        />
      </div>

      {/* Table */}
      <div className="rounded-xl border border-slate-700 overflow-hidden" style={{ background: '#141720' }}>
        {loading ? (
          <div className="flex items-center justify-center py-16">
            <div className="w-6 h-6 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin" />
          </div>
        ) : filtered.length === 0 ? (
          <p className="text-center text-slate-500 py-16 text-sm">No users found.</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-700 text-left">
                <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">User</th>
                <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">Role</th>
                <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">Status</th>
                <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">Joined</th>
                <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide">Last login</th>
                <th className="px-4 py-3 text-xs font-semibold text-slate-400 uppercase tracking-wide text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(u => {
                const isMe = u.id === me?.id
                return (
                  <tr key={u.id} className="border-b border-slate-800 hover:bg-slate-800/40 transition-colors">
                    <td className="px-4 py-3">
                      <p className="font-medium text-white">{u.full_name}</p>
                      <p className="text-xs text-slate-400">{u.email}</p>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                        u.role === 'admin'
                          ? 'bg-amber-900/50 text-amber-300 border border-amber-700'
                          : 'bg-slate-700 text-slate-300'
                      }`}>
                        {u.role === 'admin' && <ShieldCheck size={10} />}
                        {u.role}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-medium ${
                        u.is_active
                          ? 'bg-green-900/50 text-green-300 border border-green-800'
                          : 'bg-red-900/50 text-red-300 border border-red-800'
                      }`}>
                        {u.is_active ? 'Active' : 'Disabled'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-slate-400 text-xs">
                      {new Date(u.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-4 py-3 text-slate-400 text-xs">
                      {u.last_login_at ? new Date(u.last_login_at).toLocaleString() : '—'}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1 justify-end">
                        {/* Toggle role */}
                        {!isMe && (
                          <button
                            onClick={() => handlePatch(u.id, { role: u.role === 'admin' ? 'user' : 'admin' })}
                            title={u.role === 'admin' ? 'Demote to user' : 'Promote to admin'}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-amber-400 hover:bg-slate-700 transition-colors"
                          >
                            {u.role === 'admin' ? <ShieldOff size={14} /> : <ShieldCheck size={14} />}
                          </button>
                        )}

                        {/* Toggle active */}
                        {!isMe && (
                          <button
                            onClick={() => handlePatch(u.id, { is_active: !u.is_active })}
                            title={u.is_active ? 'Disable account' : 'Enable account'}
                            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
                          >
                            {u.is_active ? <UserX size={14} /> : <UserCheck size={14} />}
                          </button>
                        )}

                        {/* Delete */}
                        {!isMe && (
                          <button
                            onClick={() => handleDelete(u)}
                            title="Delete user"
                            className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-slate-700 transition-colors"
                          >
                            <Trash2 size={14} />
                          </button>
                        )}

                        {isMe && (
                          <span className="text-xs text-slate-500 pr-1">(you)</span>
                        )}
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-4 text-xs text-slate-500">
        <span className="flex items-center gap-1"><ShieldCheck size={12} className="text-amber-400" /> Promote / demote admin</span>
        <span className="flex items-center gap-1"><UserX size={12} /> Disable (blocks login)</span>
        <span className="flex items-center gap-1"><Trash2 size={12} /> Permanently delete</span>
      </div>
    </div>
  )
}
