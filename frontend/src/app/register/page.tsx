'use client'

import { useState, FormEvent } from 'react'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'

export default function RegisterPage() {
  const { register } = useAuth()
  const [email, setEmail]         = useState('')
  const [fullName, setFullName]   = useState('')
  const [password, setPassword]   = useState('')
  const [confirm, setConfirm]     = useState('')
  const [error, setError]         = useState<string | null>(null)
  const [loading, setLoading]     = useState(false)
  const [success, setSuccess]     = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)

    if (password !== confirm) {
      setError('Passwords do not match')
      return
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }

    setLoading(true)
    try {
      await register(email, fullName, password)
      setSuccess(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center" style={{ background: '#0d0f14' }}>
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-white">VisionForge</h1>
          <p className="text-sm text-slate-400 mt-1">Computer Vision Training Platform</p>
        </div>

        <div className="rounded-xl border border-slate-700 p-8" style={{ background: '#141720' }}>
          <h2 className="text-lg font-semibold text-white mb-6">Create an account</h2>

          {success ? (
            <div className="text-sm text-green-400 rounded-lg p-3 border border-green-800 text-center"
                 style={{ background: '#0f1a0f' }}>
              Account created!{' '}
              <Link href="/login" className="underline hover:text-green-300">Sign in</Link>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm text-slate-400 mb-1" htmlFor="fullName">
                  Full name
                </label>
                <input
                  id="fullName"
                  type="text"
                  required
                  autoComplete="name"
                  value={fullName}
                  onChange={e => setFullName(e.target.value)}
                  className="w-full rounded-lg border border-slate-600 px-3 py-2 text-sm text-white
                             focus:outline-none focus:border-indigo-500 transition-colors"
                  style={{ background: '#0d0f14' }}
                  placeholder="Jane Smith"
                />
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-1" htmlFor="email">
                  Email address
                </label>
                <input
                  id="email"
                  type="email"
                  required
                  autoComplete="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  className="w-full rounded-lg border border-slate-600 px-3 py-2 text-sm text-white
                             focus:outline-none focus:border-indigo-500 transition-colors"
                  style={{ background: '#0d0f14' }}
                  placeholder="you@example.com"
                />
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-1" htmlFor="password">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  required
                  autoComplete="new-password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="w-full rounded-lg border border-slate-600 px-3 py-2 text-sm text-white
                             focus:outline-none focus:border-indigo-500 transition-colors"
                  style={{ background: '#0d0f14' }}
                  placeholder="Min. 8 characters"
                />
              </div>

              <div>
                <label className="block text-sm text-slate-400 mb-1" htmlFor="confirm">
                  Confirm password
                </label>
                <input
                  id="confirm"
                  type="password"
                  required
                  autoComplete="new-password"
                  value={confirm}
                  onChange={e => setConfirm(e.target.value)}
                  className="w-full rounded-lg border border-slate-600 px-3 py-2 text-sm text-white
                             focus:outline-none focus:border-indigo-500 transition-colors"
                  style={{ background: '#0d0f14' }}
                  placeholder="••••••••"
                />
              </div>

              {error && (
                <p className="text-sm text-red-400 rounded-lg p-3 border border-red-800"
                   style={{ background: '#1a0f0f' }}>
                  {error}
                </p>
              )}

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-lg px-4 py-2.5 text-sm font-medium text-white
                           bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50
                           transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {loading ? 'Creating account…' : 'Create account'}
              </button>
            </form>
          )}

          <p className="mt-4 text-center text-sm text-slate-400">
            Already have an account?{' '}
            <Link href="/login" className="text-indigo-400 hover:text-indigo-300 transition-colors">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
