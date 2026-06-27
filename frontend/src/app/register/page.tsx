'use client'

import { useState, FormEvent, useCallback } from 'react'
import Link from 'next/link'
import { useAuth } from '@/contexts/AuthContext'

// RFC 5322-inspired email regex
const EMAIL_RE = /^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$/
const SPECIAL_RE = /[!@#$%^&*()\-_=+[\]{}|;':",./<>?]/

function validateEmail(value: string): string | null {
  if (!value) return 'Email address is required.'
  if (!EMAIL_RE.test(value)) return 'Please enter a valid email address (e.g. you@example.com).'
  const tld = value.split('@')[1]?.split('.').pop() ?? ''
  if (tld.length < 2) return 'Email domain must have a valid TLD (e.g. .com, .org).'
  return null
}

interface PasswordScore {
  score: 0 | 1 | 2 | 3 | 4
  label: string
  color: string
  tips: string[]
}

function scorePassword(pw: string): PasswordScore {
  if (!pw) return { score: 0, label: '', color: '#334155', tips: [] }
  const tips: string[] = []
  let score = 0
  if (pw.length >= 8)   score++; else tips.push('at least 8 characters')
  if (/[A-Z]/.test(pw)) score++; else tips.push('an uppercase letter')
  if (/[0-9]/.test(pw)) score++; else tips.push('a digit (0-9)')
  if (SPECIAL_RE.test(pw)) score++; else tips.push('a special character (!@#$…)')

  const map: Record<number, { label: string; color: string }> = {
    0: { label: 'Too weak',  color: '#ef4444' },
    1: { label: 'Weak',      color: '#f97316' },
    2: { label: 'Fair',      color: '#eab308' },
    3: { label: 'Strong',    color: '#22c55e' },
    4: { label: 'Very strong', color: '#10b981' },
  }
  return { score: score as PasswordScore['score'], ...map[score], tips }
}

export default function RegisterPage() {
  const { register } = useAuth()
  const [email, setEmail]         = useState('')
  const [fullName, setFullName]   = useState('')
  const [password, setPassword]   = useState('')
  const [confirm, setConfirm]     = useState('')
  const [emailError, setEmailError] = useState<string | null>(null)
  const [formError, setFormError]   = useState<string | null>(null)
  const [loading, setLoading]     = useState(false)
  const [success, setSuccess]     = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm]   = useState(false)

  const pwScore = scorePassword(password)
  const emailValid = email.length > 0 && !emailError && EMAIL_RE.test(email)
  const confirmMatch = confirm.length > 0 && confirm === password

  const handleEmailBlur = useCallback(() => {
    setEmailError(validateEmail(email))
  }, [email])

  const handleEmailChange = useCallback((v: string) => {
    setEmail(v)
    if (emailError && EMAIL_RE.test(v)) setEmailError(null)
  }, [emailError])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setFormError(null)

    const emailErr = validateEmail(email)
    if (emailErr) { setEmailError(emailErr); return }

    if (fullName.trim().length < 1) {
      setFormError('Full name is required.')
      return
    }
    if (pwScore.score < 2) {
      setFormError('Please choose a stronger password.')
      return
    }
    if (password !== confirm) {
      setFormError('Passwords do not match.')
      return
    }

    setLoading(true)
    try {
      await register(email, fullName.trim(), password)
      setSuccess(true)
    } catch (err) {
      setFormError(
        err instanceof Error
          ? err.message.replace(/^\d+:\s*/, '')
          : 'Registration failed'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center py-8" style={{ background: '#0d0f14' }}>
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl mb-3"
               style={{ background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' }}>
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white">VisionForge</h1>
          <p className="text-sm text-slate-400 mt-1">Computer Vision Training Platform</p>
        </div>

        <div className="rounded-xl border border-slate-700 p-8" style={{ background: '#141720' }}>
          <h2 className="text-lg font-semibold text-white mb-6">Create an account</h2>

          {success ? (
            <div className="text-sm text-emerald-400 rounded-lg p-4 border border-emerald-800 text-center"
                 style={{ background: '#0f1a14' }}>
              <svg className="w-8 h-8 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="font-medium mb-1">Account created!</p>
              <p className="text-slate-400 text-xs mb-3">You can now sign in with your credentials.</p>
              <Link href="/login"
                    className="inline-block px-4 py-1.5 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-white text-sm transition-colors">
                Sign in →
              </Link>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4" noValidate>
              {/* Full name */}
              <div>
                <label className="block text-sm text-slate-400 mb-1" htmlFor="fullName">Full name</label>
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

              {/* Email */}
              <div>
                <label className="block text-sm text-slate-400 mb-1" htmlFor="email">Email address</label>
                <div className="relative">
                  <input
                    id="email"
                    type="email"
                    required
                    autoComplete="email"
                    value={email}
                    onChange={e => handleEmailChange(e.target.value)}
                    onBlur={handleEmailBlur}
                    className={`w-full rounded-lg border px-3 py-2 text-sm text-white pr-9
                               focus:outline-none transition-colors
                               ${emailError
                                 ? 'border-red-500 focus:border-red-400'
                                 : emailValid
                                 ? 'border-emerald-500 focus:border-emerald-400'
                                 : 'border-slate-600 focus:border-indigo-500'}`}
                    style={{ background: '#0d0f14' }}
                    placeholder="you@example.com"
                  />
                  {email.length > 0 && (
                    <span className="absolute right-2.5 top-1/2 -translate-y-1/2 pointer-events-none">
                      {emailValid
                        ? <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                        : <svg className="w-4 h-4 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                      }
                    </span>
                  )}
                </div>
                {emailError && (
                  <p className="mt-1 text-xs text-red-400 flex items-center gap-1">
                    <svg className="w-3 h-3 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    {emailError}
                  </p>
                )}
              </div>

              {/* Password */}
              <div>
                <label className="block text-sm text-slate-400 mb-1" htmlFor="password">Password</label>
                <div className="relative">
                  <input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    required
                    autoComplete="new-password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    className="w-full rounded-lg border border-slate-600 px-3 py-2 text-sm text-white pr-9
                               focus:outline-none focus:border-indigo-500 transition-colors"
                    style={{ background: '#0d0f14' }}
                    placeholder="Min. 8 characters"
                  />
                  <button type="button" onClick={() => setShowPassword(s => !s)}
                          className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
                          tabIndex={-1} aria-label={showPassword ? 'Hide password' : 'Show password'}>
                    {showPassword
                      ? <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" /></svg>
                      : <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
                    }
                  </button>
                </div>

                {/* Password strength meter */}
                {password.length > 0 && (
                  <div className="mt-2 space-y-1.5">
                    <div className="flex gap-1">
                      {[1, 2, 3, 4].map(n => (
                        <div key={n} className="flex-1 h-1 rounded-full transition-all duration-300"
                             style={{ background: n <= pwScore.score ? pwScore.color : '#334155' }} />
                      ))}
                    </div>
                    <p className="text-xs font-medium" style={{ color: pwScore.color }}>
                      {pwScore.label}
                      {pwScore.tips.length > 0 && (
                        <span className="text-slate-500 font-normal"> — add {pwScore.tips[0]}</span>
                      )}
                    </p>
                  </div>
                )}
              </div>

              {/* Confirm password */}
              <div>
                <label className="block text-sm text-slate-400 mb-1" htmlFor="confirm">Confirm password</label>
                <div className="relative">
                  <input
                    id="confirm"
                    type={showConfirm ? 'text' : 'password'}
                    required
                    autoComplete="new-password"
                    value={confirm}
                    onChange={e => setConfirm(e.target.value)}
                    className={`w-full rounded-lg border px-3 py-2 text-sm text-white pr-9
                               focus:outline-none transition-colors
                               ${confirm.length > 0
                                 ? confirmMatch
                                   ? 'border-emerald-500 focus:border-emerald-400'
                                   : 'border-red-500 focus:border-red-400'
                                 : 'border-slate-600 focus:border-indigo-500'}`}
                    style={{ background: '#0d0f14' }}
                    placeholder="••••••••"
                  />
                  <button type="button" onClick={() => setShowConfirm(s => !s)}
                          className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-200 transition-colors"
                          tabIndex={-1} aria-label={showConfirm ? 'Hide password' : 'Show password'}>
                    {showConfirm
                      ? <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" /></svg>
                      : <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>
                    }
                  </button>
                </div>
                {confirm.length > 0 && (
                  <p className={`mt-1 text-xs flex items-center gap-1 ${confirmMatch ? 'text-emerald-400' : 'text-red-400'}`}>
                    {confirmMatch
                      ? <><svg className="w-3 h-3 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>Passwords match</>
                      : <><svg className="w-3 h-3 shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>Passwords do not match</>
                    }
                  </p>
                )}
              </div>

              {/* Form-level error */}
              {formError && (
                <div className="text-sm text-red-400 rounded-lg p-3 border border-red-800 flex items-start gap-2"
                     style={{ background: '#1a0f0f' }}>
                  <svg className="w-4 h-4 mt-0.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <span>{formError}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={loading || !!emailError || (confirm.length > 0 && !confirmMatch)}
                className="w-full rounded-lg px-4 py-2.5 text-sm font-medium text-white
                           bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed
                           transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {loading
                  ? <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Creating account…
                    </span>
                  : 'Create account'
                }
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
