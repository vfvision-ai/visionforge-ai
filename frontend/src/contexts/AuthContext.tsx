'use client'

import React, {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from 'react'
import { useRouter } from 'next/navigation'
import type { User, AuthTokens } from '@/types'
import { loginUser, registerUser, refreshAccessToken, getMe } from '@/lib/api'

const ACCESS_KEY  = 'vf_access_token'
const REFRESH_KEY = 'vf_refresh_token'
const USER_KEY    = 'vf_user'

interface AuthContextValue {
  user: User | null
  isLoading: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, full_name: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

function persist(tokens: AuthTokens) {
  localStorage.setItem(ACCESS_KEY,  tokens.access_token)
  localStorage.setItem(REFRESH_KEY, tokens.refresh_token)
  localStorage.setItem(USER_KEY,    JSON.stringify(tokens.user))
}

function clear() {
  localStorage.removeItem(ACCESS_KEY)
  localStorage.removeItem(REFRESH_KEY)
  localStorage.removeItem(USER_KEY)
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser]       = useState<User | null>(null)
  const [isLoading, setLoading] = useState(true)
  const router = useRouter()

  // On mount: restore session then always re-fetch /me to get fresh role/status
  useEffect(() => {
    const raw = localStorage.getItem(USER_KEY)
    const token = localStorage.getItem(ACCESS_KEY)
    const refreshToken = localStorage.getItem(REFRESH_KEY)

    async function init() {
      // Restore cached user immediately so UI doesn't flash empty
      if (raw && token) {
        try { setUser(JSON.parse(raw)) } catch { /* ignore */ }
      }

      // Try to get a valid access token
      let validToken = token
      if (!validToken && refreshToken) {
        try {
          const tokens = await refreshAccessToken(refreshToken)
          persist(tokens)
          validToken = tokens.access_token
        } catch {
          clear()
          setLoading(false)
          return
        }
      }

      if (!validToken) {
        clear()
        setLoading(false)
        return
      }

      // Always fetch fresh user data so role/status is up to date
      try {
        const fresh = await getMe()
        setUser(fresh)
        localStorage.setItem(USER_KEY, JSON.stringify(fresh))
      } catch {
        // Token invalid — clear and force re-login
        clear()
        setUser(null)
      } finally {
        setLoading(false)
      }
    }

    init()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    const tokens = await loginUser(email, password)
    persist(tokens)
    setUser(tokens.user)
    router.push('/')
  }, [router])

  const register = useCallback(async (email: string, full_name: string, password: string) => {
    await registerUser(email, full_name, password)
    router.push('/login')
  }, [router])

  const logout = useCallback(() => {
    clear()
    setUser(null)
    router.push('/login')
  }, [router])

  const value = useMemo(
    () => ({ user, isLoading, login, register, logout }),
    [user, isLoading, login, register, logout],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>')
  return ctx
}
