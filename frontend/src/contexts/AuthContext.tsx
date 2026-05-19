'use client'

import React, {
  createContext, useCallback, useContext, useEffect, useMemo, useState,
} from 'react'
import { useRouter } from 'next/navigation'
import type { User, AuthTokens } from '@/types'
import { loginUser, registerUser, refreshAccessToken } from '@/lib/api'

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

  // On mount: restore user from storage or try refresh
  useEffect(() => {
    const raw = localStorage.getItem(USER_KEY)
    const token = localStorage.getItem(ACCESS_KEY)
    const refreshToken = localStorage.getItem(REFRESH_KEY)

    if (raw && token) {
      try {
        setUser(JSON.parse(raw))
        setLoading(false)
        return
      } catch { /* fall through */ }
    }

    if (refreshToken) {
      refreshAccessToken(refreshToken)
        .then(tokens => { persist(tokens); setUser(tokens.user) })
        .catch(clear)
        .finally(() => setLoading(false))
    } else {
      setLoading(false)
    }
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
