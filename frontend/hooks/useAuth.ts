// hooks/useAuth.ts
"use client"

import { useQuery } from "@tanstack/react-query"
import { useAuthStore } from "@/store/auth.store"
import { getMeAction, logoutAction } from "@/lib/auth/actions"

export function useAuth() {
  const { user, setUser, clear } = useAuthStore()

  const { isLoading } = useQuery({
    queryKey: ["me"],
    queryFn: async () => {
      const me = await getMeAction()
      setUser(me)
      return me
    },
    retry: false,
    staleTime: 5 * 60 * 1000,
  })

  const logout = async () => {
    await logoutAction()
    clear()
    window.location.href = "/login"
  }

  return {
    user,
    isLoading,
    isAuthenticated: !!user,
    logout,
  }
}