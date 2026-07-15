"use client"

import { useQuery } from "@tanstack/react-query"
import { useAuthStore } from "@/store/auth.store"
import { authApi } from "@/lib/api/auth"
import Cookies from "js-cookie"

export function useAuth() {
  const { user, setUser, clear } = useAuthStore()

  const { isLoading } = useQuery({
    queryKey: ["me"],
    queryFn:  async () => {
      const me = await authApi.me()
      setUser(me)
      return me
    },
    enabled: !!Cookies.get("access_token"),
    retry:   false,
    staleTime: 5 * 60 * 1000,
  })

  const logout = () => {
    clear()
    authApi.logout()
  }

  return {
    user,
    isLoading,
    isAuthenticated: !!user,
    logout,
  }
}