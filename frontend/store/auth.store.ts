import { create } from "zustand"
import { persist } from "zustand/middleware"
import type { User } from "@/lib/types/auth"

interface AuthState {
  user:      User | null
  isLoading: boolean
  setUser:   (user: User | null) => void
  setLoading:(loading: boolean) => void
  clear:     () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user:      null,
      isLoading: false,
      setUser:   (user)    => set({ user }),
      setLoading:(loading) => set({ isLoading: loading }),
      clear:     ()        => set({ user: null }),
    }),
    {
      name:    "jurisflow-auth",
      partialize: (state) => ({ user: state.user }),
    }
  )
)