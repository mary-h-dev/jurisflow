"use client"

import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { ThemeProvider } from "next-themes"
import { Toaster } from "react-hot-toast"
import { useState } from "react"

export function Providers({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: { retry: 1, staleTime: 30_000 },
        },
      })
  )

  return (
    <QueryClientProvider client={queryClient}>
      {/*
        attribute="class" → تم رو با class روی html ست می‌کنه
        defaultTheme="system" → از تنظیمات سیستم کاربر پیروی می‌کنه
        enableSystem → اجازه تشخیص تم سیستم رو میده
      */}
      <ThemeProvider
        attribute="class"
        defaultTheme="system"
        enableSystem
        disableTransitionOnChange
      >
        {children}

        <Toaster
          position="top-center"
          toastOptions={{
            duration: 4000,
            style: {
              fontFamily: "Vazirmatn, sans-serif",
              direction:  "rtl",
              borderRadius: "8px",
            },
            success: {
              iconTheme: { primary: "#22c55e", secondary: "#fff" },
            },
            error: {
              iconTheme: { primary: "#ef4444", secondary: "#fff" },
            },
          }}
        />
      </ThemeProvider>
    </QueryClientProvider>
  )
}