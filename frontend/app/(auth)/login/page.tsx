"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import toast from "react-hot-toast"
import { authApi } from "@/lib/api/auth"
import { useAuthStore } from "@/store/auth.store"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Loader2 } from "lucide-react"

export default function LoginPage() {
  const router = useRouter()
  const { setUser } = useAuthStore()
  const [loading, setLoading] = useState(false)
  const [form, setForm] = useState({ email: "", password: "" })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      await authApi.login(form.email, form.password)
      const me = await authApi.me()
      setUser(me)

      const status = await authApi.onboardingStatus()
      if (!status.is_onboarding_complete) {
        router.push("/onboarding/role")
      } else {
        router.push("/search")
      }
    } catch {
      toast.error("ایمیل یا رمز عبور اشتباه است")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="shadow-lg border-border/50 dark:bg-card">
      <CardHeader>
        <CardTitle className="text-xl text-center">ورود به حساب</CardTitle>
      </CardHeader>

      <form onSubmit={handleSubmit}>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">ایمیل</Label>
            <Input
              id="email"
              type="email"
              placeholder="example@email.com"
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
              required
              disabled={loading}
              dir="ltr"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="password">رمز عبور</Label>
            <Input
              id="password"
              type="password"
              placeholder="••••••••"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required
              disabled={loading}
              dir="ltr"
            />
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-4">
          <Button type="submit" className="w-full" disabled={loading}>
            {loading && <Loader2 className="h-4 w-4 animate-spin ml-2" />}
            ورود
          </Button>

          <p className="text-sm text-muted-foreground text-center">
            حساب ندارید؟{" "}
            <Link
              href="/register"
              className="text-accent font-medium hover:underline"
            >
              ثبت‌نام کنید
            </Link>
          </p>
        </CardFooter>
      </form>
    </Card>
  )
}