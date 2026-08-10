// app/(auth)/register/page.tsx
"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import toast from "react-hot-toast"
import { registerAction, getMeAction } from "@/lib/auth/actions"
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
  CardDescription,
} from "@/components/ui/card"
import { Loader2 } from "lucide-react"

export default function RegisterPage() {
  const router = useRouter()
  const { setUser } = useAuthStore()
  const [loading, setLoading] = useState(false)
  const [form, setForm] = useState({ full_name: "", email: "", password: "" })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const result = await registerAction(form.email, form.password, form.full_name)
      if (!result.success) {
        toast.error("خطا در ثبت‌نام. لطفاً دوباره تلاش کنید.")
        return
      }

      const me = await getMeAction()
      setUser(me)
      router.push("/onboarding/role")
    } catch {
      toast.error("خطا در ثبت‌نام. لطفاً دوباره تلاش کنید.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="shadow-lg border-border/50 dark:bg-card">
      <CardHeader>
        <CardTitle className="text-xl text-center">ایجاد حساب جدید</CardTitle>
        <CardDescription className="text-center">رایگان شروع کنید</CardDescription>
      </CardHeader>

      <form onSubmit={handleSubmit}>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="full_name">نام و نام خانوادگی</Label>
            <Input
              id="full_name"
              placeholder="علی محمدی"
              value={form.full_name}
              onChange={(e) => setForm({ ...form, full_name: e.target.value })}
              required
              disabled={loading}
            />
          </div>

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
              placeholder="حداقل ۸ کاراکتر"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
              required
              minLength={8}
              disabled={loading}
              dir="ltr"
            />
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-4">
          <Button type="submit" className="w-full" disabled={loading}>
            {loading && <Loader2 className="h-4 w-4 animate-spin ml-2" />}
            ثبت‌نام رایگان
          </Button>

          <p className="text-sm text-muted-foreground text-center">
            حساب دارید؟{" "}
            <Link href="/login" className="text-accent font-medium hover:underline">
              وارد شوید
            </Link>
          </p>
        </CardFooter>
      </form>
    </Card>
  )
}