// app/(onboarding)/role/page.tsx
"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import toast from "react-hot-toast"
import { setRoleAction } from "@/lib/auth/actions"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Scale, GraduationCap, User } from "lucide-react"
import { cn } from "@/lib/utils"

const ROLES = [
  { value: "lawyer",  label: "وکیل",       icon: Scale },
  { value: "student",  label: "دانشجوی حقوق", icon: GraduationCap },
  { value: "citizen",  label: "شهروند",     icon: User },
] as const

export default function RoleOnboardingPage() {
  const router = useRouter()
  const [selected, setSelected] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const handleContinue = async () => {
    if (!selected) return
    setLoading(true)
    try {
      const result = await setRoleAction(selected as "lawyer" | "student" | "citizen")
      if (!result.success) {
        toast.error("خطایی رخ داد. دوباره تلاش کنید.")
        return
      }
      router.push("/search")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Card className="shadow-lg border-border/50 dark:bg-card">
      <CardContent className="pt-6 space-y-6">
        <h2 className="text-lg font-semibold text-center">
          نقش خودت رو انتخاب کن
        </h2>

        <div className="grid grid-cols-1 gap-3">
          {ROLES.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              type="button"
              onClick={() => setSelected(value)}
              className={cn(
                "flex items-center gap-3 rounded-lg border p-4 text-right transition-colors",
                selected === value
                  ? "border-accent bg-accent/10"
                  : "border-border hover:bg-secondary/50"
              )}
            >
              <Icon className="h-5 w-5 text-accent" />
              <span className="font-medium">{label}</span>
            </button>
          ))}
        </div>

        <Button
          className="w-full"
          disabled={!selected || loading}
          onClick={handleContinue}
        >
          ادامه
        </Button>
      </CardContent>
    </Card>
  )
}