"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { useAuth } from "@/hooks/useAuth"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Search,
  Brain,
  FileText,
  LogOut,
  Scale,
  Sun,
  Moon,
  Lock,
} from "lucide-react"
import { useTheme } from "next-themes"

// ── آیتم‌های ناوبری ───────────────────────────────────────────────────────────
const NAV_ITEMS = [
  {
    href:  "/search",
    label: "جستجوی قانون",
    icon:  Search,
    plans: ["free", "pro", "enterprise"],
    badge: null,
  },
  {
    href:  "/analyze",
    label: "تحلیل پرونده",
    icon:  Brain,
    plans: ["pro", "enterprise"],
    badge: "Pro",
  },
  {
    href:  "/documents",
    label: "اسناد من",
    icon:  FileText,
    plans: ["pro", "enterprise"],
    badge: "Pro",
  },
]

// ── نمایش پلن ─────────────────────────────────────────────────────────────────
const PLAN_LABELS: Record<string, string> = {
  free:       "رایگان",
  pro:        "حرفه‌ای",
  enterprise: "سازمانی",
}

const PLAN_CLASSES: Record<string, string> = {
  free:       "bg-secondary text-muted-foreground",
  pro:        "bg-accent/15 text-accent",
  enterprise: "bg-primary/10 text-primary dark:text-primary-foreground",
}

// ── Component ─────────────────────────────────────────────────────────────────
export function Sidebar() {
  const pathname        = usePathname()
  const { user, logout } = useAuth()
  const { theme, setTheme } = useTheme()

  const isDark = theme === "dark"

  return (
    <aside className={cn(
      "flex flex-col h-full w-64 shrink-0",
      "bg-card border-l border-border",
      // موبایل: hidden — دسکتاپ: نشان
      "hidden md:flex",
    )}>

      {/* ── لوگو ────────────────────────────────────────────────────────────── */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center gap-2">
          <Scale className="h-6 w-6 text-accent shrink-0" />
          <span className="text-xl font-bold tracking-tight">JurisFlow</span>
        </div>
        <p className="text-xs text-muted-foreground mt-1">
          دستیار حقوقی هوشمند
        </p>
      </div>

      {/* ── ناوبری ──────────────────────────────────────────────────────────── */}
      <nav className="flex-1 overflow-y-auto p-3 space-y-0.5">
        {NAV_ITEMS.map((item) => {
          const isActive  = pathname.startsWith(item.href)
          const isLocked  = user ? !item.plans.includes(user.plan) : false

          return (
            <Link
              key={item.href}
              href={isLocked ? "#" : item.href}
              onClick={(e) => isLocked && e.preventDefault()}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-150",
                isActive
                  ? "bg-primary text-primary-foreground font-medium shadow-sm"
                  : "text-muted-foreground hover:bg-secondary hover:text-foreground",
                isLocked && "opacity-40 cursor-not-allowed select-none",
              )}
            >
              <item.icon className="h-4 w-4 shrink-0" />
              <span className="flex-1 truncate">{item.label}</span>
              {isLocked && (
                <Lock className="h-3 w-3 opacity-60" />
              )}
              {isLocked && item.badge && (
                <Badge
                  variant="outline"
                  className="text-[10px] px-1.5 py-0 h-4"
                >
                  {item.badge}
                </Badge>
              )}
            </Link>
          )
        })}
      </nav>

      {/* ── پایین sidebar ───────────────────────────────────────────────────── */}
      <div className="p-3 border-t border-border space-y-2">

        {/* پلن کاربر */}
        {user && (
          <div className={cn(
            "flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium",
            PLAN_CLASSES[user.plan],
          )}>
            <span className="truncate">{user.full_name || user.email}</span>
            <span className="shrink-0 mr-2">{PLAN_LABELS[user.plan]}</span>
          </div>
        )}

        <Separator />

        {/* تغییر تم */}
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start gap-3 text-muted-foreground hover:text-foreground"
          onClick={() => setTheme(isDark ? "light" : "dark")}
        >
          {isDark
            ? <Sun  className="h-4 w-4" />
            : <Moon className="h-4 w-4" />
          }
          <span>{isDark ? "حالت روشن" : "حالت تاریک"}</span>
        </Button>

        {/* خروج */}
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start gap-3 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
          onClick={logout}
        >
          <LogOut className="h-4 w-4" />
          <span>خروج از حساب</span>
        </Button>
      </div>
    </aside>
  )
}