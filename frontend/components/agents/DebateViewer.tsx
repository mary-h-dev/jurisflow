"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import { ChevronDown, ChevronUp, Shield, Swords, Scale } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import type { TeammateOpinion } from "@/lib/types/agents"

interface Props {
  defender:   TeammateOpinion | null
  prosecutor: TeammateOpinion | null
  judge:      TeammateOpinion | null
}

const TEAMMATE_CONFIG = {
  defender: {
    label:     "وکیل مدافع",
    icon:      Shield,
    className: "border-green-200  bg-green-50/50  dark:border-green-800  dark:bg-green-950/30",
    badge:     "bg-green-100 text-green-700 dark:bg-green-900/50 dark:text-green-400",
  },
  prosecutor: {
    label:     "دادستان",
    icon:      Swords,
    className: "border-red-200    bg-red-50/50    dark:border-red-800    dark:bg-red-950/30",
    badge:     "bg-red-100 text-red-700 dark:bg-red-900/50 dark:text-red-400",
  },
  judge: {
    label:     "قاضی بی‌طرف",
    icon:      Scale,
    className: "border-blue-200   bg-blue-50/50   dark:border-blue-800   dark:bg-blue-950/30",
    badge:     "bg-blue-100 text-blue-700 dark:bg-blue-900/50 dark:text-blue-400",
  },
}

function TeammateCard({ opinion }: { opinion: TeammateOpinion }) {
  const config = TEAMMATE_CONFIG[opinion.role as keyof typeof TEAMMATE_CONFIG]
  if (!config) return null
  const Icon = config.icon

  return (
    <div className={cn("rounded-xl border p-4 space-y-3", config.className)}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4" />
          <span className="font-medium text-sm">{config.label}</span>
        </div>
        <span className={cn("text-xs px-2 py-0.5 rounded-full font-medium", config.badge)}>
          {opinion.confidence}
        </span>
      </div>

      <p className="text-sm text-muted-foreground italic">
        «{opinion.position}»
      </p>

      <ul className="space-y-1.5">
        {opinion.arguments.map((arg, i) => (
          <li key={i} className="flex gap-2 text-sm">
            <span className="shrink-0 text-muted-foreground">{i + 1}.</span>
            <span>{arg}</span>
          </li>
        ))}
      </ul>

      {opinion.cited_articles.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {opinion.cited_articles.map((article, i) => (
            <Badge key={i} variant="outline" className="text-xs">
              {article}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}

export function DebateViewer({ defender, prosecutor, judge }: Props) {
  const [open, setOpen] = useState(false)

  const opinions = [defender, prosecutor, judge].filter(Boolean) as TeammateOpinion[]
  if (!opinions.length) return null

  return (
    <div className="border border-border rounded-xl overflow-hidden">
      <Button
        variant="ghost"
        className="w-full flex items-center justify-between px-4 py-3 h-auto rounded-none border-b border-border bg-secondary/30"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2">
          <Scale className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">مشاهده debate حقوقی</span>
          <Badge variant="secondary" className="text-xs">
            {opinions.length} نظر
          </Badge>
        </div>
        {open
          ? <ChevronUp   className="h-4 w-4 text-muted-foreground" />
          : <ChevronDown className="h-4 w-4 text-muted-foreground" />
        }
      </Button>

      {open && (
        <div className="p-4 space-y-3 animate-fade-in">
          {opinions.map((opinion) => (
            <TeammateCard key={opinion.role} opinion={opinion} />
          ))}
        </div>
      )}
    </div>
  )
}