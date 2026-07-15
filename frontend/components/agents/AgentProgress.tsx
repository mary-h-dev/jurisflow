"use client"

import { cn } from "@/lib/utils"
import { Check, Loader2 } from "lucide-react"

const NODE_LABELS: Record<string, string> = {
  analyzer:   "تحلیل سند",
  search_laws:"جستجوی قوانین",
  defender:   "وکیل مدافع",
  prosecutor: "دادستان",
  judge:      "قاضی",
  lead:       "جمع‌بندی نهایی",
}

interface Props {
  completedNodes: string[]
  isLoading:      boolean
}

const ALL_NODES = ["analyzer", "search_laws", "defender", "prosecutor", "judge", "lead"]

export function AgentProgress({ completedNodes, isLoading }: Props) {
  const currentIndex = completedNodes.length

  return (
    <div className="w-full py-4">
      <div className="flex items-center justify-between relative">
        {/* خط اتصال */}
        <div className="absolute top-4 right-4 left-4 h-0.5 bg-border" />
        <div
          className="absolute top-4 right-4 h-0.5 bg-accent transition-all duration-500"
          style={{
            width: `${(completedNodes.length / ALL_NODES.length) * (100 - 8)}%`,
          }}
        />

        {ALL_NODES.map((node, i) => {
          const isDone    = completedNodes.includes(node)
          const isCurrent = isLoading && i === currentIndex

          return (
            <div key={node} className="flex flex-col items-center gap-2 z-10">
              <div className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all duration-300",
                isDone
                  ? "bg-accent border-accent text-white"
                  : isCurrent
                  ? "bg-background border-accent animate-pulse"
                  : "bg-background border-border",
              )}>
                {isDone
                  ? <Check className="h-4 w-4 text-white" />
                  : isCurrent
                  ? <Loader2 className="h-4 w-4 text-accent animate-spin" />
                  : <span className="text-xs text-muted-foreground">{i + 1}</span>
                }
              </div>
              <span className={cn(
                "text-[10px] text-center max-w-[60px] leading-tight",
                isDone    ? "text-accent font-medium" :
                isCurrent ? "text-foreground font-medium" :
                            "text-muted-foreground",
              )}>
                {NODE_LABELS[node]}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}