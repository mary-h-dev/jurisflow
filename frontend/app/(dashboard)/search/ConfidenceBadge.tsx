import { cn } from "@/lib/utils"
import type { Confidence } from "@/lib/types/search"

interface Props {
  confidence:    Confidence
  showBreakdown?: boolean
}

const LEVEL_CONFIG = {
  high: {
    label:     "اطمینان بالا",
    className: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
  },
  medium: {
    label:     "اطمینان متوسط",
    className: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
  },
  low: {
    label:     "اطمینان پایین",
    className: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
  },
}

export function ConfidenceBadge({ confidence, showBreakdown = false }: Props) {
  const config = LEVEL_CONFIG[confidence.level] ?? LEVEL_CONFIG.medium

  return (
    <div className="space-y-1.5">
      <span className={cn(
        "inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium",
        config.className,
      )}>
        <span className="w-1.5 h-1.5 rounded-full bg-current" />
        {config.label} — {Math.round(confidence.final * 100)}٪
      </span>

      {confidence.note && (
        <p className="text-xs text-muted-foreground">{confidence.note}</p>
      )}

      {showBreakdown && (
        <div className="flex gap-3 text-[11px] text-muted-foreground">
          <span>معنایی: {Math.round(confidence.breakdown.embedding * 100)}٪</span>
          <span>هوش مصنوعی: {Math.round(confidence.breakdown.llm * 100)}٪</span>
          <span>گراف: {Math.round(confidence.breakdown.graph * 100)}٪</span>
        </div>
      )}
    </div>
  )
}