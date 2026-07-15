import { BookOpen } from "lucide-react"
import type { SearchSource } from "@/lib/types/search"

interface Props {
  sources: SearchSource[]
}

export function SourceCard({ sources }: Props) {
  if (!sources.length) return null

  return (
    <div className="mt-3 border border-border rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 bg-secondary/50 border-b border-border">
        <BookOpen className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground">
          منابع قانونی ({sources.length})
        </span>
      </div>

      <div className="flex flex-wrap gap-2 p-3">
        {sources.map((source, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs
                       bg-accent/10 text-accent-foreground border border-accent/20
                       dark:bg-accent/15 dark:border-accent/30"
          >
            {source.article_number
              ? `ماده ${source.article_number}`
              : source.title}
            <span className="text-muted-foreground">—</span>
            <span className="text-muted-foreground truncate max-w-[120px]">
              {source.law}
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}