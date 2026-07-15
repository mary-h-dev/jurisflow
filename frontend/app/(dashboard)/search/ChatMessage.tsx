import { cn } from "@/lib/utils"
import { ConfidenceBadge } from "@/components/search/ConfidenceBadge"
import { SourceCard } from "@/components/search/SourceCard"
import { Scale, User } from "lucide-react"
import type { ChatMessage as ChatMessageType } from "@/lib/types/search"

interface Props {
  message: ChatMessageType
}

export function ChatMessage({ message }: Props) {
  const isUser = message.role === "user"

  return (
    <div className={cn(
      "flex gap-3 animate-fade-in",
      isUser ? "flex-row-reverse" : "flex-row",
    )}>
      {/* آواتار */}
      <div className={cn(
        "shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
        isUser
          ? "bg-primary text-primary-foreground"
          : "bg-accent/20 text-accent",
      )}>
        {isUser
          ? <User  className="h-4 w-4" />
          : <Scale className="h-4 w-4" />
        }
      </div>

      {/* محتوا */}
      <div className={cn(
        "flex-1 max-w-[85%] space-y-2",
        isUser ? "items-end" : "items-start",
      )}>
        <div className={cn(
          "rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-primary text-primary-foreground rounded-tr-sm"
            : "bg-card border border-border rounded-tl-sm shadow-sm",
        )}>
          {message.content}
        </div>

        {/* فقط برای پیام‌های assistant */}
        {!isUser && (
          <div className="px-1 space-y-2">
            {message.confidence && (
              <ConfidenceBadge confidence={message.confidence} />
            )}
            {message.sources && message.sources.length > 0 && (
              <SourceCard sources={message.sources} />
            )}
          </div>
        )}

        {/* زمان */}
        <p className={cn(
          "text-[10px] text-muted-foreground px-1",
          isUser ? "text-left" : "text-right",
        )}>
          {new Intl.DateTimeFormat("fa-IR", {
            hour:   "2-digit",
            minute: "2-digit",
          }).format(new Date(message.timestamp))}
        </p>
      </div>
    </div>
  )
}