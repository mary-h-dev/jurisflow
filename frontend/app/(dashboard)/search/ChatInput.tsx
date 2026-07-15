"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Send, Loader2 } from "lucide-react"

interface Props {
  onSend:    (query: string, law: string) => void
  isLoading: boolean
  laws:      string[]
}

export function ChatInput({ onSend, isLoading, laws }: Props) {
  const [query,       setQuery]      = useState("")
  const [selectedLaw, setSelectedLaw] = useState(laws[0] ?? "قانون مدنی")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault()
    const trimmed = query.trim()
    if (!trimmed || isLoading) return
    onSend(trimmed, selectedLaw)
    setQuery("")
    textareaRef.current?.focus()
  }

  // ارسال با Enter (بدون Shift)
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t border-border bg-background/95 backdrop-blur-sm p-4"
    >
      {/* انتخاب قانون */}
      <div className="flex gap-2 mb-2">
        <Select value={selectedLaw} onValueChange={setSelectedLaw}>
          <SelectTrigger className="w-48 h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {laws.map((law) => (
              <SelectItem key={law} value={law} className="text-xs">
                {law}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* ورودی متن */}
      <div className="flex gap-2 items-end">
        <Textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="سوال حقوقی خود را بنویسید... (Enter برای ارسال)"
          className="min-h-[56px] max-h-[160px] resize-none text-sm leading-relaxed"
          disabled={isLoading}
          rows={1}
        />
        <Button
          type="submit"
          size="icon"
          className="h-14 w-14 shrink-0"
          disabled={!query.trim() || isLoading}
        >
          {isLoading
            ? <Loader2 className="h-5 w-5 animate-spin" />
            : <Send    className="h-5 w-5" />
          }
        </Button>
      </div>

      <p className="text-[10px] text-muted-foreground mt-2 text-center">
        JurisFlow اطلاعات حقوقی ارائه می‌دهد، نه مشاوره حقوقی رسمی.
      </p>
    </form>
  )
}