"use client"

import { useState, useRef, useEffect } from "react"
import { useQuery } from "@tanstack/react-query"
import { v4 as uuid } from "uuid"
import toast from "react-hot-toast"
import { searchApi } from "@/lib/api/search"
import { ChatMessage } from "@/components/search/ChatMessage"
import { ChatInput } from "@/components/search/ChatInput"
import type { ChatMessage as ChatMessageType } from "@/lib/types/search"
import { Scale } from "lucide-react"

// پیام خوش‌آمدگویی
const WELCOME: ChatMessageType = {
  id:        "welcome",
  role:      "assistant",
  content:   "سلام! من JurisFlow هستم، دستیار حقوقی هوشمند. می‌توانید سوالات حقوقی خود را بپرسید و من بر اساس قوانین ایران پاسخ می‌دهم.",
  timestamp: new Date(),
}

export default function SearchPage() {
  const [messages, setMessages] = useState<ChatMessageType[]>([WELCOME])
  const [loading,  setLoading]  = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)

  // دریافت لیست قوانین
  const { data: lawsData } = useQuery({
    queryKey: ["laws"],
    queryFn:  searchApi.availableLaws,
    staleTime: Infinity,
  })

  const laws = lawsData?.laws ?? ["قانون مدنی"]

  // اسکرول به پایین هر بار که پیام جدید میاد
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const handleSend = async (query: string, law: string) => {
    // اضافه کردن پیام کاربر
    const userMsg: ChatMessageType = {
      id:        uuid(),
      role:      "user",
      content:   query,
      timestamp: new Date(),
    }
    setMessages((prev) => [...prev, userMsg])
    setLoading(true)

    try {
      const result = await searchApi.query(query, law)

      const assistantMsg: ChatMessageType = {
        id:         uuid(),
        role:       "assistant",
        content:    result.answer,
        sources:    result.sources,
        confidence: result.confidence,
        timestamp:  new Date(),
      }
      setMessages((prev) => [...prev, assistantMsg])

    } catch (err: any) {
      // اگه 429 بود — limit تموم شده
      if (err?.response?.status === 429) {
        toast.error("سقف روزانه شما تمام شده. پلن خود را ارتقا دهید.")
      } else {
        toast.error("خطا در دریافت پاسخ. لطفاً دوباره تلاش کنید.")
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen">

      {/* هدر */}
      <header className="shrink-0 flex items-center gap-3 px-6 py-4 border-b border-border bg-background/95 backdrop-blur-sm">
        <Scale className="h-5 w-5 text-accent" />
        <div>
          <h1 className="font-semibold text-sm">جستجوی قانون</h1>
          <p className="text-xs text-muted-foreground">
            پرسش‌های حقوقی خود را مطرح کنید
          </p>
        </div>
      </header>

      {/* لیست پیام‌ها */}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}

          {/* loading indicator */}
          {loading && (
            <div className="flex gap-3 animate-fade-in">
              <div className="shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-accent/20">
                <Scale className="h-4 w-4 text-accent" />
              </div>
              <div className="bg-card border border-border rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                <div className="flex gap-1 items-center h-5">
                  <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:0ms]" />
                  <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:150ms]" />
                  <span className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce [animation-delay:300ms]" />
                </div>
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* ورودی چت */}
      <div className="shrink-0 max-w-3xl mx-auto w-full">
        <ChatInput
          onSend={handleSend}
          isLoading={loading}
          laws={laws}
        />
      </div>
    </div>
  )
}