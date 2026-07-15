"use client"

import { useState } from "react"
import toast from "react-hot-toast"
import { agentsApi } from "@/lib/api/agents"
import { AgentProgress } from "@/components/agents/AgentProgress"
import { CaseReport } from "@/components/agents/CaseReport"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Brain, FileText, Loader2 } from "lucide-react"
import type { CaseAnalysis } from "@/lib/types/agents"

type Step = "input" | "analyzing" | "done"

export default function AnalyzePage() {
  const [step,      setStep]      = useState<Step>("input")
  const [text,      setText]      = useState("")
  const [context,   setContext]   = useState("")
  const [analysis,  setAnalysis]  = useState<CaseAnalysis | null>(null)
  const [completed, setCompleted] = useState<string[]>([])

  const handleAnalyze = async () => {
    if (text.trim().length < 50) {
      toast.error("متن پرونده باید حداقل ۵۰ کاراکتر باشد.")
      return
    }

    setStep("analyzing")
    setCompleted([])

    // simulate progress — چون backend sequential هست
    const nodes = ["analyzer", "search_laws", "defender", "prosecutor", "judge", "lead"]
    let i = 0
    const interval = setInterval(() => {
      if (i < nodes.length - 1) {
        setCompleted((prev) => [...prev, nodes[i]])
        i++
      }
    }, 2500)

    try {
      const result = await agentsApi.analyze(text, context)
      clearInterval(interval)
      setCompleted(result.completed_nodes)
      setAnalysis(result)
      setStep("done")
    } catch (err: any) {
      clearInterval(interval)
      if (err?.response?.status === 403) {
        toast.error("تحلیل پرونده نیاز به پلن Pro دارد.")
      } else {
        toast.error("خطا در تحلیل پرونده. لطفاً دوباره تلاش کنید.")
      }
      setStep("input")
    }
  }

  const handleReset = () => {
    setStep("input")
    setAnalysis(null)
    setCompleted([])
    setText("")
    setContext("")
  }

  return (
    <div className="min-h-screen">
      {/* هدر */}
      <header className="sticky top-0 z-10 flex items-center justify-between px-6 py-4 border-b border-border bg-background/95 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <Brain className="h-5 w-5 text-accent" />
          <div>
            <h1 className="font-semibold text-sm">تحلیل پرونده</h1>
            <p className="text-xs text-muted-foreground">
              تحلیل با Multi-Agent Debate
            </p>
          </div>
        </div>
        {step === "done" && (
          <Button variant="outline" size="sm" onClick={handleReset}>
            پرونده جدید
          </Button>
        )}
      </header>

      <div className="max-w-3xl mx-auto px-4 py-8">

        {/* ── مرحله ورود متن ───────────────────────────────────────────────── */}
        {step === "input" && (
          <div className="space-y-6 animate-fade-in">
            <div className="text-center space-y-2">
              <div className="w-16 h-16 bg-accent/10 rounded-2xl flex items-center justify-center mx-auto">
                <FileText className="h-8 w-8 text-accent" />
              </div>
              <h2 className="text-xl font-bold">متن پرونده را وارد کنید</h2>
              <p className="text-muted-foreground text-sm">
                متن پرونده، قرارداد یا سند حقوقی خود را paste کنید
              </p>
            </div>

            <div className="space-y-3">
              <Textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="متن پرونده یا سند را اینجا وارد کنید..."
                className="min-h-[240px] text-sm leading-relaxed resize-none"
              />

              <Textarea
                value={context}
                onChange={(e) => setContext(e.target.value)}
                placeholder="توضیح اضافه (اختیاری) — مثلاً: می‌خواهم بدانم چه اقداماتی انجام دهم"
                className="min-h-[80px] text-sm resize-none"
              />
            </div>

            <Button
              onClick={handleAnalyze}
              className="w-full h-12 text-base"
              disabled={text.trim().length < 50}
            >
              <Brain className="h-5 w-5 ml-2" />
              شروع تحلیل هوشمند
            </Button>

            <p className="text-xs text-muted-foreground text-center">
              تحلیل توسط ۵ agent مستقل انجام می‌شود و معمولاً ۱۰-۱۵ ثانیه طول می‌کشد
            </p>
          </div>
        )}

        {/* ── مرحله در حال تحلیل ───────────────────────────────────────────── */}
        {step === "analyzing" && (
          <div className="space-y-8 animate-fade-in">
            <div className="text-center space-y-2">
              <Loader2 className="h-12 w-12 animate-spin text-accent mx-auto" />
              <h2 className="text-lg font-bold">در حال تحلیل پرونده...</h2>
              <p className="text-sm text-muted-foreground">
                سه agent به صورت موازی در حال بررسی هستند
              </p>
            </div>

            <AgentProgress
              completedNodes={completed}
              isLoading={true}
            />
          </div>
        )}

        {/* ── نتیجه ────────────────────────────────────────────────────────── */}
        {step === "done" && analysis && (
          <div className="space-y-6">
            <AgentProgress
              completedNodes={completed}
              isLoading={false}
            />
            <CaseReport analysis={analysis} />
          </div>
        )}
      </div>
    </div>
  )
}