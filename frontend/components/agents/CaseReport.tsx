import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { DebateViewer } from "@/components/agents/DebateViewer"
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  ArrowLeft,
  BookOpen,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { CaseAnalysis } from "@/lib/types/agents"

interface Props {
  analysis: CaseAnalysis
}

const WIN_CHANCE_CONFIG: Record<string, { label: string; className: string }> = {
  "بالا":    { label: "احتمال موفقیت بالا",   className: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400" },
  "متوسط":  { label: "احتمال موفقیت متوسط",  className: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400" },
  "پایین":  { label: "احتمال موفقیت پایین",  className: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400" },
}

const SEVERITY_CLASSES: Record<string, string> = {
  "بالا":   "text-red-600    dark:text-red-400",
  "متوسط": "text-yellow-600 dark:text-yellow-400",
  "پایین": "text-green-600  dark:text-green-400",
}

export function CaseReport({ analysis }: Props) {
  const winConfig = WIN_CHANCE_CONFIG[analysis.win_chance] ?? WIN_CHANCE_CONFIG["متوسط"]

  return (
    <div className="space-y-6 animate-slide-up">

      {/* هدر */}
      <div className="flex flex-wrap items-center gap-3">
        <Badge className={cn("text-sm px-3 py-1", winConfig.className)}>
          {winConfig.label}
        </Badge>
        <Badge variant="outline">{analysis.case_type}</Badge>
        <Badge variant="outline">
          استراتژی: {analysis.strategy}
        </Badge>
      </div>

      {/* موضوع پرونده */}
      <div className="bg-secondary/40 dark:bg-secondary/20 rounded-xl p-4">
        <h3 className="text-xs font-medium text-muted-foreground mb-1">موضوع پرونده</h3>
        <p className="font-medium">{analysis.case_subject}</p>
      </div>

      {/* طرفین */}
      {analysis.parties.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-3">طرفین دعوا</h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {analysis.parties.map((party, i) => (
              <div key={i} className="border border-border rounded-lg p-3 text-sm">
                <span className="text-xs text-muted-foreground block mb-0.5">{party.role}</span>
                <span className="font-medium">{party.name}</span>
                {party.details && (
                  <p className="text-xs text-muted-foreground mt-1">{party.details}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <Separator />

      {/* نقاط قوت و ضعف */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* قوت */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp className="h-4 w-4 text-green-500" />
            <h3 className="text-sm font-medium">نقاط قوت</h3>
          </div>
          <ul className="space-y-2">
            {analysis.strengths.map((s, i) => (
              <li key={i} className="flex gap-2 text-sm">
                <CheckCircle2 className="h-4 w-4 text-green-500 shrink-0 mt-0.5" />
                <span>{s}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* ضعف */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="h-4 w-4 text-red-500" />
            <h3 className="text-sm font-medium">نقاط ضعف</h3>
          </div>
          <ul className="space-y-2">
            {analysis.weaknesses.map((w, i) => (
              <li key={i} className="flex gap-2 text-sm">
                <AlertTriangle className="h-4 w-4 text-red-500 shrink-0 mt-0.5" />
                <span>{w}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ریسک‌ها */}
      {analysis.risks.length > 0 && (
        <>
          <Separator />
          <div>
            <h3 className="text-sm font-medium mb-3">ریسک‌ها</h3>
            <div className="space-y-2">
              {analysis.risks.map((risk, i) => (
                <div key={i} className="border border-border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm font-medium">{risk.title}</span>
                    <span className={cn("text-xs font-medium", SEVERITY_CLASSES[risk.severity])}>
                      {risk.severity}
                    </span>
                  </div>
                  <p className="text-xs text-muted-foreground">{risk.description}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      <Separator />

      {/* مواد قانونی */}
      {analysis.cited_articles.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <BookOpen className="h-4 w-4 text-accent" />
            <h3 className="text-sm font-medium">مواد قانونی مستند</h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {analysis.cited_articles.map((article, i) => (
              <Badge key={i} variant="outline" className="text-xs">
                {article}
              </Badge>
            ))}
          </div>
        </div>
      )}

      <Separator />

      {/* توصیه نهایی */}
      <div className="bg-accent/10 dark:bg-accent/5 border border-accent/20 rounded-xl p-4">
        <h3 className="text-sm font-medium mb-2">توصیه نهایی</h3>
        <p className="text-sm leading-relaxed">{analysis.recommendation}</p>
      </div>

      {/* اقدامات بعدی */}
      {analysis.next_steps.length > 0 && (
        <div>
          <h3 className="text-sm font-medium mb-3">اقدامات پیشنهادی</h3>
          <ol className="space-y-2">
            {analysis.next_steps.map((step, i) => (
              <li key={i} className="flex gap-3 text-sm">
                <span className="shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-bold">
                  {i + 1}
                </span>
                <span className="mt-0.5">{step}</span>
              </li>
            ))}
          </ol>
        </div>
      )}

      <Separator />

      {/* debate */}
      <DebateViewer
        defender={analysis.defender_opinion}
        prosecutor={analysis.prosecutor_opinion}
        judge={analysis.judge_opinion}
      />
    </div>
  )
}