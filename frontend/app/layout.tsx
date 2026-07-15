import type { Metadata } from "next"
import { Providers } from "@/components/common/Providers"
import "@fontsource/vazirmatn/400.css"
import "@fontsource/vazirmatn/500.css"
import "@fontsource/vazirmatn/700.css"
import "./globals.css"

export const metadata: Metadata = {
  title:       "JurisFlow — دستیار حقوقی هوشمند",
  description: "جستجو و تحلیل هوشمند قوانین ایران با GraphRAG",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    /*
      suppressHydrationWarning لازمه چون next-themes
      کلاس dark/light رو روی html ست می‌کنه
      و ممکنه با server render فرق داشته باشه
    */
    <html lang="fa" dir="rtl" suppressHydrationWarning>
      <body className="font-sans antialiased min-h-screen bg-background text-foreground">
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}