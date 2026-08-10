import { Sidebar } from "@/components/layout/Sidebar"

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div
      style={{ display: "flex", height: "100vh", overflow: "hidden" }}
      className="bg-background"
    >
      {/* Sidebar — سمت راست در RTL */}
      <Sidebar />

      {/* محتوای اصلی */}
      <main
        style={{ flex: 1, overflowY: "auto" }}
        className="bg-background"
      >
        {children}
      </main>
    </div>
  )
}