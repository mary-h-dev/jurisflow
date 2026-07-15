
import { Sidebar } from "../../components/layout/Sidebar"

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar — راست صفحه (RTL) */}
      <Sidebar />

      {/* محتوای اصلی */}
      <main className="flex-1 overflow-y-auto">
        {children}
      </main>
    </div>
  )
}