// app/(onboarding)/layout.tsx
export default function OnboardingLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-secondary/30 dark:bg-background px-4">
      <div className="w-full max-w-lg">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-primary dark:text-foreground">
            JurisFlow
          </h1>
          <p className="text-muted-foreground mt-2 text-sm">
            چند قدم مانده تا شروع
          </p>
        </div>
        {children}
      </div>
    </div>
  )
}