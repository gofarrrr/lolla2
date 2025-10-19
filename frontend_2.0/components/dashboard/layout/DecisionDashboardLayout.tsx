import { ReactNode } from 'react'
import { PermanentNav } from '@/components/PermanentNav'
import { LeftRail } from '@/components/dashboard/nav/LeftRail'

export function DecisionDashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-cream-bg text-warm-black">
      <PermanentNav />
      <div className="flex min-h-[calc(100vh-3.5rem)]">
        <LeftRail />
        <main className="flex-1">
          {children}
        </main>
      </div>
    </div>
  )
}
