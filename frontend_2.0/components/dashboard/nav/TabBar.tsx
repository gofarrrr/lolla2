import clsx from 'clsx'
import { useDashboardStore } from '@/lib/state/dashboardStore'

export function TabBar() {
  const { activeTab, setActiveTab } = useDashboardStore()
  const tabs = [
    { key: 'strategy', label: 'Strategy' },
    { key: 'process', label: 'Process' },
    { key: 'team', label: 'Team' },
    { key: 'evidence', label: 'Evidence' },
  ] as const

  return (
    <div className="border-b border-border-default bg-cream-bg/80 backdrop-blur-md">
      <div className="mx-auto max-w-7xl px-6 py-2 flex gap-2">
        {tabs.map((t) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key as any)}
            className={clsx(
              'rounded-full px-4 py-1.5 text-sm font-medium transition',
              activeTab === t.key
                ? 'bg-bright-green text-white shadow'
                : 'text-text-body hover:text-warm-black'
            )}
          >
            {t.label}
          </button>
        ))}
      </div>
    </div>
  )
}
