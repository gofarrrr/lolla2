import Link from 'next/link'
import { useDashboardStore } from '@/lib/state/dashboardStore'
import clsx from 'clsx'

export function LeftRail() {
  const { leftRailCollapsed, toggleLeftRail, setActiveTab } = useDashboardStore()

  const items = [
    { key: 'strategy', label: 'Strategy' },
    { key: 'process', label: 'Process' },
    { key: 'team', label: 'Team' },
    { key: 'evidence', label: 'Evidence' },
    { key: 'raw', label: 'Raw' },
  ] as const

  return (
    <aside
      className={clsx(
        'border-r border-border-default bg-cream-bg/70 backdrop-blur-sm transition-all duration-300',
        leftRailCollapsed ? 'w-[56px]' : 'w-[200px]'
      )}
    >
      <div className="flex items-center justify-between px-3 py-2">
        <div className={clsx('text-[11px] font-semibold uppercase tracking-wider text-text-label', leftRailCollapsed && 'opacity-0')}>Navigation</div>
        <button
          aria-label="Toggle navigation"
          className="text-text-label hover:text-warm-black"
          onClick={toggleLeftRail}
        >
          {leftRailCollapsed ? '»' : '«'}
        </button>
      </div>
      <nav className="flex flex-col gap-1 px-2 pb-3">
        {items.map((it) => (
          <button
            key={it.key}
            onClick={() => setActiveTab(it.key as any)}
            className={clsx(
              'text-left rounded-lg px-3 py-2 text-sm hover:bg-white border border-transparent hover:border-border-default',
              'text-text-body hover:text-warm-black'
            )}
          >
            {leftRailCollapsed ? it.label[0] : it.label}
          </button>
        ))}
      </nav>
    </aside>
  )
}
