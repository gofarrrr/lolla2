"use client";

import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { ChevronLeft, Menu } from 'lucide-react'

export type SidebarConsultant = { id: string; name: string }

export type LeftSidebarProps = {
  consultants: SidebarConsultant[]
  activeKey: string
  onChange: (key: string) => void
  collapsed?: boolean
  onToggleCollapse?: () => void
}

export function LeftSidebar({ consultants, activeKey, onChange, collapsed = false, onToggleCollapse }: LeftSidebarProps) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [isCollapsed, setIsCollapsed] = useState<boolean>(collapsed)

  useEffect(() => {
    const stored = localStorage.getItem('report-v2-sidebar-collapsed')
    if (stored != null) {
      try { setIsCollapsed(JSON.parse(stored)) } catch {}
    }
  }, [])

  useEffect(() => {
    localStorage.setItem('report-v2-sidebar-collapsed', JSON.stringify(isCollapsed))
  }, [isCollapsed])

  const toggleExpanded = (id: string) => setExpanded((s) => ({ ...s, [id]: !s[id] }))

  const NavItem = ({ label, keyVal }: { label: string; keyVal: string }) => (
    <button
      type="button"
      onClick={() => onChange(keyVal)}
      className={clsx(
        'w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors border-l-2',
        activeKey === keyVal ? 'border-accent-green text-warm-black' : 'border-transparent text-text-body hover:bg-cream-bg/40'
      )}
    >
      {label}
    </button>
  )

  if (isCollapsed) {
    return (
      <aside className="w-[50px] shrink-0">
        <div className="sticky top-16 py-4 flex flex-col items-center gap-3">
          <button
            type="button"
            title="Expand navigation"
            aria-label="Expand navigation"
            onClick={() => { setIsCollapsed(false); onToggleCollapse?.() }}
            className="w-9 h-9 rounded-lg bg-white border border-border-default shadow-sm hover:shadow-md hover:bg-cream-bg transition-all flex items-center justify-center"
          >
            <Menu size={16} className="text-text-body" />
          </button>
        </div>
      </aside>
    )
  }

  return (
    <aside className="w-[220px] shrink-0">
      <nav className="sticky top-16 py-4">
        <div className="flex items-center justify-between px-2 mb-3">
          <div className="text-[10px] font-bold tracking-[0.12em] text-text-label uppercase">Explore</div>
          <button
            type="button"
            title="Collapse navigation"
            aria-label="Collapse navigation"
            onClick={() => { setIsCollapsed(true); onToggleCollapse?.() }}
            className="p-1.5 rounded-md hover:bg-cream-bg transition-colors"
          >
            <ChevronLeft size={14} className="text-text-label" />
          </button>
        </div>
        <div className="space-y-0.5">
          <NavItem label="Summary" keyVal="summary" />
          <NavItem label="Senior Advisor" keyVal="senior" />

          {consultants.length > 0 && (
            <>
              <div className="px-2 pt-3 pb-1.5 text-[10px] uppercase tracking-[0.12em] text-text-label font-bold">
                Consultants
              </div>
              <div className="space-y-0.5">
                {consultants.map((c) => (
                  <div key={c.id}>
                    <button
                      type="button"
                      className={clsx(
                        'w-full text-left px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center justify-between group border-l-2',
                        activeKey.startsWith(`consultant:${c.id}`) ? 'border-accent-green text-warm-black' : 'border-transparent text-text-body hover:bg-cream-bg/40'
                      )}
                      onClick={() => {
                        toggleExpanded(c.id);
                        if (!expanded[c.id]) {
                          onChange(`consultant:${c.id}:analysis`);
                        }
                      }}
                    >
                      <span className="truncate text-sm">{c.name}</span>
                      <span className={clsx('ml-2 text-xs text-text-label transition-transform', expanded[c.id] && 'rotate-90')}>▸</span>
                    </button>
                    {expanded[c.id] && (
                      <div className="ml-3 mt-0.5 space-y-0.5 border-l-2 border-border-default pl-2">
                        <button
                          type="button"
                          onClick={() => onChange(`consultant:${c.id}:analysis`)}
                          className={clsx(
                            'w-full text-left px-2 py-1 rounded text-xs transition-colors',
                            activeKey === `consultant:${c.id}:analysis`
                              ? 'text-warm-black font-medium border-l-2 border-accent-green'
                              : 'text-text-body hover:border-accent-green/40 border-l-2 border-transparent'
                          )}
                        >
                          Analysis
                        </button>
                        <button
                          type="button"
                          onClick={() => onChange(`consultant:${c.id}:da`)}
                          className={clsx(
                            'w-full text-left px-2 py-1 rounded text-xs transition-colors',
                            activeKey === `consultant:${c.id}:da`
                              ? 'text-warm-black font-medium border-l-2 border-accent-green'
                              : 'text-text-body hover:border-accent-green/40 border-l-2 border-transparent'
                          )}
                        >
                          DA Critique
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}

          <div className="pt-1">
            <NavItem label="Research" keyVal="research" />
          </div>
        </div>
      </nav>
    </aside>
  )
}
