import { DissentSignalVM } from '@/lib/mappers/report'

export function TriggersTimeline({ signals }: { signals: DissentSignalVM[] }) {
  const items = (signals || []).slice().sort((a, b) => {
    const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0
    const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0
    return ta - tb
  })

  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <h3 className="text-lg font-semibold mb-3">Triggers Timeline</h3>
      {items.length === 0 ? (
        <p className="text-sm text-text-body">No triggers recorded.</p>
      ) : (
        <ul className="space-y-2">
          {items.map((s) => (
            <li key={s.id} className="rounded-xl border border-border-default p-3 bg-white/95">
              <div className="flex items-center justify-between">
                <div className="text-sm font-semibold text-warm-black">{labelForKind(s.kind)}</div>
                <SeverityPill level={s.severity} />
              </div>
              <div className="mt-1 text-sm text-text-body">{s.message}</div>
              {s.timestamp && (
                <div className="mt-1 text-[11px] text-text-label">{new Date(s.timestamp).toLocaleString()}</div>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}

function labelForKind(kind?: string) {
  switch ((kind || '').toLowerCase()) {
    case 'da_bias': return "Devil's Advocate Bias"
    case 'da_complete': return "Devil's Advocate"
    case 'contradiction': return 'Contradiction Detected'
    case 'tension': return 'Tension Identified'
    default: return 'Dissent Signal'
  }
}

function SeverityPill({ level }: { level?: string }) {
  const l = (level || '').toLowerCase()
  if (l === 'high') return <span className="text-xs px-2 py-0.5 rounded bg-red-50 text-red-700 border border-red-200">High</span>
  if (l === 'medium') return <span className="text-xs px-2 py-0.5 rounded bg-yellow-50 text-yellow-700 border border-yellow-200">Medium</span>
  if (l === 'low') return <span className="text-xs px-2 py-0.5 rounded bg-green-50 text-green-700 border border-green-200">Low</span>
  return <span className="text-xs px-2 py-0.5 rounded bg-gray-50 text-gray-700 border border-gray-200">None</span>
}
