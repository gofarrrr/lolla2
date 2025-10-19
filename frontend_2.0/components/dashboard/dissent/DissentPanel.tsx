import { DissentSignalVM } from '@/lib/mappers/report'

export function DissentPanel({ signals }: { signals: DissentSignalVM[] }) {
  const total = signals?.length || 0
  const sev = (level: string) => signals.filter(s => s.severity === level).length
  const high = sev('high')
  const medium = sev('medium')
  const low = sev('low')

  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-lg font-semibold">Dissent & Triggers</h3>
        <span className="text-xs text-text-label">{total} signal{total === 1 ? '' : 's'}</span>
      </div>

      {total === 0 ? (
        <p className="text-sm text-text-body">No dissent signals detected for this analysis.</p>
      ) : (
        <div className="space-y-4">
          <div className="flex gap-2 text-xs">
            <span className="px-2 py-1 rounded bg-red-50 text-red-700 border border-red-200">High: {high}</span>
            <span className="px-2 py-1 rounded bg-yellow-50 text-yellow-700 border border-yellow-200">Medium: {medium}</span>
            <span className="px-2 py-1 rounded bg-green-50 text-green-700 border border-green-200">Low: {low}</span>
          </div>

          <ul className="space-y-2">
            {signals.slice(0, 3).map((s) => (
              <li key={s.id} className="rounded-xl border border-border-default p-3 bg-white/95">
                <div className="text-sm font-semibold text-warm-black">
                  {labelForKind(s.kind)}
                </div>
                <div className="mt-1 text-sm text-text-body">{s.message}</div>
                {s.timestamp && (
                  <div className="mt-1 text-[11px] text-text-label">{new Date(s.timestamp).toLocaleString()}</div>
                )}
              </li>
            ))}
          </ul>
        </div>
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
