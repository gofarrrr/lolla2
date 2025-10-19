import { ResearchProviderEventVM } from '@/lib/mappers/report'

export function ResearchProviderTrail({ events }: { events: ResearchProviderEventVM[] }) {
  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <h3 className="text-lg font-semibold mb-3">Research Provider Trail</h3>
      {(!events || events.length === 0) ? (
        <p className="text-sm text-text-body">No research provider events recorded.</p>
      ) : (
        <ul className="space-y-2">
          {events.map((ev) => (
            <li key={ev.id} className="rounded-xl border border-border-default p-3 bg-white/95">
              <div className="text-sm font-semibold text-warm-black flex items-center justify-between">
                <span>{ev.provider ?? 'Provider'}</span>
                <span className="text-xs text-text-label">{ev.eventType}</span>
              </div>
              <div className="mt-1 text-xs text-text-label flex gap-4">
                {typeof ev.citations === 'number' && <span>Citations: {ev.citations}</span>}
                {typeof ev.latencyMs === 'number' && <span>Latency: {ev.latencyMs} ms</span>}
                {ev.fallbackProvider && <span>Fallback: {ev.fallbackProvider}</span>}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
