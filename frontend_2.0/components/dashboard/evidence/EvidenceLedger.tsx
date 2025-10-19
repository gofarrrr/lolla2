import { EvidenceVM } from '@/lib/mappers/report'

export function EvidenceLedger({ items }: { items: EvidenceVM[] }) {
  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <h3 className="text-lg font-semibold mb-3">Evidence Ledger</h3>
      {(!items || items.length === 0) ? (
        <p className="text-sm text-text-body">No evidence items captured for this analysis.</p>
      ) : (
        <ul className="space-y-2">
          {items.map((it) => (
            <li key={it.id} className="rounded-xl border border-border-default p-3 bg-white/95">
              <div className="text-xs font-semibold text-bright-green">{it.source}</div>
              <div className="mt-1 text-sm text-warm-black">{it.snippet}</div>
              <div className="mt-1 text-[11px] text-text-label">
                {it.provenance ? `${it.provenance.toUpperCase()} SOURCE` : 'SOURCE'}
                {typeof it.confidence === 'number' && ` • Confidence ${Math.round(it.confidence * 100)}%`}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
