import { MeceNode } from '@/lib/mappers/report'

export function MECEFrameworkVisualizer({ nodes }: { nodes: MeceNode[] }) {
  if (!nodes || nodes.length === 0) {
    return (
      <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-2">MECE Framework</h3>
        <p className="text-sm text-text-body">No problem structure captured for this analysis.</p>
      </section>
    )
  }

  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <h3 className="text-lg font-semibold mb-4">MECE Framework</h3>
      <div className="grid gap-3 md:grid-cols-2">
        {nodes.map((n) => (
          <div key={n.id} className="rounded-xl border border-border-default p-4 hover:border-bright-green transition">
            <div className="flex items-start justify-between gap-2">
              <div className="text-base font-semibold">{n.label}</div>
              {typeof n.heat === 'number' && (
                <span className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded">
                  {Math.round(n.heat * 100)}% focus
                </span>
              )}
            </div>
            {n.description && <p className="mt-2 text-sm text-text-body">{n.description}</p>}
          </div>
        ))}
      </div>
    </section>
  )
}
