import { ConsultantVM } from '@/lib/mappers/report'
import clsx from 'clsx'

export function ConsultantList({
  consultants,
  selectedId,
  onSelect,
}: {
  consultants: ConsultantVM[]
  selectedId?: string
  onSelect: (id: string) => void
}) {
  if (!consultants || consultants.length === 0) {
    return (
      <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-2">Consultant Team</h3>
        <p className="text-sm text-text-body">No consultants found for this analysis.</p>
      </section>
    )
  }

  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <h3 className="text-lg font-semibold mb-4">Consultant Team</h3>
      <div className="grid gap-3 md:grid-cols-2">
        {consultants.map((c) => (
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={clsx(
              'rounded-xl border p-4 text-left hover:border-bright-green transition',
              'border-border-default bg-white',
              selectedId === c.id && 'ring-2 ring-bright-green shadow-lg'
            )}
          >
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="text-base font-semibold">{c.name}</div>
                <div className="text-xs text-text-label">{c.role ?? 'Advisor'}</div>
              </div>
              {typeof c.selectionScore === 'number' && (
                <span className="text-xs bg-green-50 text-green-700 px-2 py-1 rounded">
                  Match {Math.round(c.selectionScore * 100)}%
                </span>
              )}
            </div>
            {(c.keyInsights?.length ?? 0) > 0 && (
              <div className="mt-2 text-sm text-text-body line-clamp-2">
                {c.keyInsights[0]}
              </div>
            )}
          </button>
        ))}
      </div>
    </section>
  )
}
