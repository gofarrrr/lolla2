import { ConsultantVM } from '@/lib/mappers/report'

export function ConsultantMemoViewer({ consultant }: { consultant?: ConsultantVM }) {
  if (!consultant) {
    return (
      <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-2">Consultant Memo</h3>
        <p className="text-sm text-text-body">Select a consultant to view their full memo.</p>
      </section>
    )
  }

  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-lg font-semibold">{consultant.name} — Memo</h3>
        {typeof consultant.selectionScore === 'number' && (
          <span className="text-xs bg-green-50 text-green-700 px-2 py-1 rounded">
            Match {Math.round(consultant.selectionScore * 100)}%
          </span>
        )}
      </div>
      {consultant.memo ? (
        <pre className="whitespace-pre-wrap text-sm text-text-body bg-gray-50 rounded-xl p-4 border border-border-default">
{consultant.memo}
        </pre>
      ) : (
        <p className="text-sm text-text-body">No memo captured for this consultant.</p>
      )}
    </section>
  )
}
