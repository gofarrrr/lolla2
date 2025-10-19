export function SocraticQuestionsList({ questions }: { questions: string[] }) {
  if (!questions || questions.length === 0) {
    return (
      <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
        <h3 className="text-lg font-semibold mb-2">Socratic Questions</h3>
        <p className="text-sm text-text-body">No strategic questions were generated for this analysis.</p>
      </section>
    )
  }

  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <h3 className="text-lg font-semibold mb-3">Socratic Questions</h3>
      <ul className="space-y-2">
        {questions.map((q, idx) => (
          <li key={`${idx}-${q.slice(0, 8)}`} className="flex gap-2 text-sm text-text-body">
            <span className="text-bright-green font-bold">?</span>
            <span>{q}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}
