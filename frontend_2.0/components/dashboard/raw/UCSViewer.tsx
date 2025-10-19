export function UCSViewerPlaceholder() {
  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <h3 className="text-lg font-semibold mb-3">Raw Context Stream</h3>
      <p className="text-sm text-text-body">
        UnifiedContextStream events viewer will appear here. In the next iteration, we will
        fetch and filter events (LLM calls, research providers, errors) for full forensics.
      </p>
    </section>
  )
}
