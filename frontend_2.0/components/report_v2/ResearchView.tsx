"use client";

import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { RawJson } from './RawJson'
import { useGlassBoxStore } from '@/lib/state/glassBox'

export function ResearchView({ data }: { data: any }) {
  const { glassBox } = useGlassBoxStore()
  const research = data?.hybrid_data_research || data?.report?.hybrid_data_research
  const providerEvents = data?.research_provider_events

  if (glassBox) {
    return (
      <div className="space-y-4">
        <RawJson title="Research (Raw)" data={research ?? {}} />
        {providerEvents && <RawJson title="Research Provider Events (Raw)" data={providerEvents} />}
      </div>
    )
  }

  const memo = research?.briefing_memo

  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">Research Findings</h3>
        <div className="text-xs text-text-label">Source: Oracle + Perplexity</div>
      </div>

      {memo ? (
        <div className="space-y-4">
          {typeof memo.summary === 'string' && (
            <article className="prose prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{memo.summary}</ReactMarkdown>
            </article>
          )}
          {Array.isArray(memo.key_findings) && memo.key_findings.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-2">Key Findings</h4>
              <ul className="list-disc pl-5 text-sm text-text-body space-y-1">
                {memo.key_findings.map((f: any, idx: number) => (
                  <li key={idx}>{typeof f === 'string' ? f : f?.text || 'Finding'}</li>
                ))}
              </ul>
            </div>
          )}
          {Array.isArray(memo.citations) && memo.citations.length > 0 && (
            <div>
              <h4 className="text-sm font-semibold mb-2">Sources</h4>
              <ul className="list-decimal pl-5 text-sm text-text-body space-y-2">
                {memo.citations.map((c: any, idx: number) => (
                  <li key={idx}>
                    <div className="font-medium">{c?.source || c?.title || 'Source'}</div>
                    {c?.confidence != null && (
                      <div className="text-xs text-text-label">Confidence: {Math.round((c.confidence || 0) * 100)}%</div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ) : (
        Array.isArray(providerEvents) && providerEvents.length > 0 ? null : (
          <p className="text-sm text-text-body">No research briefing memo available.</p>
        )
      )}

      {Array.isArray(providerEvents) && providerEvents.length > 0 && (
        <div className="mt-6">
          <h4 className="text-sm font-semibold mb-2">Research Queries</h4>
          <ul className="text-sm text-text-body space-y-2">
            {providerEvents.map((ev: any, idx: number) => (
              <li key={ev?.event_id || idx} className="rounded-xl border border-border-default p-3">
                <div className="flex flex-wrap gap-2 items-center">
                  <span className="text-xs bg-cream-bg px-2 py-1 rounded">{ev?.provider || 'provider'}</span>
                  {ev?.query && <span className="truncate">{ev.query}</span>}
                </div>
                <div className="text-xs text-text-label mt-1">
                  {ev?.cost_usd != null && <span>Cost: ${ev.cost_usd} · </span>}
                  {ev?.duration_ms != null && <span>Duration: {Math.round(ev.duration_ms / 1000)}s</span>}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
