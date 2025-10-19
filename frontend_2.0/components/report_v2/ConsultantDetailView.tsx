"use client";

import React, { useMemo } from 'react'
import { RawJson } from './RawJson'
import { useGlassBoxStore } from '@/lib/state/glassBox'
import { PERSONA_LABELS } from '@/lib/constants/personas'

export function ConsultantDetailView({ data, consultantId, view }: { data: any; consultantId: string; view: 'analysis' | 'da' }) {
  const { glassBox } = useGlassBoxStore()

  const consultant = useMemo(() => {
    const arr = data?.parallel_analysis?.consultant_analyses || data?.consultant_analyses || []
    return Array.isArray(arr) ? arr.find((c: any) => c?.consultant_id === consultantId || c?.id === consultantId) : undefined
  }, [data, consultantId])

  if (glassBox) {
    if (view === 'da') {
      return <RawJson title={`Devil's Advocate (${consultantId})`} data={consultant?.devils_advocate ?? data?.devils_advocate ?? {}} />
    }
    return <RawJson title={`Consultant Analysis (${consultantId})`} data={consultant ?? {}} />
  }

  if (!consultant) {
    return (
      <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
        <div className="text-center py-8">
          <div className="w-12 h-12 rounded-full bg-gray-100 mx-auto mb-3 flex items-center justify-center">
            <svg className="w-6 h-6 text-gray-400" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
              <path d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-warm-black mb-2">
            Consultant Data Not Available
          </h3>
          <p className="text-sm text-text-body max-w-md mx-auto">
            This analysis is complete, but consultant analyses are not included in the report bundle.
            The senior advisor synthesis and recommendations are available in other sections.
          </p>
        </div>
      </section>
    )
  }

  if (view === 'da') {
    const da = consultant?.devils_advocate || data?.devils_advocate
    return (
      <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
        <div className="flex items-start justify-between mb-3">
          <h3 className="text-lg font-semibold">Devil's Advocate: {consultant?.consultant_name || PERSONA_LABELS[String(consultant?.consultant_id || consultantId)] || consultant?.name || consultantId}</h3>
          {typeof da?.robustness_score === 'number' && (
            <span className="text-xs px-2 py-1 rounded border border-accent-green bg-white text-warm-black">Robustness {Math.round(da.robustness_score * 1000) / 10}%</span>
          )}
        </div>
        {Array.isArray(da?.challenged_assumptions) && da.challenged_assumptions.length > 0 && (
          <div className="mb-3">
            <h4 className="text-sm font-semibold mb-1">Challenged Assumptions</h4>
            <ul className="list-disc pl-5 text-sm text-text-body space-y-1">
              {da.challenged_assumptions.map((a: any, idx: number) => (
                <li key={idx}>
                  <div className="font-medium">{a?.assumption || 'Assumption'}</div>
                  <div className="text-xs text-text-label">Before: {Math.round((a?.before_weight ?? 0) * 100)}% · After: {Math.round((a?.after_weight ?? 0) * 100)}%</div>
                </li>
              ))}
            </ul>
          </div>
        )}
        {Array.isArray(da?.key_challenges) && da.key_challenges.length > 0 && (
          <div className="mb-3">
            <h4 className="text-sm font-semibold mb-1">Key Challenges</h4>
            <ul className="list-disc pl-5 text-sm text-text-body space-y-1">
              {da.key_challenges.map((c: any, idx: number) => (
                <li key={idx}>{typeof c === 'string' ? c : c?.text || 'Challenge'}</li>
              ))}
            </ul>
          </div>
        )}
        {typeof da?.da_transcript === 'string' && (
          <div>
            <h4 className="text-sm font-semibold mb-1">Transcript Excerpt</h4>
            <pre className="whitespace-pre-wrap text-sm text-text-body bg-gray-50 rounded-xl p-4 border border-border-default max-h-80 overflow-auto">{da.da_transcript.slice(0, 4000)}</pre>
          </div>
        )}
      </section>
    )
  }

  // analysis view
  return (
    <section className="bg-white rounded-2xl border border-border-default p-6 shadow-sm">
      <div className="flex items-start justify-between mb-3">
        <h3 className="text-lg font-semibold">{consultant?.consultant_name || PERSONA_LABELS[String(consultant?.consultant_id || consultantId)] || consultant?.name || 'Consultant'} — Analysis</h3>
        {typeof consultant?.selection_score === 'number' && (
          <span className="text-xs px-2 py-1 rounded border border-accent-green bg-white text-warm-black">Selection {Math.round(consultant.selection_score * 100)}%</span>
        )}
      </div>
      {Array.isArray(consultant?.key_insights) && consultant.key_insights.length > 0 && (
        <div className="mb-3">
          <h4 className="text-sm font-semibold mb-1">Key Insights</h4>
          <ul className="list-disc pl-5 text-sm text-text-body space-y-1">
            {consultant.key_insights.map((i: any, idx: number) => (
              <li key={idx}>{typeof i === 'string' ? i : i?.text || 'Insight'}</li>
            ))}
          </ul>
        </div>
      )}
      {Array.isArray(consultant?.recommendations) && consultant.recommendations.length > 0 && (
        <div className="mb-3">
          <h4 className="text-sm font-semibold mb-1">Recommendations</h4>
          <ol className="list-decimal pl-5 text-sm text-text-body space-y-1">
            {consultant.recommendations.map((r: any, idx: number) => (
              <li key={idx}>{typeof r === 'string' ? r : r?.text || r?.recommendation || 'Recommendation'}</li>
            ))}
          </ol>
        </div>
      )}
      {Array.isArray(consultant?.risk_factors) && consultant.risk_factors.length > 0 && (
        <div className="mb-3">
          <h4 className="text-sm font-semibold mb-1">Risk Factors</h4>
          <ul className="list-disc pl-5 text-sm text-text-body space-y-1">
            {consultant.risk_factors.map((r: any, idx: number) => (
              <li key={idx}>{typeof r === 'string' ? r : r?.text || 'Risk'}</li>
            ))}
          </ul>
        </div>
      )}
      {Array.isArray(consultant?.opportunities) && consultant.opportunities.length > 0 && (
        <div className="mb-3">
          <h4 className="text-sm font-semibold mb-1">Opportunities</h4>
          <ul className="list-disc pl-5 text-sm text-text-body space-y-1">
            {consultant.opportunities.map((o: any, idx: number) => (
              <li key={idx}>{typeof o === 'string' ? o : o?.text || 'Opportunity'}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
