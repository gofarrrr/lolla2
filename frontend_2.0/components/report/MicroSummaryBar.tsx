'use client';

import React from 'react';

interface MicroSummaryBarProps {
  decision: 'PROCEED' | 'CAUTION' | 'HALT' | 'ANALYSIS COMPLETE';
  confidence: number; // 0-1
  riskLabel: 'LOW' | 'MEDIUM' | 'MEDIUM-HIGH' | 'HIGH';
  riskFacets?: { key: string; label: string; level: number }[];
  governingThought: string;
}

function DecisionChip({ decision }: { decision: MicroSummaryBarProps['decision'] }) {
  const styleMap: Record<string, { bg: string; text: string; border: string }> = {
    PROCEED: { bg: 'bg-gray-900', text: 'text-white', border: 'border-black/5' },
    CAUTION: { bg: 'bg-gray-100', text: 'text-gray-900', border: 'border-black/5' },
    HALT: { bg: 'bg-white', text: 'text-gray-900', border: 'border-gray-300' },
    'ANALYSIS COMPLETE': { bg: 'bg-gray-800', text: 'text-white', border: 'border-black/5' },
  };

  const normalizedDecision = decision.toUpperCase() as keyof typeof styleMap;
  const { bg, text, border } = styleMap[normalizedDecision] || styleMap['ANALYSIS COMPLETE'];

  return (
    <div
      className={`whitespace-nowrap rounded-xl px-4 py-2 text-sm font-medium ${bg} ${text} border ${border} shadow-[0_6px_20px_rgba(0,0,0,0.08)]`}
    >
      {decision.replace('_', ' ')}
    </div>
  );
}

function ConfidenceBar({ value, width = 160 }: { value: number; width?: number }) {
  const pct = Math.round(value * 100);
  const fill =
    pct >= 80 ? 'bg-gray-800' : pct >= 50 ? 'bg-gray-600' : 'bg-gray-400';

  return (
    <div className="flex items-center gap-2" style={{ width }}>
      <div className="relative h-3 w-full overflow-hidden rounded-full border border-black/5 bg-white shadow-inner">
        <div className={`h-full ${fill}`} style={{ width: `${pct}%` }} />
        <div className="absolute inset-0 flex items-center justify-end pr-2 text-[11px] font-semibold text-gray-900">
          {pct}%
        </div>
      </div>
    </div>
  );
}

function RiskBand({
  label,
  facets,
  width = 140,
}: {
  label: MicroSummaryBarProps['riskLabel'];
  facets?: { key: string; label: string; level: number }[];
  width?: number;
}) {
  if (!facets || facets.length === 0) {
    return (
      <div className="flex items-center gap-2">
        <span className="rounded-full border border-black/5 bg-white px-3 py-1 text-xs font-semibold text-gray-700 shadow-sm">
          {label}
        </span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2" style={{ width }}>
      <div className="flex h-3 w-full items-stretch overflow-hidden rounded-full border border-black/5 bg-white shadow-inner">
        {facets.map((f) => (
          <div
            key={f.key}
            className="h-full border-r border-white/60 last:border-r-0 bg-gray-800"
            style={{ width: `${100 / facets.length}%`, opacity: 0.3 + 0.7 * f.level }}
            title={`${f.label}: ${Math.round(f.level * 100)}%`}
          />
        ))}
      </div>
      <span className="text-xs font-semibold text-gray-700">{label}</span>
    </div>
  );
}

export default function MicroSummaryBar(props: MicroSummaryBarProps) {
  const {
    decision,
    confidence,
    riskLabel,
    riskFacets,
    governingThought,
  } = props;

  return (
    <div
      className="sticky top-0 z-40 border-b border-gray-200 bg-white/95 backdrop-blur-md"
      role="banner"
    >
      <div className="mx-auto flex w-full max-w-[1400px] items-center gap-4 px-4 py-3">
        <DecisionChip decision={decision} />

        <div className="hidden items-center gap-2 sm:flex">
          <span className="text-xs font-medium text-gray-700">Confidence</span>
          <ConfidenceBar value={confidence} width={160} />
        </div>

        <div className="hidden items-center gap-2 sm:flex">
          <span className="text-xs font-medium text-gray-700">Risk</span>
          <RiskBand label={riskLabel} facets={riskFacets} width={140} />
        </div>

        <div className="mx-2 hidden h-6 w-px bg-gray-200 sm:block" />

        <div className="min-w-0 flex-1 overflow-hidden">
          <div
            className="line-clamp-2 break-words text-sm leading-relaxed text-gray-700"
            title={governingThought}
          >
            {governingThought}
          </div>
        </div>

        {/* Buttons removed per user request - keep micro-summary always expanded */}
      </div>
    </div>
  );
}
