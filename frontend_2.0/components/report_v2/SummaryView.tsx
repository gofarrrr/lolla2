"use client";

import React from 'react';
import { AlertCircle, CheckCircle2, TrendingUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface SummaryViewProps {
  data: any;
}

export function SummaryView({ data }: SummaryViewProps) {
  const summary = data?.executive_summary || data?.report?.executive_summary || '';
  const quality = data?.quality_metrics || {};
  const recommendations = data?.strategic_recommendations || data?.report?.strategic_recommendations || [];

  const overallConfidence = quality?.overall_confidence || 0;
  const cognitiveDiv = quality?.cognitive_diversity || 0;
  const evidenceStrength = quality?.evidence_strength || 0;
  const execTime = quality?.execution_time_ms || 0;

  // Show full summary (no truncation)
  const preview = summary || ''

  return (
    <div className="space-y-6">
      {/* Header Card */}
      <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
        <div className="flex items-start justify-between mb-6">
          <div className="flex-1">
            <h1 className="text-2xl font-bold text-warm-black mb-2">
              Strategic Decision Analysis
            </h1>
            <p className="text-sm text-text-body max-w-2xl">
              {summary ? (
                <span className="block break-words whitespace-pre-wrap">{preview}</span>
              ) : (
                'Comprehensive strategic analysis completed with actionable recommendations.'
              )}
            </p>
          </div>
          <div className="flex flex-col items-end gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs uppercase tracking-wider text-text-label font-semibold">
                Confidence
              </span>
              <div className="text-2xl font-bold text-warm-black">
                {Math.round(overallConfidence * 100)}%
              </div>
            </div>
          </div>
        </div>

        {/* Quality Metrics Bar */}
        <div className="grid grid-cols-4 gap-4">
          <MetricCard
            label="Overall Confidence"
            value={Math.round(overallConfidence * 100)}
            unit="%"
            icon={<CheckCircle2 className="text-text-label" size={18} />}
          />
          <MetricCard
            label="Cognitive Diversity"
            value={Math.round(cognitiveDiv * 100)}
            unit="%"
            icon={<TrendingUp className="text-text-label" size={18} />}
          />
          <MetricCard
            label="Evidence Strength"
            value={Math.round(evidenceStrength * 100)}
            unit="%"
            icon={<CheckCircle2 className="text-text-label" size={18} />}
          />
          <MetricCard
            label="Processing Time"
            value={Math.round(execTime / 1000)}
            unit="s"
            icon={<AlertCircle className="text-text-label" size={18} />}
          />
        </div>
      </section>

      {/* Executive Summary */}
      {summary && summary.length > 100 && (
        <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
          <h2 className="text-lg font-semibold text-warm-black mb-4">Executive Summary</h2>
          <div className="prose prose-sm max-w-none text-text-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {summary}
            </ReactMarkdown>
          </div>
        </section>
      )}

      {/* Key Recommendations */}
      {recommendations.length > 0 && (
        <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
          <h2 className="text-lg font-semibold text-warm-black mb-4">
            Strategic Recommendations
          </h2>
          <div className="space-y-4">
            {recommendations.slice(0, 3).map((rec: any, index: number) => {
              const recText = typeof rec === 'string' ? rec : rec?.recommendation || rec?.title || '';
              const priority = rec?.priority || 'MEDIUM';
              const confidence = rec?.confidence_level || rec?.confidence || 0;

              return (
                <div
                  key={index}
                  className="border-l-2 border-accent-green rounded-r-lg p-4 bg-white"
                >
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="text-sm font-semibold text-warm-black flex items-center gap-2">
                      <span className="text-ink-3">{index + 1}.</span>
                      {recText.length > 100 ? `${recText.substring(0, 100)}...` : recText}
                    </h3>
                    <span
                      className={`text-xs px-2 py-0.5 rounded font-medium border ${
                        priority === 'HIGH' || priority === 'CRITICAL'
                          ? 'bg-white border-accent-orange text-ink-1'
                          : priority === 'MEDIUM' || priority === 'IMPORTANT'
                          ? 'bg-white border-accent-yellow text-ink-1'
                          : 'bg-white border-border-default text-text-label'
                      }`}
                    >
                      {priority}
                    </span>
                  </div>
                  {rec?.rationale && (
                    <p className="text-xs text-text-body mt-2 line-clamp-2">
                      {rec.rationale}
                    </p>
                  )}
                  {confidence > 0 && (
                    <div className="mt-2 text-xs text-text-label">
                      Confidence: {Math.round((typeof confidence === 'number' ? confidence : parseFloat(confidence) || 0) * 100)}%
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Empty State */}
      {recommendations.length === 0 && (
        <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
          <div className="text-center py-8">
            <AlertCircle className="mx-auto text-text-label mb-3" size={32} />
            <h3 className="text-lg font-semibold text-warm-black mb-2">
              Analysis In Progress
            </h3>
            <p className="text-sm text-text-body max-w-md mx-auto">
              Strategic recommendations are being generated. Check back shortly to see the final analysis.
            </p>
          </div>
        </section>
      )}
    </div>
  );
}

function MetricCard({ label, value, unit, icon }: { label: string; value: number; unit: string; icon: React.ReactNode }) {
  return (
    <div className="bg-cream-bg/30 rounded-xl border border-border-default p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] uppercase tracking-[0.12em] text-text-label font-bold">
          {label}
        </span>
        {icon}
      </div>
      <div className="text-2xl font-bold text-warm-black">
        {value}
        <span className="text-sm font-normal text-text-label ml-1">{unit}</span>
      </div>
    </div>
  );
}
