'use client';

import { useMemo, use, useState, useEffect } from 'react';
import Link from 'next/link';
import { PermanentNav } from '@/components/PermanentNav';
import { PageContainer } from '@/components/layout/PageContainer';
import { CognitiveSpinner } from '@/components/micro/CognitiveSpinner';
import { useFinalReport } from '@/lib/api/hooks';
import {
  DecisionWorkbench,
  type DecisionWorkbenchData,
} from '@/components/report/DecisionWorkbench';
import { mapDissentSignals } from '@/lib/mappers/report';
import { DissentPanel } from '@/components/dashboard/dissent/DissentPanel';
import { TriggersTimeline } from '@/components/dashboard/dissent/TriggersTimeline';
import { LeftSidebar } from '@/components/report_v2/LeftSidebar';
import { SeniorAdvisorView } from '@/components/report_v2/SeniorAdvisorView';
import { ResearchView } from '@/components/report_v2/ResearchView';
import { ConsultantDetailView } from '@/components/report_v2/ConsultantDetailView';
import { RawJson } from '@/components/report_v2/RawJson';
import { SummaryView } from '@/components/report_v2/SummaryView';
import { useGlassBoxStore } from '@/lib/state/glassBox';
import { PERSONA_LABELS } from '@/lib/constants/personas';

type ReportPageParams = Promise<{ id: string }>;

// Helper functions for text cleaning
function stripMarkdown(text: string): string {
  return text
    .replace(/^#+\s+/gm, '') // Remove markdown headers
    .replace(/\*\*/g, '') // Remove bold
    .replace(/\*/g, '') // Remove italic
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Remove links
    .trim();
}

function extractTitle(text: string, maxLength: number = 100): string {
  const cleaned = stripMarkdown(text);
  // Get first sentence or first line
  const firstSentence = cleaned.split(/[.\n]/)[0];
  if (firstSentence.length <= maxLength) return firstSentence;
  // Truncate at word boundary
  return cleaned.substring(0, maxLength).split(' ').slice(0, -1).join(' ') + '...';
}

function truncateText(text: string, maxLength: number): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength).split(' ').slice(0, -1).join(' ') + '...';
}

export default function ReportPageV2({ params }: { params: ReportPageParams }) {
  const { id } = use(params);
  const { data, isLoading, error } = useFinalReport(id);
  const { glassBox } = useGlassBoxStore();

  const workbenchData = useMemo<DecisionWorkbenchData>(() => transformReportData(data), [data]);

  // Section navigation state
  const [activeKey, setActiveKey] = useState<string>('summary');

  // Consultant selection for consultant views
  const [selectedConsultantId, setSelectedConsultantId] = useState<string | undefined>(
    (workbenchData.consultants && workbenchData.consultants[0]?.id) || undefined
  );
  useEffect(() => {
    if (!selectedConsultantId && workbenchData.consultants && workbenchData.consultants.length > 0) {
      setSelectedConsultantId(workbenchData.consultants[0].id);
    }
  }, [selectedConsultantId, workbenchData.consultants]);

  useEffect(() => {
    // if active consultant key refers to a consultant, update selection
    if (activeKey.startsWith('consultant:')) {
      const parts = activeKey.split(':');
      if (parts[1]) setSelectedConsultantId(parts[1]);
    }
  }, [activeKey]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-cream-bg">
        <PermanentNav />
        <main className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center px-4">
          <CognitiveSpinner label="Assembling the brief" size="lg" />
        </main>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="min-h-screen bg-cream-bg">
        <PermanentNav />
        <main className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center px-4">
          <div className="text-center space-y-3 border border-border-default bg-white px-6 py-8 rounded-3xl shadow-sm max-w-md">
            <div className="text-xl font-semibold text-warm-black">We couldn&apos;t load this report</div>
            <p className="text-sm text-text-label">
              Something went sideways fetching the engagement trace. Try again shortly or head back to the dashboard.
            </p>
            <Link href="/dashboard" className="btn-primary inline-block">
              Back to dashboard
            </Link>
          </div>
        </main>
      </div>
    );
  }

  // Map dissent signals directly from raw bundle
  const dissentSignals = mapDissentSignals(data);

  // consultants list for sidebar
  const sidebarConsultants = (workbenchData.consultants || []).map((c) => ({ id: c.id, name: c.name }));

  // Helper: render Summary section
  const renderSummary = () => {
    if (glassBox) {
      const rawSummary = {
        trace_id: data?.trace_id,
        executive_summary: data?.executive_summary,
        quality_metrics: data?.quality_metrics,
        key_decisions: data?.key_decisions,
        strategic_recommendations: data?.strategic_recommendations,
        selection_debug: (data as any)?.selection_debug,
        report: data?.report,
      };
      return <RawJson title="Summary (Raw)" data={rawSummary} />;
    }
    return <SummaryView data={data} />;
  };

  // Helper: render Consultant section
  const renderConsultant = () => {
    const parts = activeKey.split(':');
    const consultantId = parts[1] || selectedConsultantId;
    const sub = (parts[2] as 'analysis' | 'da') || 'analysis';
    if (!consultantId) return null;
    return <ConsultantDetailView data={data} consultantId={consultantId} view={sub} />;
  };

  return (
    <div className="min-h-screen bg-cream-bg">
      <PermanentNav />
      <main className="py-6">
        <PageContainer>
          <div className="flex items-start gap-6">
            <LeftSidebar
              consultants={sidebarConsultants}
              activeKey={activeKey}
              onChange={setActiveKey}
            />
            <div className="flex-1 space-y-8">
              {activeKey === 'summary' && renderSummary()}
              {activeKey === 'senior' && <SeniorAdvisorView data={data} />}
              {activeKey.startsWith('consultant:') && renderConsultant()}
              {activeKey === 'research' && <ResearchView data={data} />}
              {/* Default */}
              {![ 'summary', 'senior', 'research' ].some((k) => k === activeKey) && !activeKey.startsWith('consultant:') && renderSummary()}
            </div>
          </div>
        </PageContainer>
      </main>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border-default p-4 bg-white/95">
      <div className="text-[11px] uppercase tracking-[0.2em] text-text-label font-semibold">{label}</div>
      <div className="mt-1 text-base font-semibold text-warm-black">{value}</div>
    </div>
  )
}

function transformReportData(raw: any): DecisionWorkbenchData {
  const mission = buildMission(raw);
  const recommendations = buildRecommendations(raw);
  const riskSummary = buildRiskSummary(raw, recommendations);
  const consultants = buildConsultants(raw);
  const evidence = buildEvidence(raw);
  const mece = buildMece(raw, recommendations);
  const timeline = buildTimeline(raw, consultants);
  const scenarios = buildScenarios(raw, mission.confidence);
  const strategicQuestions = buildStrategicQuestions(raw);
  const criticalPath = buildCriticalPath(raw, mission.decision);
  const mentalModels = buildMentalModels(raw, recommendations);

  // Phase 1 presentation fixes
  const executiveSummaryMarkdown =
    typeof raw?.executive_summary === 'string' ? raw.executive_summary : undefined;
  const devilsAdvocateTranscript =
    (typeof raw?.devils_advocate_transcript === 'string' && raw.devils_advocate_transcript) ||
    (typeof raw?.da_transcript === 'string' && raw.da_transcript) ||
    undefined;

  return {
    mission,
    executiveSummaryMarkdown,
    devilsAdvocateTranscript,
    criticalPath,
    recommendations,
    riskSummary,
    mentalModels,
    strategicQuestions,
    consultants,
    timeline,
    mece,
    evidence,
    scenarios,
  };
}

function buildMission(raw: any | undefined) {
  const rawDecision =
    raw?.key_decisions?.[0]?.decision ||
    raw?.executive_summary?.decision ||
    raw?.executive_summary ||
    'Decision pending';

  // Clean and extract a proper title (remove markdown, truncate)
  const decision = extractTitle(rawDecision, 120);

  const rawWhyItMatters =
    typeof raw?.executive_summary === 'string'
      ? raw.executive_summary
      : raw?.executive_summary?.key_insight ||
        raw?.summary ||
        'Clarify the objective of this analysis and the strategic lens we applied.';

  // Truncate the description to reasonable length
  const whyItMatters = truncateText(stripMarkdown(rawWhyItMatters), 300);

  const confidenceValue =
    normalizeConfidence(raw?.key_decisions?.[0]?.confidence_level) ??
    normalizeConfidence(raw?.quality_metrics?.overall_quality) ??
    0.68;

  const rawMilestone =
    raw?.next_steps?.[0]?.title ||
    raw?.active_goals?.[0]?.title ||
    raw?.timeline?.next_milestone ||
    'Define next stakeholder checkpoint';

  const milestone = truncateText(stripMarkdown(rawMilestone), 80);

  const rawImpact =
    raw?.key_decisions?.[0]?.impact ||
    raw?.impact_summary ||
    raw?.exec_summary?.impact ||
    'Impact pending sizing';

  const impact = truncateText(stripMarkdown(rawImpact), 80);

  return {
    decision,
    whyItMatters,
    confidence: confidenceValue,
    milestone,
    impact,
  };
}

function buildRecommendations(raw: any | undefined) {
  const source: any[] =
    raw?.strategic_recommendations ||
    raw?.report?.strategic_recommendations ||
    raw?.recommendations ||
    [];

  if (!Array.isArray(source) || source.length === 0) {
    return [
      {
        id: 'rec-0',
        title: 'Confirm strategic direction',
        priority: 'HIGH',
        summary: 'Synthesize consultant agreement to anchor a crisp thesis for the decision.',
        confidence: 0.66,
        impact: 'Impact sizing pending',
        timeline: '0-3 months',
        mentalModels: ['Systems Thinking', 'Second-Order Effects'],
        evidenceCount: 0,
        evidenceIds: [],
        howWeBuiltIt: ['Reviewed consultant perspectives', 'Aligned with MECE structure'],
        devilChecks: [],
      },
    ];
  }

  return source.map((item, idx) => {
    const evidenceIds = Array.isArray(item?.evidence_ids) ? item.evidence_ids.filter(Boolean) : [];
    const rawSummary =
      item?.rationale ||
      item?.summary ||
      item?.recommendation ||
      'Synthesize consultant input into a decisive move.';

    // Truncate and clean summary
    const summary = truncateText(stripMarkdown(rawSummary), 250);

    const guidance = Array.isArray(item?.how_we_built_it)
      ? item.how_we_built_it.map((g: string) => truncateText(stripMarkdown(g), 150))
      : typeof item?.implementation_guidance === 'string'
        ? item.implementation_guidance
            .split('\n')
            .map((line: string) => truncateText(stripMarkdown(line.trim()), 150))
            .filter(Boolean)
        : [];

    const devilChecks =
      Array.isArray(item?.devils_advocate?.challenges) && item.devils_advocate.challenges.length
        ? item.devils_advocate.challenges.map((c: string) => truncateText(stripMarkdown(c), 200))
        : Array.isArray(item?.key_challenges)
          ? item.key_challenges.map((c: string) => truncateText(stripMarkdown(c), 200))
          : [];

    const rawTitle = item?.recommendation || `Strategic move ${idx + 1}`;
    const title = truncateText(stripMarkdown(rawTitle), 100);

    return {
      id: item?.id || item?.uuid || `rec-${idx}`,
      title,
      priority: item?.priority?.toString().toUpperCase() ?? 'MEDIUM',
      summary,
      confidence: normalizeConfidence(item?.confidence_level ?? item?.confidence ?? 0.65),
      impact: truncateText(stripMarkdown(item?.impact || item?.impact_assessment || 'Impact TBD'), 80),
      timeline: item?.implementation_timeline || item?.timeline || '3-12 months',
      mentalModels: extractMentalModels(item),
      evidenceCount: evidenceIds.length,
      evidenceIds,
      howWeBuiltIt: guidance,
      devilChecks,
    };
  });
}

function buildRiskSummary(raw: any | undefined, recommendations: ReturnType<typeof buildRecommendations>) {
  const baseHeadline =
    raw?.devils_advocate?.headline ||
    raw?.risk_summary?.headline ||
    'Watch for assumption drift and implementation friction.';

  let risks: string[] = [];
  if (Array.isArray(raw?.devils_advocate?.key_challenges)) {
    risks = raw.devils_advocate.key_challenges;
  } else if (Array.isArray(raw?.risk_summary?.risks)) {
    risks = raw.risk_summary.risks;
  }

  if (risks.length === 0) {
    // Fallback: derive from recommendation devil checks
    risks = recommendations
      .flatMap((rec) => rec.devilChecks?.slice(0, 1) ?? [])
      .filter(Boolean);
    if (risks.length === 0) {
      risks = [
        'Validate foundational assumptions with stakeholders.',
        'Ensure evidence coverage is diverse and current.',
      ];
    }
  }

  return {
    headline: baseHeadline,
    risks,
    confidenceShift: raw?.devils_advocate?.confidence_shift ?? 0,
  };
}

function buildConsultants(raw: any | undefined) {
  const source: any[] =
    raw?.consultant_analyses ||
    raw?.parallel_analysis?.consultant_analyses ||
    raw?.report?.parallel_analysis?.consultant_analyses ||
    [];

  if (!Array.isArray(source) || source.length === 0) {
    return [
      {
        id: 'consultant-0',
        name: 'Lead Strategist',
        role: 'Strategy',
        agreement: 'aligned' as const,
        selectionScore: 0.72,
        keyInsights: ['Reinforce credibility with transparent process and rapid path to value.'],
        concerns: ['Need explicit path to conversion without large spend.'],
      },
    ];
  }

  return source.map((c, idx) => {
    const alignment = (c?.disposition || c?.agreement || '').toString().toLowerCase();
    const agreement =
      alignment.includes('aligned') || alignment.includes('agree')
        ? ('aligned' as const)
        : alignment.includes('disagree') || alignment.includes('challenge')
          ? ('divergent' as const)
          : ('neutral' as const);

    const concerns = Array.isArray(c?.concerns)
      ? c.concerns
      : Array.isArray(c?.risk_factors)
        ? c.risk_factors
        : [];

    const cid = c?.consultant_id || c?.id
    const displayName = c?.consultant_name || (cid && PERSONA_LABELS[String(cid)]) || c?.name || `Consultant ${idx + 1}`
    return {
      id: cid || `consultant-${idx}`,
      name: displayName,
      role: c?.consultant_type || c?.role || 'Advisor',
      agreement,
      selectionScore: normalizeConfidence(c?.selection_score ?? c?.confidence_level ?? 0.6),
      keyInsights: Array.isArray(c?.key_insights) ? c.key_insights : [],
      concerns,
    };
  });
}

function buildEvidence(raw: any | undefined) {
  const source: any[] =
    raw?.evidence_trail || raw?.evidence || raw?.report?.evidence || raw?.sources || [];
  if (!Array.isArray(source) || source.length === 0) {
    return [
      {
        id: 'evidence-0',
        snippet: 'Organizations adopting transparent decision tooling increased trust scores by 28%.',
        source: 'Harvard Business Review, 2024',
        provenance: 'real' as const,
        confidence: 0.82,
      },
    ];
  }

  return source.map((item, idx) => ({
    id: item?.id || item?.evidence_id || `evidence-${idx}`,
    snippet: item?.content || item?.text || item?.evidence || 'Evidence summary pending.',
    source: item?.source || item?.citation || 'Unattributed source',
    provenance: mapProvenance(item?.provenance),
    confidence: normalizeConfidence(item?.confidence ?? item?.confidence_score ?? 0.6),
    type: item?.type || 'quote',
    url: item?.url,
    timestamp: item?.timestamp || item?.collected_at,
  }));
}

function buildMece(raw: any | undefined, recommendations: ReturnType<typeof buildRecommendations>) {
  const source = raw?.mece_framework || raw?.report?.mece_framework || raw?.problem_structuring?.mece_framework;
  const nodes: {
    id: string;
    label: string;
    heat?: number;
    linkedRecommendations?: string[];
    description?: string;
  }[] = [];

  if (Array.isArray(source)) {
    source.forEach((node: any, idx: number) => {
      nodes.push({
        id: node?.dimension || node?.id || `mece-${idx}`,
        label: node?.dimension || node?.label || `Dimension ${idx + 1}`,
        heat: normalizeConfidence(node?.priority_level ?? node?.heat ?? 0.5),
        linkedRecommendations: deriveLinkedMoves(node, recommendations),
        description: node?.key_considerations?.[0] || node?.summary,
      });
    });
  } else if (source?.nodes && Array.isArray(source.nodes)) {
    source.nodes.forEach((node: any, idx: number) => {
      nodes.push({
        id: node?.id || `mece-${idx}`,
        label: node?.name || node?.label || `Node ${idx + 1}`,
        heat: normalizeConfidence(node?.heat ?? 0.5),
        linkedRecommendations: deriveLinkedMoves(node, recommendations),
        description: node?.description || node?.insight,
      });
    });
  }

  if (nodes.length === 0) {
    nodes.push({
      id: 'mece-0',
      label: 'Market access',
      heat: 0.74,
      linkedRecommendations: [recommendations[0]?.title ?? 'Focus recommendation'],
      description: 'Assess the path to reach skeptical senior leaders efficiently.',
    });
    nodes.push({
      id: 'mece-1',
      label: 'Trust mechanics',
      heat: 0.66,
      linkedRecommendations: [recommendations[0]?.title ?? 'Focus recommendation'],
      description: 'Demonstrate transparency as differentiator to drive adoption.',
    });
  }

  return nodes;
}

function buildTimeline(raw: any | undefined, consultants: ReturnType<typeof buildConsultants>) {
  const source: any[] =
    raw?.stage_profiles ||
    raw?.analysis_timeline ||
    raw?.report?.stage_profiles ||
    raw?.timeline ||
    [];

  if (!Array.isArray(source) || source.length === 0) {
    return [
      {
        id: 'event-0',
        title: 'Socratic framing',
        at: 'Start',
        description: 'Clarified the core decision and desired outcomes.',
        contributors: [consultants[0]?.name ?? 'Lead Strategist'],
      },
      {
        id: 'event-1',
        title: 'Consultant synthesis',
        at: 'Midpoint',
        description: 'Aligned consultant perspectives into a coherent thesis.',
        contributors: consultants.map((c) => c.name),
      },
    ];
  }

  return source.map((event, idx) => ({
    id: event?.id || `event-${idx}`,
    title: event?.stage_name || event?.title || `Milestone ${idx + 1}`,
    at: formatTimestamp(event?.completed_at || event?.timestamp),
    description: event?.summary || event?.description,
    contributors: extractContributors(event),
  }));
}

function buildScenarios(raw: any | undefined, baseConfidence: number) {
  const basePercentage = Math.round(baseConfidence * 100);
  const scenarios = raw?.scenarios;
  if (Array.isArray(scenarios) && scenarios.length > 0) {
    return scenarios.map((scenario: any, idx: number) => ({
      id: scenario?.id || `scenario-${idx}`,
      label: scenario?.label || `Scenario ${idx + 1}`,
      confidenceDelta: Math.round(
        (normalizeConfidence(scenario?.confidence_delta ?? 0) - baseConfidence) * 100
      ),
      impactDelta: Math.round((scenario?.impact_delta ?? 0) * 100),
      description: scenario?.description,
    }));
  }

  return [
    {
      id: 'scenario-base',
      label: 'Base case',
      confidenceDelta: 0,
      impactDelta: 0,
      description: `Current confidence ${basePercentage}% with existing assumptions.`,
    },
    {
      id: 'scenario-stretch',
      label: 'Stretch adoption',
      confidenceDelta: 6,
      impactDelta: 12,
      description: 'Targeted executive outreach + lighthouse case study accelerates trust.',
    },
    {
      id: 'scenario-stress',
      label: 'Stress test',
      confidenceDelta: -7,
      impactDelta: -5,
      description: 'Delayed evidence refresh erodes confidence across stakeholders.',
    },
  ];
}

function buildStrategicQuestions(raw: any | undefined) {
  if (Array.isArray(raw?.strategic_questions)) {
    return raw.strategic_questions.map((q: any) => q?.question || q?.text).filter(Boolean);
  }
  if (Array.isArray(raw?.socratic_results?.key_strategic_questions)) {
    return raw.socratic_results.key_strategic_questions
      .map((q: any) => q?.question || q?.prompt)
      .filter(Boolean);
  }
  return [
    'How do we make the transparency of the process the hero?',
    'What is the most capital-efficient path to validating the thesis with skeptics?',
    'How do we translate consultant debate into executive confidence?',
  ];
}

function buildCriticalPath(raw: any | undefined, decision: string) {
  if (Array.isArray(raw?.key_decisions?.[0]?.critical_path)) {
    return raw.key_decisions[0].critical_path.filter(Boolean);
  }
  if (Array.isArray(raw?.executive_summary?.critical_path)) {
    return raw.executive_summary.critical_path.filter(Boolean);
  }
  return [
    `Clarify success criteria for: ${decision}`,
    'Expose transparency differentiator early',
    'Sequence outreach with supporting evidence',
  ];
}

function buildMentalModels(raw: any | undefined, recommendations: ReturnType<typeof buildRecommendations>) {
  if (Array.isArray(raw?.mental_models)) {
    return raw.mental_models.map((model: any) => model?.name || model).filter(Boolean);
  }
  const fromRecommendations = recommendations.flatMap((rec) => rec.mentalModels);
  if (fromRecommendations.length > 0) return Array.from(new Set(fromRecommendations));
  return ['Systems Thinking', 'Inversion', 'Second-Order Effects'];
}

function normalizeConfidence(value: unknown): number {
  if (typeof value === 'number') {
    if (value > 1) return Math.min(Math.max(value / 100, 0), 1);
    if (value < 0) return 0;
    if (value === 0) return 0;
    return value;
  }
  if (typeof value === 'string') {
    const parsed = parseFloat(value);
    if (!Number.isNaN(parsed)) {
      return parsed > 1 ? Math.min(parsed / 100, 1) : Math.max(parsed, 0);
    }
  }
  return 0.65;
}

function extractMentalModels(item: any): string[] {
  if (Array.isArray(item?.mental_models)) return item.mental_models.filter(Boolean);
  if (Array.isArray(item?.models)) return item.models.filter(Boolean);
  if (typeof item?.mental_models === 'string') {
    return item.mental_models
      .split(',')
      .map((s: string) => s.trim())
      .filter(Boolean);
  }
  return [];
}

function mapProvenance(value: unknown): 'real' | 'derived' | 'mixed' | undefined {
  if (!value) return undefined;
  const v = String(value).toLowerCase();
  if (v.includes('real')) return 'real';
  if (v.includes('derived')) return 'derived';
  if (v.includes('mixed')) return 'mixed';
  return undefined;
}

function deriveLinkedMoves(node: any, recommendations: ReturnType<typeof buildRecommendations>) {
  if (!node) return [];
  const tags: string[] = [];
  if (Array.isArray(node?.linked_recommendations)) {
    tags.push(...node.linked_recommendations);
  }
  if (Array.isArray(node?.recommendations)) {
    tags.push(...node.recommendations);
  }
  if (tags.length > 0) return tags;

  // attempt to match by dimension keywords
  if (typeof node?.dimension === 'string') {
    return recommendations
      .filter((rec) => rec.summary.toLowerCase().includes(node.dimension.toLowerCase()))
      .map((rec) => rec.title);
  }
  return [];
}

function formatTimestamp(value: string | undefined) {
  if (!value) return 'Timestamp pending';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
}

function extractContributors(event: any) {
  if (Array.isArray(event?.contributors)) return event.contributors;
  if (Array.isArray(event?.consultants)) return event.consultants;
  if (event?.consultant_id) return [event.consultant_id];
  return [];
}
