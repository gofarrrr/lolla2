import { useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type Recommendation = {
  id: string;
  title: string;
  priority?: string;
  summary: string;
  confidence: number;
  impact?: string;
  timeline?: string;
  mentalModels: string[];
  evidenceCount: number;
  evidenceIds: string[];
  howWeBuiltIt: string[];
  devilChecks: string[];
};

type Consultant = {
  id: string;
  name: string;
  role?: string;
  agreement?: 'aligned' | 'divergent' | 'neutral';
  selectionScore?: number;
  keyInsights: string[];
  concerns: string[];
};

type TimelineEvent = {
  id: string;
  title: string;
  at: string;
  description?: string;
  contributors?: string[];
};

type MeceNode = {
  id: string;
  label: string;
  heat?: number;
  linkedRecommendations?: string[];
  description?: string;
};

type EvidenceItem = {
  id: string;
  snippet: string;
  source: string;
  provenance?: 'real' | 'derived' | 'mixed';
  confidence?: number;
  type?: string;
  url?: string;
  timestamp?: string;
};

type Scenario = {
  id: string;
  label: string;
  confidenceDelta: number;
  impactDelta: number;
  description?: string;
};

export type DecisionWorkbenchData = {
  mission: {
    decision: string;
    whyItMatters: string;
    confidence: number;
    milestone: string;
    impact: string;
  };
  // Raw markdown for executive summary (Phase 1 presentation fix)
  executiveSummaryMarkdown?: string;
  // Full Devil's Advocate transcript
  devilsAdvocateTranscript?: string;
  criticalPath: string[];
  recommendations: Recommendation[];
  riskSummary: {
    headline: string;
    risks: string[];
    confidenceShift?: number;
  };
  mentalModels: string[];
  strategicQuestions: string[];
  consultants: Consultant[];
  timeline: TimelineEvent[];
  mece: MeceNode[];
  evidence: EvidenceItem[];
  scenarios: Scenario[];
};

type ContextTarget =
  | { type: 'mission' }
  | { type: 'recommendation'; recommendationId: string }
  | { type: 'risk' }
  | { type: 'consultant'; consultantId: string }
  | { type: 'timeline'; eventId: string }
  | { type: 'mece'; nodeId: string };

type PaneView = 'trace' | 'debate' | 'evidence' | 'consultants';

const palette = {
  base: 'bg-cream-bg',
  card: 'bg-white',
  cream: 'bg-cream-bg',
  border: 'border-border-default',
  accent: '#68DE7C',
  accentMuted: '#5ACC6E',
  text: '#1A1A1A',
  textMuted: '#666666',
};

function Chip({
  label,
  active,
  onClick,
}: {
  label: string;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        'px-3 py-1 rounded-full text-xs font-medium transition border',
        active
          ? 'bg-card-active-bg text-warm-black border-bright-green shadow-sm'
          : 'bg-white text-text-label border-border-default hover:bg-gray-50'
      )}
    >
      {label}
    </button>
  );
}

function Tag({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full border border-border-default bg-white px-3 py-1 text-[11px] font-medium text-text-label">
      <span className="h-1.5 w-1.5 rounded-full bg-bright-green" />
      {label}
    </span>
  );
}

function MetricCard({
  title,
  value,
  accent,
  footprint,
  onActivate,
  active,
}: {
  title: string;
  value: string;
  accent?: string;
  footprint?: string;
  onActivate?: () => void;
  active?: boolean;
}) {
  return (
    <button
      onClick={onActivate}
      className={clsx(
        'w-full rounded-2xl border px-3 py-2 text-left transition-all duration-300',
        'bg-white',
        'border-border-default shadow-sm hover:shadow-md hover:-translate-y-0.5',
        active && 'ring-2 ring-bright-green shadow-lg border-bright-green'
      )}
    >
      <div className="text-[10px] uppercase tracking-[0.1em] text-text-label font-semibold">
        {title}
      </div>
      <div className="mt-0.5 text-base font-semibold text-warm-black truncate">{value}</div>
      {accent && <div className="mt-0.5 text-[11px] text-bright-green truncate">{accent}</div>}
      {footprint && <div className="mt-1 text-[10px] text-text-label truncate">{footprint}</div>}
    </button>
  );
}

function SectionHeader({
  eyebrow,
  title,
  subtitle,
  actions,
}: {
  eyebrow: string;
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div>
        <div className="text-[11px] uppercase tracking-wider text-text-label font-semibold">
          {eyebrow}
        </div>
        <div className="text-lg font-semibold text-warm-black">{title}</div>
        {subtitle && <div className="text-sm text-text-body mt-1">{subtitle}</div>}
      </div>
      {actions}
    </div>
  );
}

type ViewState = 'overview' | 'focused' | 'immersive';

export function DecisionWorkbench({ data }: { data: DecisionWorkbenchData }) {
  const [mode, setMode] = useState<'brief' | 'lab'>('brief');
  const [context, setContext] = useState<ContextTarget>({ type: 'mission' });
  const [paneView, setPaneView] = useState<PaneView>('trace');
  const [scenarioOpen, setScenarioOpen] = useState(false);
  const [scenarioId, setScenarioId] = useState<string>(data.scenarios[0]?.id ?? '');
  const [viewState, setViewState] = useState<ViewState>('overview');
  const [headerCollapsed, setHeaderCollapsed] = useState(false);

  const activeRecommendation = useMemo(() => {
    if (context.type !== 'recommendation') return undefined;
    return data.recommendations.find((rec) => rec.id === context.recommendationId);
  }, [context, data.recommendations]);

  const activeConsultant = useMemo(() => {
    if (context.type !== 'consultant') return undefined;
    return data.consultants.find((c) => c.id === context.consultantId);
  }, [context, data.consultants]);

  const activeTimeline = useMemo(() => {
    if (context.type !== 'timeline') return undefined;
    return data.timeline.find((evt) => evt.id === context.eventId);
  }, [context, data.timeline]);

  const activeMeceNode = useMemo(() => {
    if (context.type !== 'mece') return undefined;
    return data.mece.find((node) => node.id === context.nodeId);
  }, [context, data.mece]);

  const scenario = useMemo(
    () => data.scenarios.find((s) => s.id === scenarioId),
    [data.scenarios, scenarioId]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // ESC: Reset to overview / collapse header
      if (e.key === 'Escape') {
        if (context.type !== 'mission') {
          setContext({ type: 'mission' });
        } else {
          setHeaderCollapsed(false);
        }
      }

      // CMD+\ or CTRL+\: Toggle header collapse
      if ((e.metaKey || e.ctrlKey) && e.key === '\\') {
        e.preventDefault();
        setHeaderCollapsed(!headerCollapsed);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [context, headerCollapsed]);

  return (
    <div className={clsx('min-h-screen', palette.base, 'text-warm-black')}>
      <div className="border-b border-border-default bg-cream-bg/80 backdrop-blur-md sticky top-0 z-10 transition-all duration-300">
        <div className={clsx(
          'mx-auto flex max-w-7xl flex-col gap-4 px-6 md:flex-row md:items-start md:justify-between overflow-hidden transition-all duration-300',
          headerCollapsed ? 'py-2' : 'py-4'
        )}>
          <div className={clsx('space-y-1 flex-1 min-w-0 transition-all duration-300', headerCollapsed && 'opacity-0 max-h-0')}>
            <div className="text-[10px] uppercase tracking-[0.2em] text-text-label font-semibold">
              Strategic Decision
            </div>
            <h1 className="text-xl md:text-2xl font-semibold text-warm-black leading-tight line-clamp-2">
              {data.mission.decision}
            </h1>
            <p className="max-w-2xl text-sm text-text-body leading-relaxed line-clamp-3">{data.mission.whyItMatters}</p>
          </div>
          <div className={clsx(
            'grid w-full md:w-auto md:min-w-[400px] gap-2 text-sm shrink-0 transition-all duration-300',
            headerCollapsed ? 'grid-cols-4' : 'grid-cols-2'
          )}>
            <MetricCard
              title="Confidence"
              value={`${Math.round(data.mission.confidence * 100)}%`}
              onActivate={() => setContext({ type: 'mission' })}
            />
            <MetricCard
              title="Next Milestone"
              value={data.mission.milestone}
              onActivate={() => setContext({ type: 'mission' })}
            />
            <MetricCard
              title="Impact Signal"
              value={data.mission.impact}
              onActivate={() => setContext({ type: 'mission' })}
            />
            <MetricCard
              title="Active Scenario"
              value={scenario?.label ?? 'Base Case'}
              accent={
                scenario
                  ? `${scenario.confidenceDelta >= 0 ? '+' : ''}${scenario.confidenceDelta}% confidence`
                  : undefined
              }
              onActivate={() => setScenarioOpen((v) => !v)}
              active={scenarioOpen}
            />
          </div>
          <button
            onClick={() => setHeaderCollapsed(!headerCollapsed)}
            className="absolute right-4 top-2 p-1 text-text-label hover:text-warm-black transition-colors"
            aria-label={headerCollapsed ? 'Expand header' : 'Collapse header'}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              {headerCollapsed ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              )}
            </svg>
          </button>
        </div>
      </div>

      <div className="mx-auto max-w-[1200px] px-8 py-8">
        {/* Mode Toggle */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex gap-1 rounded-full bg-white border border-border-default p-1 shadow-sm">
            <button
              onClick={() => setMode('brief')}
              className={clsx(
                'rounded-full px-6 py-2 text-sm font-medium transition',
                mode === 'brief'
                  ? 'bg-bright-green text-white shadow'
                  : 'text-text-body hover:text-warm-black'
              )}
            >
              One-Pager Report
            </button>
            <button
              onClick={() => setMode('lab')}
              className={clsx(
                'rounded-full px-6 py-2 text-sm font-medium transition',
                mode === 'lab'
                  ? 'bg-bright-green text-white shadow'
                  : 'text-text-body hover:text-warm-black'
              )}
            >
              Explore Process
            </button>
          </div>
        </div>

        {/* Single Column Content */}
        {mode === 'brief' ? (
          <OnePagerReport data={data} />
        ) : (
          <ProcessLabPane
            data={data}
            setContext={setContext}
            activeContext={context}
            openPane={(view) => setPaneView(view)}
          />
        )}
      </div>


      {scenarioOpen && (
        <ScenarioSheet
          scenarios={data.scenarios}
          activeId={scenarioId}
          onClose={() => setScenarioOpen(false)}
          onSelect={(id) => setScenarioId(id)}
        />
      )}
    </div>
  );
}

function OnePagerReport({ data }: { data: DecisionWorkbenchData }) {
  const [expandedRec, setExpandedRec] = useState<string | null>(null);

  return (
    <div className="space-y-8">
      {/* Executive Summary */}
      <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
        <div className="flex items-start justify-between mb-4">
          <div>
            <div className="text-xs uppercase tracking-wider text-text-label font-semibold mb-2">
              Executive Summary
            </div>
            <h2 className="text-2xl font-bold text-warm-black mb-3">
              {data.mission.decision}
            </h2>
          </div>
          <div className="flex gap-3 text-sm">
            <div className="text-center">
              <div className="text-xs text-text-label uppercase tracking-wide">Confidence</div>
              <div className="text-lg font-bold text-bright-green">
                {Math.round(data.mission.confidence * 100)}%
              </div>
            </div>
          </div>
        </div>
        <div className="prose prose-lg max-w-none text-text-body">
          {data.executiveSummaryMarkdown ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{data.executiveSummaryMarkdown}</ReactMarkdown>
          ) : (
            <p className="text-base leading-relaxed">{data.mission.whyItMatters}</p>
          )}
        </div>
        <div className="mt-6 flex flex-wrap gap-2">
          {data.mentalModels.slice(0, 5).map((model) => (
            <span
              key={model}
              className="px-3 py-1 bg-cream-bg text-text-body text-sm rounded-full border border-border-default"
            >
              {model}
            </span>
          ))}
        </div>
      </section>

      {/* Strategic Recommendations */}
      <section className="space-y-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl font-bold text-warm-black">Strategic Recommendations</h3>
          <span className="text-sm text-text-label">{data.recommendations.length} recommendations</span>
        </div>

        {data.recommendations.slice(0, 3).map((rec, index) => (
          <div
            key={rec.id}
            className="bg-white rounded-2xl border border-border-default shadow-sm overflow-hidden transition-all duration-300"
          >
            <button
              onClick={() => setExpandedRec(expandedRec === rec.id ? null : rec.id)}
              className="w-full p-6 text-left hover:bg-gray-50 transition"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="flex items-center justify-center w-8 h-8 rounded-full bg-bright-green text-white font-bold text-sm">
                      {index + 1}
                    </span>
                    <h4 className="text-lg font-semibold text-warm-black">{rec.title}</h4>
                  </div>
                  <p className="text-base text-text-body leading-relaxed mt-2">
                    {rec.summary}
                  </p>
                  <div className="flex flex-wrap gap-2 mt-4">
                    <span className="px-3 py-1 bg-blue-50 text-blue-700 text-xs rounded-full">
                      Confidence: {Math.round(rec.confidence * 100)}%
                    </span>
                    <span className="px-3 py-1 bg-green-50 text-green-700 text-xs rounded-full">
                      {rec.evidenceCount} evidence pieces
                    </span>
                    {rec.priority && (
                      <span className="px-3 py-1 bg-orange-50 text-orange-700 text-xs rounded-full font-medium">
                        {rec.priority}
                      </span>
                    )}
                  </div>
                </div>
                <div className="text-text-label">
                  {expandedRec === rec.id ? '▲' : '▼'}
                </div>
              </div>
            </button>

            {/* Expanded Content */}
            {expandedRec === rec.id && (
              <div className="px-6 pb-6 pt-2 border-t border-border-default bg-gray-50 space-y-4 animate-in fade-in slide-in-from-top-2 duration-300">
                {rec.howWeBuiltIt.length > 0 && (
                  <div>
                    <h5 className="text-sm font-semibold text-warm-black mb-2 uppercase tracking-wide">
                      How We Built This
                    </h5>
                    <ul className="space-y-2">
                      {rec.howWeBuiltIt.map((step, idx) => (
                        <li key={idx} className="flex gap-2 text-sm text-text-body">
                          <span className="text-bright-green font-bold">→</span>
                          <span>{step}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {rec.devilChecks.length > 0 && (
                  <div>
                    <h5 className="text-sm font-semibold text-warm-black mb-2 uppercase tracking-wide">
                      Devil&apos;s Advocate Challenges
                    </h5>
                    <ul className="space-y-2">
                      {rec.devilChecks.map((check, idx) => (
                        <li key={idx} className="flex gap-2 text-sm text-text-body">
                          <span className="text-orange-500">⚠</span>
                          <span>{check}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {rec.mentalModels.length > 0 && (
                  <div>
                    <h5 className="text-sm font-semibold text-warm-black mb-2 uppercase tracking-wide">
                      Mental Models Applied
                    </h5>
                    <div className="flex flex-wrap gap-2">
                      {rec.mentalModels.map((model) => (
                        <span
                          key={model}
                          className="px-2 py-1 bg-white text-text-body text-xs rounded border border-border-default"
                        >
                          {model}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </section>

      {/* Risks & Devil&apos;s Advocate */}
      <section className="bg-orange-50 rounded-2xl border border-orange-200 p-8 shadow-sm">
        <div className="flex items-start gap-3 mb-4">
          <span className="text-2xl">⚠️</span>
          <div>
            <h3 className="text-xl font-bold text-warm-black mb-2">Risks & Challenges</h3>
            <p className="text-base text-orange-900 font-medium">{data.riskSummary.headline}</p>
          </div>
        </div>
        <ul className="space-y-3 mt-4">
          {data.riskSummary.risks.map((risk, idx) => (
            <li key={idx} className="flex gap-3 text-sm text-text-body">
              <span className="text-orange-600 font-bold mt-0.5">•</span>
              <span className="leading-relaxed">{risk}</span>
            </li>
          ))}
        </ul>
        {data.riskSummary.confidenceShift && (
          <div className="mt-4 pt-4 border-t border-orange-200">
            <span className="text-sm text-orange-800">
              Confidence adjustment: {data.riskSummary.confidenceShift > 0 ? '+' : ''}
              {data.riskSummary.confidenceShift}%
            </span>
          </div>
        )}
      </section>

      {/* Evidence & Confidence Summary */}
      <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
        <h3 className="text-xl font-bold text-warm-black mb-6">Analysis Foundation</h3>
        <div className="grid grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-bright-green">{data.evidence.length}</div>
            <div className="text-sm text-text-label mt-1">Evidence Pieces</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-bright-green">{data.consultants.length}</div>
            <div className="text-sm text-text-label mt-1">Expert Consultants</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-bright-green">{data.mentalModels.length}</div>
            <div className="text-sm text-text-label mt-1">Mental Models</div>
          </div>
        </div>
      </section>
    </div>
  );
}

function AdvisorBriefPane({
  data,
  setContext,
  activeContext,
}: {
  data: DecisionWorkbenchData;
  setContext: (ctx: ContextTarget) => void;
  activeContext: ContextTarget;
}) {
  return (
    <div className="space-y-5">
      <div className={clsx('rounded-3xl border p-5 md:p-6 shadow-sm', palette.card, palette.border)}>
        <SectionHeader
          eyebrow="Mission Control"
          title="Always-visible core"
          subtitle="Where we stand, what matters, how to act next."
        />
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <MetricCard
            title="Situation"
            value={data.criticalPath[0] ?? 'Clarify primary challenge'}
            footprint="Pulled from critical path analysis."
            onActivate={() => setContext({ type: 'mission' })}
            active={activeContext.type === 'mission'}
          />
          <MetricCard
            title="Strategic Thesis"
            value={
              data.recommendations.length > 0
                ? data.recommendations[0].summary
                : 'Revisit consultant synthesis'
            }
            footprint="Synthesized from consultant alignment."
            onActivate={() =>
              setContext({
                type: 'recommendation',
                recommendationId: data.recommendations[0]?.id ?? '',
              })
            }
            active={
              activeContext.type === 'recommendation' &&
              activeContext.recommendationId === data.recommendations[0]?.id
            }
          />
          <MetricCard
            title="Top Move"
            value={data.recommendations[0]?.title ?? 'Define strategic move'}
            footprint={`${data.recommendations[0]?.evidenceCount ?? 0} evidence anchors • mental models in play`}
            onActivate={() =>
              setContext({
                type: 'recommendation',
                recommendationId: data.recommendations[0]?.id ?? '',
              })
            }
            active={
              activeContext.type === 'recommendation' &&
              activeContext.recommendationId === data.recommendations[0]?.id
            }
          />
          <MetricCard
            title="Risk Pulse"
            value={data.riskSummary.headline}
            footprint={`${data.riskSummary.risks.length} pressure points flagged`}
            onActivate={() => setContext({ type: 'risk' })}
            active={activeContext.type === 'risk'}
          />
        </div>
      </div>

      <div className={clsx('rounded-3xl border p-5 md:p-6 shadow-sm', palette.card, palette.border)}>
        <SectionHeader
          eyebrow="Narrative"
          title="How we frame this decision"
          subtitle="A single spine: Problem → Insight → Recommendation → Next action."
        />

        <div className="mt-5 space-y-4">
          {data.recommendations.map((rec) => {
            const isActive = activeContext.type === 'recommendation' && activeContext.recommendationId === rec.id;
            const isCollapsed = activeContext.type === 'recommendation' && activeContext.recommendationId !== rec.id;

            return (
              <button
                key={rec.id}
                onClick={() => setContext({ type: 'recommendation', recommendationId: rec.id })}
                className={clsx(
                  'w-full rounded-3xl border border-border-default bg-white/95 text-left transition-all duration-300 hover:shadow-md min-w-0',
                  isActive ? 'ring-2 ring-bright-green shadow-lg p-4' : 'p-4',
                  isCollapsed && 'opacity-50 hover:opacity-100'
                )}
              >
                <div className="flex items-start justify-between gap-3 min-w-0">
                  <div className="space-y-1 flex-1 min-w-0">
                    <Tag label="Recommendation" />
                    <div className={clsx(
                      'font-semibold text-warm-black transition-all duration-300',
                      isActive ? 'text-lg' : 'text-base line-clamp-2'
                    )}>{rec.title}</div>
                  </div>
                  <Tag label={rec.priority ?? 'PRIORITIZE'} />
                </div>
                <div className={clsx(
                  'mt-2 text-sm text-text-body transition-all duration-300 overflow-hidden',
                  isActive ? 'max-h-[500px]' : 'line-clamp-3 max-h-[80px]',
                  isCollapsed && 'max-h-0 opacity-0'
                )}>{rec.summary}</div>
                <div className={clsx(
                  'mt-3 flex flex-wrap items-center gap-2 text-[11px] text-text-label min-w-0 transition-all duration-300',
                  isCollapsed && 'max-h-0 opacity-0 overflow-hidden'
                )}>
                  {rec.mentalModels.slice(0, isActive ? 10 : 3).map((model) => (
                    <Tag key={model} label={model} />
                  ))}
                  <Tag label={`Evidence • ${rec.evidenceCount}`} />
                  <Tag label={`Confidence • ${Math.round(rec.confidence * 100)}%`} />
                </div>
                {isActive && (
                  <div className="mt-4 space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
                    {rec.howWeBuiltIt.length > 0 && (
                      <div className="border-t border-border-default pt-3">
                        <div className="text-xs uppercase tracking-wider text-text-label font-semibold mb-2">
                          How We Built It
                        </div>
                        <ul className="space-y-1 text-sm text-text-body">
                          {rec.howWeBuiltIt.map((step, idx) => (
                            <li key={idx} className="flex gap-2">
                              <span className="text-bright-green">•</span>
                              <span>{step}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {rec.devilChecks.length > 0 && (
                      <div className="border-t border-border-default pt-3">
                        <div className="text-xs uppercase tracking-wider text-text-label font-semibold mb-2">
                          Devil&apos;s Advocate Checks
                        </div>
                        <ul className="space-y-1 text-sm text-text-body">
                          {rec.devilChecks.map((check, idx) => (
                            <li key={idx} className="flex gap-2">
                              <span className="text-orange-500">⚠</span>
                              <span>{check}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

interface ProcessLabPaneProps {
  data: DecisionWorkbenchData;
  setContext: (ctx: ContextTarget) => void;
  activeContext: ContextTarget;
  openPane: (view: PaneView) => void;
}

function ProcessLabPane({ data, setContext, activeContext, openPane }: ProcessLabPaneProps) {
  return <div>Process Lab Mode - Under Construction</div>;
}

function ProcessLabPaneOLD({ data, setContext, activeContext, openPane }: ProcessLabPaneProps) {
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  return (
    <div className="space-y-8">
      <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
        <button
          onClick={() => setExpandedSection(expandedSection === 'problem' ? null : 'problem')}
          className="w-full flex items-start justify-between text-left"
        >
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="flex items-center justify-center w-10 h-10 rounded-full bg-blue-100 text-blue-700 font-bold">
                1
              </span>
              <h3 className="text-xl font-bold text-warm-black">Problem Framing & Decomposition</h3>
            </div>
            <p className="text-sm text-text-body ml-13">
              How we structured and understood the decision space
            </p>
          </div>
          <div className="text-text-label text-xl">
            {expandedSection === 'problem' ? '▲' : '▼'}
          </div>
        </button>

        {expandedSection === 'problem' && (
          <div className="mt-6 ml-13 space-y-6 animate-in fade-in slide-in-from-top-2 duration-300">
            {/* Initial Problem Statement */}
            <div>
              <h4 className="text-sm font-semibold text-warm-black mb-2 uppercase tracking-wide">
                Initial Problem Statement
              </h4>
              <div className="bg-gray-50 rounded-xl p-4 border border-border-default">
                <p className="text-base text-text-body leading-relaxed">{data.mission.decision}</p>
              </div>
            </div>

            {/* MECE Breakdown */}
            <div>
              <h4 className="text-sm font-semibold text-warm-black mb-3 uppercase tracking-wide">
                MECE Breakdown
              </h4>
              <div className="grid gap-3 md:grid-cols-2">
                {data.mece.map((node) => (
                  <div
                    key={node.id}
                    className="bg-white rounded-xl p-4 border border-border-default hover:border-bright-green transition cursor-pointer"
                    onClick={() => {
                      setContext({ type: 'mece', nodeId: node.id });
                      openPane('trace');
                    }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <h5 className="font-semibold text-warm-black">{node.label}</h5>
                      <span className="text-xs bg-blue-50 text-blue-700 px-2 py-1 rounded">
                        {Math.round((node.heat ?? 0) * 100)}% focus
                      </span>
                    </div>
                    {node.description && (
                      <p className="text-sm text-text-body">{node.description}</p>
                    )}
                    <div className="mt-2 text-xs text-text-label">
                      {node.linkedRecommendations?.length ?? 0} linked recommendations
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Mental Models */}
            <div>
              <h4 className="text-sm font-semibold text-warm-black mb-3 uppercase tracking-wide">
                Mental Models Applied
              </h4>
              <div className="flex flex-wrap gap-2">
                {data.mentalModels.map((model) => (
                  <span
                    key={model}
                    className="px-4 py-2 bg-green-50 text-green-700 text-sm rounded-full border border-green-200"
                  >
                    {model}
                  </span>
                ))}
              </div>
            </div>

            {/* Strategic Questions */}
            <div>
              <h4 className="text-sm font-semibold text-warm-black mb-3 uppercase tracking-wide">
                Strategic Questions Generated
              </h4>
              <ul className="space-y-2">
                {data.strategicQuestions.map((question, idx) => (
                  <li key={idx} className="flex gap-2 text-sm text-text-body">
                    <span className="text-bright-green font-bold">?</span>
                    <span>{question}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </section>

      <div className={clsx('rounded-3xl border p-5 md:p-6 shadow-sm', palette.card, palette.border)}>
        <SectionHeader
          eyebrow="Problem Structure"
          title="MECE map of the decision space"
          subtitle="Highlight where insights and evidence land across the structure."
          actions={
            <button
              onClick={() => openPane('evidence')}
              className="text-xs font-semibold text-bright-green underline-offset-4 hover:underline"
            >
              View linked evidence
            </button>
          }
        />
        <div className="mt-6 grid gap-3 md:grid-cols-2">
          {data.mece.map((node) => {
            const isActive = activeContext.type === 'mece' && activeContext.nodeId === node.id;
            return (
              <button
                key={node.id}
                onClick={() => {
                  setContext({ type: 'mece', nodeId: node.id });
                  openPane('trace');
                }}
                className={clsx(
                  'rounded-3xl border border-border-default bg-white/95 p-4 text-left transition hover:shadow-md',
                  isActive && 'ring-2 ring-bright-green shadow-lg'
                )}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="space-y-1">
                    <Tag label="MECE dimension" />
                    <div className="text-base font-semibold text-warm-black">{node.label}</div>
                  </div>
                  <Tag label={`${Math.round((node.heat ?? 0) * 100)}% focus`} />
                </div>
                {node.description && (
                  <div className="mt-2 text-sm text-text-body">{node.description}</div>
                )}
                <div className="mt-3 text-[11px] text-text-label">
                  Linked moves: {node.linkedRecommendations?.length ?? 0}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Section 2: Consultant Gallery */}
      <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
        <button
          onClick={() => setExpandedSection(expandedSection === 'consultants' ? null : 'consultants')}
          className="w-full flex items-start justify-between text-left"
        >
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="flex items-center justify-center w-10 h-10 rounded-full bg-green-100 text-green-700 font-bold">
                2
              </span>
              <h3 className="text-xl font-bold text-warm-black">Expert Consultants & Analysis</h3>
            </div>
            <p className="text-sm text-text-body ml-13">
              {data.consultants.length} independent consultants contributed their expertise
            </p>
          </div>
          <div className="text-text-label text-xl">
            {expandedSection === 'consultants' ? '▲' : '▼'}
          </div>
        </button>

        {expandedSection === 'consultants' && (
          <div className="mt-6 ml-13 space-y-4 animate-in fade-in slide-in-from-top-2 duration-300">
            {data.consultants.map((consultant) => (
              <div
                key={consultant.id}
                className="bg-white rounded-xl p-6 border-2 border-border-default hover:border-bright-green transition"
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h4 className="text-lg font-bold text-warm-black mb-1">{consultant.name}</h4>
                    <p className="text-sm text-text-label">{consultant.role ?? 'Strategic Analyst'}</p>
                  </div>
                  <div
                    className={clsx(
                      'rounded-full px-3 py-1 text-xs font-semibold',
                      consultant.agreement === 'aligned'
                        ? 'bg-green-100 text-green-700'
                        : consultant.agreement === 'divergent'
                          ? 'bg-red-100 text-red-700'
                          : 'bg-gray-100 text-gray-700'
                    )}
                  >
                    {consultant.agreement === 'aligned'
                      ? 'Aligned'
                      : consultant.agreement === 'divergent'
                        ? 'Divergent View'
                        : 'Nuanced'}
                  </div>
                </div>

                {/* Key Insights */}
                {consultant.keyInsights.length > 0 && (
                  <div className="mb-4">
                    <h5 className="text-xs font-semibold text-warm-black mb-2 uppercase tracking-wide">
                      Key Insights
                    </h5>
                    <ul className="space-y-2">
                      {consultant.keyInsights.slice(0, 3).map((insight, idx) => (
                        <li key={idx} className="flex gap-2 text-sm text-text-body">
                          <span className="text-bright-green font-bold">•</span>
                          <span>{insight}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Concerns */}
                {consultant.concerns.length > 0 && (
                  <div className="mb-4">
                    <h5 className="text-xs font-semibold text-warm-black mb-2 uppercase tracking-wide">
                      Concerns Raised
                    </h5>
                    <ul className="space-y-2">
                      {consultant.concerns.slice(0, 2).map((concern, idx) => (
                        <li key={idx} className="flex gap-2 text-sm text-orange-700">
                          <span className="font-bold">⚠</span>
                          <span>{concern}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Performance Metrics */}
                <div className="flex gap-4 pt-3 border-t border-border-default">
                  {consultant.selectionScore && (
                    <div className="text-center">
                      <div className="text-xs text-text-label">Selection Score</div>
                      <div className="text-sm font-semibold text-warm-black">
                        {Math.round(consultant.selectionScore * 100)}%
                      </div>
                    </div>
                  )}
                  <div className="text-center">
                    <div className="text-xs text-text-label">Insights</div>
                    <div className="text-sm font-semibold text-warm-black">
                      {consultant.keyInsights.length}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-xs text-text-label">Concerns</div>
                    <div className="text-sm font-semibold text-warm-black">
                      {consultant.concerns.length}
                    </div>
                  </div>
                </div>

                {/* View Full Report Button */}
                <button
                  className="mt-4 w-full py-2 text-sm font-medium text-bright-green hover:text-green-hover transition"
                  onClick={() => {
                    setContext({ type: 'consultant', consultantId: consultant.id });
                    openPane('consultants');
                  }}
                >
                  View Full Consultant Report →
                </button>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function RightPane({
  data,
  context,
  paneView,
  onPaneChange,
  setContext,
}: {
  data: DecisionWorkbenchData;
  context: ContextTarget;
  paneView: PaneView;
  onPaneChange: (view: PaneView) => void;
  setContext: (ctx: ContextTarget) => void;
}) {
  const recommendation =
    context.type === 'recommendation'
      ? data.recommendations.find((rec) => rec.id === context.recommendationId)
      : undefined;
  const consultant =
    context.type === 'consultant'
      ? data.consultants.find((c) => c.id === context.consultantId)
      : undefined;
  const timeline =
    context.type === 'timeline'
      ? data.timeline.find((evt) => evt.id === context.eventId)
      : undefined;
  const meceNode =
    context.type === 'mece' ? data.mece.find((node) => node.id === context.nodeId) : undefined;

  return (
    <div className={clsx('rounded-3xl border shadow-sm', palette.card, palette.border)}>
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border-default px-4 py-3">
        <div className="text-sm font-semibold text-warm-black">
          {context.type === 'mission' && 'Trace: Mission Summary'}
          {context.type === 'recommendation' && `Trace: ${recommendation?.title ?? 'Move'}`}
          {context.type === 'risk' && 'Trace: Risk Pulse'}
          {context.type === 'consultant' && `Trace: ${consultant?.name ?? 'Consultant'}`}
          {context.type === 'timeline' && `Trace: ${timeline?.title ?? 'Timeline'}`}
          {context.type === 'mece' && `Trace: ${meceNode?.label ?? 'Problem slice'}`}
        </div>
        <div className="flex gap-2">
          {(['trace', 'debate', 'evidence', 'consultants'] as PaneView[]).map((view) => (
            <Chip
              key={view}
              label={view.charAt(0).toUpperCase() + view.slice(1)}
              active={paneView === view}
              onClick={() => onPaneChange(view)}
            />
          ))}
        </div>
      </div>

      <div className="space-y-4 px-4 py-5 text-sm text-warm-black">
        {paneView === 'trace' && (
          <>
            {context.type === 'mission' && (
              <div className="space-y-3">
                <p className="leading-relaxed">{data.mission.whyItMatters}</p>
                <div className="rounded-3xl border border-border-default bg-white/90 px-3 py-2 text-xs text-orange-700">
                  Critical path: {data.criticalPath.join(' → ')}
                </div>
                <div className="space-y-2">
                  <Tag label="Strategic questions" />
                  <ul className="space-y-1 text-sm text-text-body">
                    {data.strategicQuestions.slice(0, 4).map((question) => (
                      <li key={question}>• {question}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {recommendation && (
              <RecommendationTrace recommendation={recommendation} data={data} />
            )}

            {context.type === 'risk' && (
              <div className="space-y-3">
                <Tag label="Risk pulse" />
                <div className="rounded-3xl border border-orange-50 bg-orange-50 px-3 py-3 text-orange-700">
                  {data.riskSummary.headline}
                </div>
                <ul className="space-y-2 text-sm text-text-body">
                  {data.riskSummary.risks.map((risk) => (
                    <li key={risk}>• {risk}</li>
                  ))}
                </ul>
              </div>
            )}

            {consultant && <ConsultantTrace consultant={consultant} />}

            {timeline && (
              <div className="space-y-2">
                <Tag label="Event summary" />
                <p className="text-sm text-text-body">{timeline.description}</p>
                <div className="rounded-3xl border border-border-default bg-white/90 px-3 py-2 text-xs text-text-label">
                  Contributors: {timeline.contributors?.join(', ') || '—'}
                </div>
              </div>
            )}

            {meceNode && (
              <div className="space-y-2">
                <Tag label="Node insight" />
                <p className="text-sm text-text-body">{meceNode.description}</p>
                <div className="text-xs text-text-label">
                  Linked recommendations:{' '}
                  {meceNode.linkedRecommendations?.length
                    ? meceNode.linkedRecommendations.join(', ')
                    : '—'}
                </div>
              </div>
            )}
          </>
        )}

        {paneView === 'debate' && (
          <div className="space-y-4">
            <div className="space-y-3">
              <Tag label="Devil&apos;s Advocate Highlights" />
              {recommendation ? (
                <ul className="space-y-2 text-sm text-text-body">
                  {recommendation.devilChecks.length > 0 ? (
                    recommendation.devilChecks.map((item) => <li key={item}>• {item}</li>)
                  ) : (
                    <li>No critical challenges logged for this move.</li>
                  )}
                </ul>
              ) : (
                <p className="text-sm text-text-body">
                  Select a recommendation to view targeted challenges.
                </p>
              )}
            </div>
            {data.devilsAdvocateTranscript && (
              <div className="mt-2">
                <Tag label="Devil&apos;s Advocate Transcript" />
                <div className="mt-2 rounded-2xl border border-border-default bg-white/90 p-3 text-sm text-text-body whitespace-pre-wrap">
                  {data.devilsAdvocateTranscript}
                </div>
              </div>
            )}
          </div>
        )}

        {paneView === 'evidence' && (
          <div className="space-y-3">
            <Tag label="Evidence Trail" />
            <div className="space-y-2">
              {data.evidence.slice(0, 12).map((item) => (
                <div
                  key={item.id}
                  className="rounded-3xl border border-border-default bg-white/90 px-3 py-2 text-sm text-text-body"
                >
                  <div className="text-xs font-semibold text-bright-green">{item.source}</div>
                  <div className="mt-1 text-sm text-warm-black">{item.snippet}</div>
                  <div className="mt-1 text-[11px] text-text-label">
                    {item.provenance ? `${item.provenance.toUpperCase()} SOURCE` : 'SOURCE'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {paneView === 'consultants' && consultant && (
          <ConsultantTrace consultant={consultant} />
        )}

        {paneView === 'consultants' && !consultant && (
          <p className="text-sm text-text-body">Select a consultant to inspect their rationale.</p>
        )}
      </div>
    </div>
  );
}

function RecommendationTrace({
  recommendation,
  data,
}: {
  recommendation: Recommendation;
  data: DecisionWorkbenchData;
}) {
  const linkedEvidence = data.evidence.filter((e) =>
    recommendation.evidenceIds.includes(e.id)
  );

  return (
    <div className="space-y-4">
      <div>
        <Tag label="Why this matters" />
        <p className="mt-1 text-sm text-text-body">{recommendation.summary}</p>
      </div>

      <div>
        <Tag label="How we built it" />
        <ul className="mt-2 space-y-2 text-sm text-warm-black">
          {recommendation.howWeBuiltIt.length > 0 ? (
            recommendation.howWeBuiltIt.map((item) => <li key={item}>• {item}</li>)
          ) : (
            <li>Consultant synthesis and MECE mapping.</li>
          )}
        </ul>
      </div>

      <div>
        <Tag label="Mental models" />
        <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-text-label">
          {recommendation.mentalModels.length > 0 ? (
            recommendation.mentalModels.map((model) => <Tag key={model} label={model} />)
          ) : (
            <span>No mental models tagged.</span>
          )}
        </div>
      </div>

      <div>
        <Tag label="Evidence" />
        <div className="mt-2 space-y-2 text-sm text-text-body">
          {linkedEvidence.length > 0 ? (
            linkedEvidence.map((item) => (
              <div
                key={item.id}
                className="rounded-3xl border border-border-default bg-white/90 px-3 py-2 text-sm text-text-body"
              >
                <div className="text-xs font-semibold text-bright-green">{item.source}</div>
                <div className="mt-1 text-sm text-warm-black">{item.snippet}</div>
              </div>
            ))
          ) : (
            <p>No dedicated evidence captured for this recommendation yet.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function ConsultantTrace({ consultant }: { consultant: Consultant }) {
  return (
    <div className="space-y-4">
      <div>
        <Tag label="Key insights" />
        <ul className="mt-2 space-y-2 text-sm text-warm-black">
          {consultant.keyInsights.length > 0 ? (
            consultant.keyInsights.map((item) => <li key={item}>• {item}</li>)
          ) : (
            <li>No insights captured.</li>
          )}
        </ul>
      </div>
      <div>
        <Tag label="Concerns surfaced" />
        <ul className="mt-2 space-y-2 text-sm text-text-body">
          {consultant.concerns.length > 0 ? (
            consultant.concerns.map((item) => <li key={item}>• {item}</li>)
          ) : (
            <li>No blockers raised.</li>
          )}
        </ul>
      </div>
    </div>
  );
}

function ScenarioSheet({
  scenarios,
  activeId,
  onClose,
  onSelect,
}: {
  scenarios: Scenario[];
  activeId: string;
  onClose: () => void;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="fixed inset-x-0 bottom-0 z-40 border-t border-border-default bg-cream-bg shadow-2xl">
      <div className="mx-auto flex max-w-4xl flex-col gap-4 px-4 py-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xs uppercase tracking-wide text-text-label font-semibold">
              Scenario simulator
            </div>
            <div className="text-base font-semibold text-warm-black">
              How does confidence shift under different assumptions?
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-full border border-border-default px-3 py-1 text-xs text-text-body hover:bg-white"
          >
            Close
          </button>
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          {scenarios.map((scenario) => {
            const active = scenario.id === activeId;
            return (
              <button
                key={scenario.id}
                onClick={() => onSelect(scenario.id)}
                className={clsx(
                  'rounded-3xl border border-border-default bg-white/95 p-4 text-left transition hover:shadow-md',
                  active && 'ring-2 ring-bright-green shadow-lg'
                )}
              >
                <div className="text-sm font-semibold text-warm-black">{scenario.label}</div>
                {scenario.description && (
                  <div className="mt-1 text-xs text-text-body">{scenario.description}</div>
                )}
                <div className="mt-3 text-[11px] text-text-label">
                  Confidence Δ {scenario.confidenceDelta >= 0 ? '+' : ''}
                  {scenario.confidenceDelta}% • Impact Δ {scenario.impactDelta >= 0 ? '+' : ''}
                  {scenario.impactDelta}%
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function OneClickOutputs() {
  const options = [
    { title: 'Board PDF', desc: '6-slide brief auto-generated' },
    { title: 'Exec Email', desc: 'Decision, why, and clear next moves' },
    { title: 'Notion Update', desc: 'Sync outcomes into project hub' },
  ];
  return (
    <div className="rounded-3xl border border-border-default bg-white/95 p-4 shadow-sm">
      <Tag label="One-click outputs" />
      <div className="mt-3 grid gap-3 md:grid-cols-3">
        {options.map((opt) => (
          <div
            key={opt.title}
            className="rounded-3xl border border-border-default bg-gradient-to-br from-white via-white to-gray-50 p-3 shadow-sm"
          >
            <div className="text-sm font-semibold text-warm-black">{opt.title}</div>
            <div className="mt-1 text-xs text-text-body">{opt.desc}</div>
            <div className="mt-3 flex gap-2 text-xs">
              <button className="btn-primary rounded-lg px-3 py-1 shadow-sm">
                Generate
              </button>
              <button className="rounded-lg border border-border-default px-3 py-1 text-text-body hover:bg-white">
                Configure
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StickyRail({ onScenarioToggle }: { onScenarioToggle: () => void }) {
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-0 z-30">
      <div className="pointer-events-auto mx-auto mb-4 flex max-w-4xl items-center gap-3 rounded-full border border-border-default bg-white/85 px-4 py-2 shadow-lg backdrop-blur">
        <button className="btn-primary rounded-full px-4 py-1 text-sm font-semibold shadow-sm">
          Promote to brief
        </button>
        <button className="rounded-full border border-border-default px-4 py-1 text-sm text-text-body">
          Generate outputs
        </button>
        <button className="rounded-full border border-border-default px-4 py-1 text-sm text-text-body">
          Ask a new question
        </button>
        <div className="ml-auto flex items-center gap-2 text-xs text-text-body">
          Active scenario:
          <button
            onClick={onScenarioToggle}
            className="rounded-full border border-border-default px-3 py-1 text-xs text-text-body hover:bg-white"
          >
            Adjust →
          </button>
        </div>
      </div>
    </div>
  );
}
