'use client';

import { use, useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAnalysisStatus } from '@/lib/api/hooks';
import StageProgress from '@/components/StageProgress';
import { PermanentNav } from '@/components/PermanentNav';
import { Button } from '@/components/ui/Button';
import { StatusBadge } from '@/components/micro/StatusBadge';
import { CognitiveSpinner } from '@/components/micro/CognitiveSpinner';
import { PageContainer } from '@/components/layout/PageContainer';

const PIPELINE_STAGES = [
  { name: 'Socratic Questions', key: 'socratic' },
  { name: 'Problem Structuring', key: 'structuring' },
  { name: 'Oracle Research', key: 'oracle' },
  { name: 'Consultant Selection', key: 'selection' },
  { name: 'Parallel Analysis', key: 'analysis' },
  { name: 'Devils Advocate', key: 'devils_advocate' },
  { name: 'Senior Advisor', key: 'senior_advisor' },
  { name: 'Final Report', key: 'report' },
];

const EDUCATIONAL_CONTENT = [
  "The Lollapalooza Effect: When 4-5 consultant perspectives converge, results aren't additive—they're exponential.",
  "Charlie Munger: 'Really big effects come only from large combinations of factors.' That's exactly what's happening now.",
  'Cognitive Diversity Index: We target >0.4 orthogonality to ensure truly independent perspectives.',
  'High orthogonality = better decisions. We\'re measuring consultant perspective independence in real-time.',
  'Inversion Thinking: Your consultants are asking "What could go wrong?" to stress-test recommendations.',
  'First Principles Analysis: Breaking down your question to fundamental truths, then rebuilding.',
  'You\'re the CEO. AI consultants handle analysis; you make final decisions with full transparency.',
  'Balaji Srinivasan: "AI lowers the cost of being your own CEO." You\'re experiencing this right now.',
  'Unlike ChatGPT, you\'ll see every consultant\'s reasoning, all evidence sources, and quality scores.',
  'Complete audit trail: You can trace every recommendation back to its source evidence.',
  'Ethan Mollick (Wharton): AI-augmented consultants produce 40% higher quality results.',
  'McKinsey deployed 12,000 AI agents. BCG saw 25% faster analysis. Now it\'s your turn.',
  'Built from 200+ authoritative sources, not internet scraping.',
  'Sources: Kahneman, Munger, Dalio, McKinsey/BCG/Bain, systems thinking literature.',
];

const STAGE_INSIGHTS = {
  socratic: 'Generating strategic questions to clarify your objectives and constraints.',
  structuring: 'Building MECE framework to ensure comprehensive coverage.',
  oracle: 'Researching latest market data and competitive intelligence.',
  selection: 'Matching consultant expertise to your specific question.',
  analysis: 'Running parallel analyses with different mental models.',
  devils_advocate: 'Testing assumptions and challenging groupthink.',
  senior_advisor: 'Synthesizing insights into executive-ready recommendations.',
  report: 'Compiling glass-box report with complete transparency.',
};

const NWAY_RELATIONS = [
  { name: 'Game Theory Payoffs', status: 'complete' },
  { name: 'Nash Equilibrium', status: 'complete' },
  { name: "Prisoner's Dilemma", status: 'active' },
  { name: 'Incentive Design', status: 'pending' },
  { name: 'Principal-Agent Analysis', status: 'pending' },
];

function formatStatus(value: string) {
  return value
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
    .join(' ');
}

export default function ProcessingPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const { data: status, isLoading } = useAnalysisStatus(id);
  const [currentContentIndex, setCurrentContentIndex] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Rotate educational content every 12 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentContentIndex((prev) => (prev + 1) % EDUCATIONAL_CONTENT.length);
    }, 12000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const s = (status?.status ?? '').toString().toLowerCase();
    if (s === 'failed') {
      setError('Analysis failed. Please try starting a new analysis.');
    } else {
      setError(null);
    }
  }, [status]);

  // Check if completed - redirect to report
  useEffect(() => {
    const s = (status?.status ?? '').toString().toLowerCase();
    if (s === 'completed') {
      router.push(`/analysis/${id}/report_v2`);
    }
  }, [status, id, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-cream-bg">
        <PermanentNav />
        <main className="py-20">
          <PageContainer className="flex min-h-[60vh] max-w-5xl items-center justify-center">
            <CognitiveSpinner label="Synchronizing your analysis" size="lg" />
          </PageContainer>
        </main>
      </div>
    );
  }

  const progressRaw = status?.progress_percentage ?? 0;
  const boundedProgress = Math.min(Math.max(progressRaw, 0), 100);
  const currentStageIndex = Math.min(
    PIPELINE_STAGES.length - 1,
    Math.floor((boundedProgress / 100) * PIPELINE_STAGES.length)
  );
  const currentStage = PIPELINE_STAGES[currentStageIndex];
  const nextStage = PIPELINE_STAGES[currentStageIndex + 1];
  const stageInsight =
    STAGE_INSIGHTS[currentStage?.key as keyof typeof STAGE_INSIGHTS] ??
    'Advancing your analysis with the full consultant bench.';
  const formattedStatus = formatStatus(status?.status ?? 'RUNNING');
  const statusValue = (status?.status ?? '').toString().toLowerCase();
  const statusVariant: 'success' | 'energy' =
    statusValue === 'failed' || statusValue === 'paused_for_user_input' || statusValue === 'paused'
      ? 'energy'
      : 'success';
  const educationalBlurb = EDUCATIONAL_CONTENT[currentContentIndex];
  const isPaused = statusValue === 'paused_for_user_input' || statusValue === 'paused';

  return (
    <div className="min-h-screen bg-cream-bg">
      <PermanentNav />
      <main className="py-10">
        <PageContainer className="max-w-6xl space-y-10">
          {error && (
            <section className="rounded-3xl border border-brand-persimmon bg-brand-persimmon/10 p-6 shadow-sm">
              <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                <div className="space-y-2">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-brand-persimmon font-semibold">
                    Analysis paused
                  </div>
                  <h2 className="text-lg font-semibold text-warm-black">We hit a snag resuming your run</h2>
                  <p className="text-sm leading-relaxed text-text-body">{error}</p>
                </div>
                <Button href="/analyze" variant="accent" size="sm">
                  Launch new analysis
                </Button>
              </div>
            </section>
          )}

          <section className="grid gap-6 rounded-3xl border border-border-default bg-white p-8 shadow-sm lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
            <div className="space-y-6">
              <span className="inline-flex items-center gap-2 rounded-full border border-brand-lime/30 bg-brand-lime/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-warm-black">
                Cognitive orchestration
              </span>
              <div className="space-y-3">
                <h1 className="text-3xl font-semibold leading-tight text-warm-black">
                  {currentStage?.name ?? 'Analysis in motion'}
                </h1>
                <p className="text-sm leading-relaxed text-text-body">{stageInsight}</p>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <StatusBadge variant={statusVariant} size="sm">
                  {formattedStatus}
                </StatusBadge>
                <StatusBadge variant="info" size="sm">
                  {Math.round(boundedProgress)}% complete
                </StatusBadge>
                <StatusBadge variant="pending" size="sm">
                  {nextStage ? `Next: ${nextStage.name}` : 'Compiling final report'}
                </StatusBadge>
              </div>
              <div>
                <div className="text-[11px] uppercase tracking-[0.2em] text-text-label font-semibold">
                  Overall progress
                </div>
                <div className="mt-2 h-2 w-full rounded-full bg-neutral-200">
                  <div
                    className="h-full rounded-full bg-brand-lime transition-all duration-500"
                    style={{ width: `${boundedProgress}%` }}
                  />
                </div>
                <div className="mt-2 flex justify-between text-xs text-text-label">
                  <span>{Math.round(boundedProgress)}% complete</span>
                  <span>{nextStage ? `Up next: ${nextStage.name}` : 'Report delivery'}</span>
                </div>
              </div>
            </div>
            <div className="space-y-4 rounded-3xl border border-border-default bg-cream-bg/70 p-6 shadow-inner">
              <div className="text-[11px] uppercase tracking-[0.2em] text-text-label font-semibold">
                Cognitive flywheel
              </div>
              <div className="space-y-3">
                {NWAY_RELATIONS.map((relation) => {
                  const variant: 'success' | 'info' | 'pending' =
                    relation.status === 'complete'
                      ? 'success'
                      : relation.status === 'active'
                      ? 'info'
                      : 'pending';
                  return (
                    <div
                      key={relation.name}
                      className="flex items-center justify-between rounded-2xl border border-border-default bg-white px-4 py-3 text-sm font-medium text-warm-black shadow-sm"
                    >
                      <span>{relation.name}</span>
                      <StatusBadge variant={variant} size="sm">
                        {formatStatus(relation.status)}
                      </StatusBadge>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-border-default bg-white p-6 shadow-sm">
            <StageProgress
              stages={PIPELINE_STAGES.map((stage) => stage.name)}
              currentStage={currentStageIndex}
              className="stage-progress-animated"
            />
          </section>

          {isPaused && (
            <section className="rounded-3xl border border-brand-persimmon bg-brand-persimmon/10 p-6 shadow-sm">
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div className="space-y-2">
                  <div className="text-[11px] uppercase tracking-[0.2em] text-brand-persimmon font-semibold">
                    We need your perspective
                  </div>
                  <p className="text-sm leading-relaxed text-text-body">
                    Consultants paused the analysis to gather more context. Answer the strategic questions to unlock the next stage.
                  </p>
                </div>
                <Button
                  href={`/analysis/${id}/questions`}
                  variant="accent"
                  size="md"
                  iconPosition="right"
                >
                  Answer strategic questions
                </Button>
              </div>
            </section>
          )}

          <section className="grid gap-6 md:grid-cols-2">
            <div className="rounded-3xl border border-border-default bg-white p-6 shadow-sm">
              <div className="text-[11px] uppercase tracking-[0.2em] text-text-label font-semibold">
                What’s happening right now
              </div>
              <p className="mt-3 text-lg font-semibold leading-snug text-warm-black">
                {stageInsight}
              </p>
              <ul className="mt-4 space-y-2 text-sm leading-relaxed text-text-body">
                <li>• Consultants are aligning on mental models and evidence trails.</li>
                <li>• Confidence scores update once this stage completes.</li>
                <li>• You’ll receive proactive pings if assumptions need your review.</li>
              </ul>
            </div>
            <div className="rounded-3xl border border-border-default bg-white p-6 shadow-sm">
              <div className="text-[11px] uppercase tracking-[0.2em] text-text-label font-semibold">
                Inside the lab
              </div>
              <p className="mt-3 text-sm leading-relaxed text-text-body">{educationalBlurb}</p>
              <div className="mt-4 rounded-2xl border border-cream-bg bg-cream-bg/70 p-4 text-xs text-text-label">
                Rotates every 12 seconds — the principles guiding this stage of the run.
              </div>
            </div>
          </section>
        </PageContainer>
      </main>
    </div>
  );
}
