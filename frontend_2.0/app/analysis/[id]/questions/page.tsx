'use client';

import { use, useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import clsx from 'clsx';
import { api } from '@/lib/api/client';
import { PermanentNav } from '@/components/PermanentNav';
import StageProgress from '@/components/StageProgress';
import { Button } from '@/components/ui/Button';
import { StatusBadge } from '@/components/micro/StatusBadge';
import { CognitiveSpinner } from '@/components/micro/CognitiveSpinner';
import { PageContainer } from '@/components/layout/PageContainer';

const BATCH_SIZE = 3;
const UNLOCK_THRESHOLD = 2;
const PIPELINE_STAGES = [
  'Socratic Questions',
  'Problem Structuring',
  'Oracle Research',
  'Consultant Selection',
  'Parallel Analysis',
  "Devil's Advocate",
  'Senior Advisor',
  'Final Report',
];

export default function QuestionsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [questions, setQuestions] = useState<any[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [researchRequests, setResearchRequests] = useState<Record<string, boolean>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [visibleQuestions, setVisibleQuestions] = useState(BATCH_SIZE);

  useEffect(() => {
    const fetchQuestions = async () => {
      try {
        const data = await api.getQuestions(id);
        const questionsData = data.questions || [];
        setQuestions(questionsData);

        // Initialize answers and researchRequests state for all questions
        const initialAnswers: Record<string, string> = {};
        const initialResearchRequests: Record<string, boolean> = {};
        questionsData.forEach((q: any) => {
          const questionId = q.id || q.question;
          initialAnswers[questionId] = '';
          initialResearchRequests[questionId] = false;
        });
        setAnswers(initialAnswers);
        setResearchRequests(initialResearchRequests);
      } catch (error) {
        console.error('Failed to fetch questions:', error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchQuestions();
  }, [id]);

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      // Combine answered and research questions into single array
      // For research questions, use placeholder answer
      const allAnswers = [
        ...Object.entries(answers)
          .filter(([_, answer]) => answer.trim())
          .map(([questionId, answer]) => ({
            question_id: questionId,
            answer,
          })),
        ...Object.entries(researchRequests)
          .filter(([_, requested]) => requested)
          .map(([questionId]) => ({
            question_id: questionId,
            answer: '[REQUEST_RESEARCH]', // Backend will recognize this
          })),
      ];

      await api.submitAnswers(id, allAnswers);
      router.push(`/analysis/${id}`);
    } catch (error) {
      console.error('Failed to resume analysis:', error);
      alert('Failed to submit answers. Please try again.');
      setIsSubmitting(false);
    }
  };

  const totalCompleted = Object.values(answers).filter(a => a.trim()).length +
                        Object.values(researchRequests).filter(r => r).length;
  const answeredCount = Object.values(answers).filter(a => a.trim()).length;
  const researchCount = Object.values(researchRequests).filter(r => r).length;

  // Unlock more questions as user progresses
  // Formula: For every UNLOCK_THRESHOLD completions, unlock BATCH_SIZE more questions
  useEffect(() => {
    // Calculate minimum required visible questions based on total completions
    const batchesUnlocked = Math.floor(totalCompleted / UNLOCK_THRESHOLD);
    const minRequired = batchesUnlocked * BATCH_SIZE + BATCH_SIZE;
    const newVisible = Math.min(minRequired, questions.length);

    if (newVisible > visibleQuestions) {
      setVisibleQuestions(newVisible);
    }
  }, [totalCompleted, visibleQuestions, questions.length]);

  const displayedQuestions = questions.slice(0, visibleQuestions);
  const hasMoreQuestions = visibleQuestions < questions.length;

  const handleSkip = async () => {
    if (!confirm('⚠️ Skipping questions means less context for analysis.\n\nMore answers = better analysis quality.\n\nAre you sure you want to skip?')) {
      return;
    }
    setIsSubmitting(true);
    try {
      await api.submitAnswers(id, []); // Empty answers array
      router.push(`/analysis/${id}`);
    } catch (error) {
      console.error('Failed to resume:', error);
      setIsSubmitting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-cream-bg">
        <PermanentNav />
        <main className="py-20">
          <PageContainer className="flex min-h-[60vh] max-w-5xl items-center justify-center">
            <CognitiveSpinner label="Preparing strategic questions" size="lg" />
          </PageContainer>
        </main>
      </div>
    );
  }

  const remainingToUnlock = Math.max(0, UNLOCK_THRESHOLD - (totalCompleted % BATCH_SIZE));
  const pendingQuestions = questions.length - visibleQuestions;

  return (
    <div className="min-h-screen bg-cream-bg">
      <PermanentNav />
      <main className="py-10">
        <PageContainer className="max-w-6xl space-y-10">
          <section className="grid gap-6 rounded-3xl border border-border-default bg-white p-8 shadow-sm lg:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
            <div className="space-y-5">
              <span className="inline-flex items-center gap-2 rounded-full border border-brand-lime/30 bg-brand-lime/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-warm-black">
                Stage 01 • Socratic Questions
              </span>
              <h1 className="text-3xl font-semibold leading-tight text-warm-black">
                Unlock sharper context for your consultant bench
              </h1>
              <p className="text-sm leading-relaxed text-text-body">
                Every answer refines the MECE map, the oracle’s research prompts, and the senior advisor’s final narrative.
                Capture the constraints, hidden pressures, and success criteria your team already knows by heart.
              </p>
              <div className="flex flex-wrap gap-3 text-sm">
                <span className="rounded-full border border-brand-lime/40 bg-brand-lime/10 px-4 py-2 font-medium text-warm-black">
                  {answeredCount} answered
                </span>
                <span className="rounded-full border border-brand-persimmon/35 bg-brand-persimmon/10 px-4 py-2 font-medium text-warm-black">
                  {researchCount} marked for research
                </span>
                <span className="rounded-full border border-border-default bg-cream-bg px-4 py-2 font-medium text-text-body">
                  {questions.length} total prompts
                </span>
              </div>
            </div>
            <div className="space-y-4 rounded-3xl border border-border-default bg-cream-bg/80 p-6 shadow-inner">
              <div>
                <div className="text-[11px] uppercase tracking-[0.2em] text-text-label font-semibold">
                  What moves the needle
                </div>
                <ul className="mt-3 space-y-2 text-sm leading-relaxed text-text-body">
                  <li className="flex gap-2">
                    <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-brand-lime" />
                    Share the stakeholder pressures, non-negotiables, and existing commitments.
                  </li>
                  <li className="flex gap-2">
                    <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-brand-lime" />
                    Flag unknowns for research when data is missing or would take hours to compile.
                  </li>
                  <li className="flex gap-2">
                    <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-brand-lime" />
                    Call out risks you’re already tracking so devil’s advocates can stress-test them.
                  </li>
                </ul>
              </div>
              <div className="rounded-2xl border border-brand-persimmon/30 bg-white px-5 py-4 text-sm text-text-body shadow-sm">
                <div className="text-[11px] uppercase tracking-[0.2em] text-brand-persimmon font-semibold">
                  Hint
                </div>
                <p className="mt-2 leading-relaxed">
                  Mark at least {UNLOCK_THRESHOLD} prompts and the next batch of questions unlocks automatically. More context unlocks richer consultant pairings.
                </p>
              </div>
            </div>
          </section>

          <section className="rounded-3xl border border-border-default bg-white p-6 shadow-sm">
            <StageProgress stages={PIPELINE_STAGES} currentStage={0} />
          </section>

          <section className="rounded-3xl border border-brand-lime/35 border-dashed bg-brand-lime/5 p-6 text-sm leading-relaxed text-warm-black shadow-sm">
            <p className="flex items-baseline gap-2">
              <span className="text-lg">✨</span>
              Capture ambitions, constraints, and deal-breakers in your own words. We translate that signal into consultant briefs, oracle queries, and devil’s advocate checks.
            </p>
          </section>

          <section className="space-y-6">
            {displayedQuestions.map((question) => {
              const questionId = question.id || question.question;
              const answer = answers[questionId] || '';
              const markedForResearch = researchRequests[questionId] || false;

              return (
                <article
                  key={questionId}
                  className="rounded-3xl border border-border-default bg-white p-6 shadow-sm"
                >
                  <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                    <header>
                      <div className="text-[11px] uppercase tracking-[0.2em] text-text-label font-semibold">
                        Consultant question
                      </div>
                      <h2 className="mt-3 text-lg font-semibold leading-snug text-warm-black">
                        {question.question}
                      </h2>
                    </header>
                    <StatusBadge
                      variant={
                        markedForResearch
                          ? 'energy'
                          : answer.trim()
                          ? 'success'
                          : 'pending'
                      }
                      size="sm"
                    >
                      {markedForResearch
                        ? 'Research requested'
                        : answer.trim()
                        ? 'Answer captured'
                        : 'Awaiting input'}
                    </StatusBadge>
                  </div>

                  <p className="mt-3 text-sm leading-relaxed text-text-body">
                    {question.context ||
                      'Add the context you already know so consultants can skip discovery work.'}
                  </p>

                  <div className="mt-5 space-y-3">
                    <textarea
                      value={answer}
                      onChange={(event) =>
                        setAnswers((prev) => ({
                          ...prev,
                          [questionId]: event.target.value,
                        }))
                      }
                      placeholder="Share the constraints, non-negotiables, or insider context the team already has."
                      className={clsx(
                        'w-full rounded-2xl border border-border-default px-4 py-3 text-sm leading-relaxed transition focus:outline-none focus:ring-2 focus:ring-bright-green focus:border-transparent',
                        markedForResearch && 'border-brand-persimmon/60 bg-brand-persimmon/10'
                      )}
                      rows={markedForResearch ? 3 : 5}
                    />

                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <label className="inline-flex items-center gap-2 text-xs font-semibold text-text-body">
                        <input
                          type="checkbox"
                          checked={markedForResearch}
                          onChange={(event) =>
                            setResearchRequests((prev) => ({
                              ...prev,
                              [questionId]: event.target.checked,
                            }))
                          }
                          className="h-4 w-4 rounded border border-border-default"
                        />
                        <span className="rounded-full border border-brand-persimmon/30 bg-brand-persimmon/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-brand-persimmon">
                          Flag for research
                        </span>
                      </label>

                      <div className="text-xs text-text-label">
                        {answer.trim().length} characters — mark for research if this requires fresh data pulls.
                      </div>
                    </div>
                  </div>
                </article>
              );
            })}
          </section>

          {hasMoreQuestions && (
            <section className="rounded-3xl border border-dashed border-border-default bg-white p-6 text-sm text-text-body shadow-sm">
              <div className="flex flex-col items-start gap-2 md:flex-row md:items-center md:justify-between">
                <p>
                  <span className="font-semibold text-warm-black">
                    {remainingToUnlock} more question{remainingToUnlock === 1 ? '' : 's'}
                  </span>{' '}
                  to unlock the next batch.
                </p>
                <p className="text-xs text-text-label">
                  {pendingQuestions} question{pendingQuestions === 1 ? '' : 's'} waiting in buffer • Answer or mark for research to unlock.
                </p>
              </div>
            </section>
          )}
        </PageContainer>
      </main>

      <div className="sticky bottom-0 left-0 right-0 border-t border-border-default bg-white/95 shadow-[0_-8px_24px_rgba(26,26,26,0.08)] backdrop-blur">
        <PageContainer className="max-w-6xl flex flex-col gap-3 py-6 md:flex-row md:items-center">
          <Button
            onClick={handleSubmit}
            disabled={totalCompleted === 0 || isSubmitting}
            variant="primary"
            size="lg"
            fullWidth
            type="button"
            className="md:flex-1"
          >
            {isSubmitting
              ? 'Submitting…'
              : `Continue analysis (${answeredCount} answered, ${researchCount} to research)`}
          </Button>
          <Button
            onClick={handleSkip}
            disabled={isSubmitting}
            variant="secondary"
            size="lg"
            fullWidth
            type="button"
            className="md:w-auto"
          >
            Skip all
          </Button>
        </PageContainer>
        <PageContainer className="max-w-6xl">
          <p className="pb-6 text-center text-xs text-text-label">
            More context = stronger recommendations. Share what you know so the experts can push further.
          </p>
        </PageContainer>
      </div>
    </div>
  );
}
