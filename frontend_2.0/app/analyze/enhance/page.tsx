'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import axios from 'axios';

const BATCH_SIZE = 3;
const UNLOCK_THRESHOLD = 2;

interface Question {
  id: string;
  tier: number;
  tier_name: string;
  question: string;
  reasoning: string;
  impact_score: number;
}

export default function EnhanceQuestionsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get('query') || '';

  const [isLoading, setIsLoading] = useState(true);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [sessionId, setSessionId] = useState<string>('');
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [researchRequests, setResearchRequests] = useState<Record<string, boolean>>({});
  const [visibleQuestions, setVisibleQuestions] = useState(BATCH_SIZE);

  // Load questions from API
  useEffect(() => {
    const loadQuestions = async () => {
      try {
        setIsLoading(true);
        const response = await axios.post('http://localhost:8000/api/progressive-questions/generate', {
          statement: initialQuery || 'Should our SaaS company expand into enterprise?',
          context: {},
        });

        const data = response.data;
        setSessionId(data.engagement_id);

        // Transform backend response to frontend format
        const transformedQuestions: Question[] = [];
        let questionIndex = 0;

        data.levels.forEach((level: any, levelIndex: number) => {
          const tier = levelIndex + 1;
          const tierName = level.id === 'essential' ? 'Essential' :
                          level.id === 'strategic' ? 'Strategic' : 'Expert';

          level.questions.forEach((questionText: string) => {
            transformedQuestions.push({
              id: `q-${questionIndex}`,
              tier,
              tier_name: tierName,
              question: questionText,
              reasoning: level.description,
              impact_score: tier === 1 ? 0.98 : tier === 2 ? 0.94 : 0.90,
            });
            questionIndex++;
          });
        });

        setQuestions(transformedQuestions);

        // Initialize answers with empty strings for all questions to ensure controlled inputs
        const initialAnswers: Record<string, string> = {};
        transformedQuestions.forEach(q => {
          initialAnswers[q.id] = '';
        });
        setAnswers(initialAnswers);

        setIsLoading(false);
      } catch (error) {
        console.error('Failed to load questions:', error);
        setIsLoading(false);
        // Keep mock questions as fallback
      }
    };

    loadQuestions();
  }, [initialQuery]);

  const handleAnswerChange = (questionId: string, value: string) => {
    const newAnswers = { ...answers, [questionId]: value };
    setAnswers(newAnswers);

    if (value.trim().length > 0 && researchRequests[questionId]) {
      setResearchRequests(prev => ({ ...prev, [questionId]: false }));
    }

    checkAndUnlockNextBatch(newAnswers, researchRequests);
  };

  const handleResearchToggle = (questionId: string) => {
    const newResearchRequests = {
      ...researchRequests,
      [questionId]: !researchRequests[questionId]
    };
    setResearchRequests(newResearchRequests);

    if (!researchRequests[questionId]) {
      setAnswers(prev => ({ ...prev, [questionId]: '' }));
    }

    checkAndUnlockNextBatch(answers, newResearchRequests);
  };

  const checkAndUnlockNextBatch = (
    currentAnswers: Record<string, string>,
    currentResearchRequests: Record<string, boolean>
  ) => {
    const currentBatchStartIndex = Math.floor((visibleQuestions - 1) / BATCH_SIZE) * BATCH_SIZE;
    const currentBatchEndIndex = currentBatchStartIndex + BATCH_SIZE;
    const currentBatchQuestions = questions.slice(currentBatchStartIndex, currentBatchEndIndex);

    const currentBatchCompleted = currentBatchQuestions.filter(q =>
      (currentAnswers[q.id] && currentAnswers[q.id].trim().length > 0) ||
      currentResearchRequests[q.id]
    ).length;

    if (currentBatchCompleted >= UNLOCK_THRESHOLD && visibleQuestions < questions.length) {
      setVisibleQuestions(Math.min(visibleQuestions + BATCH_SIZE, questions.length));
    }
  };

  const handleContinue = async () => {
    try {
      // Submit to backend with research requests
      const answeredQuestions = Object.entries(answers)
        .filter(([_, answer]) => answer.trim().length > 0)
        .map(([questionId, answer]) => ({
          question_id: questionId,
          question_text: questions.find(q => q.id === questionId)?.question || '',
          answer,
        }));

      const researchQuestionsList = Object.entries(researchRequests)
        .filter(([_, requested]) => requested)
        .map(([questionId]) => ({
          question_id: questionId,
          question_text: questions.find(q => q.id === questionId)?.question || '',
        }));

      // Start analysis with enhanced context
      const analysisResponse = await axios.post('http://localhost:8000/api/engagements/start', {
        user_query: initialQuery || 'Should our SaaS company expand into enterprise?',
        enhancement_questions_session_id: sessionId,
        answered_questions: answeredQuestions,
        research_questions: researchQuestionsList,
        quality_target: Math.round(qualityScore) / 100,
        interactive_mode: true, // Enable ULTRATHINK pause for user answers
      });

      router.push(`/analysis/${analysisResponse.data.trace_id}`);
    } catch (error) {
      console.error('Failed to start analysis:', error);
      alert('Failed to start analysis. Please try again.');
    }
  };

  const handleSkip = async () => {
    try {
      const analysisResponse = await axios.post('http://localhost:8000/api/engagements/start', {
        user_query: initialQuery || 'Should our SaaS company expand into enterprise?',
      });
      router.push(`/analysis/${analysisResponse.data.trace_id}`);
    } catch (error) {
      console.error('Failed to start analysis:', error);
      alert('Failed to start analysis. Please try again.');
    }
  };

  const answeredCount = Object.keys(answers).filter(id => answers[id]?.trim().length > 0).length;
  const researchCount = Object.keys(researchRequests).filter(id => researchRequests[id]).length;
  const totalCompleted = answeredCount + researchCount;
  const totalQuestions = questions.length;

  // Calculate quality score
  const tier1Answered = questions.slice(0, 5).filter(q => answers[q.id]?.trim()).length;
  const tier1Research = questions.slice(0, 5).filter(q => researchRequests[q.id]).length;
  const tier2Answered = questions.slice(5, 13).filter(q => answers[q.id]?.trim()).length;
  const tier2Research = questions.slice(5, 13).filter(q => researchRequests[q.id]).length;
  const tier3Answered = questions.slice(13).filter(q => answers[q.id]?.trim()).length;
  const tier3Research = questions.slice(13).filter(q => researchRequests[q.id]).length;

  let qualityScore = 50;
  if (tier2Answered > 0 || tier2Research > 0) {
    const tier2Total = questions.slice(5, 13).length || 1;
    qualityScore += ((tier2Answered + tier2Research * 0.7) / tier2Total) * 30;
  }
  if (tier3Answered > 0 || tier3Research > 0) {
    const tier3Total = questions.slice(13).length || 1;
    qualityScore += ((tier3Answered + tier3Research * 0.7) / tier3Total) * 20;
  }

  const displayedQuestions = questions.slice(0, visibleQuestions);
  const hasMoreQuestions = visibleQuestions < questions.length;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="text-2xl font-bold mb-2">Generating Questions...</div>
          <div className="text-gray-600">Creating personalized enhancement questions for your analysis</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b-2 border-black">
        <div className="container-wide py-3">
          <div className="flex justify-between items-center">
            <Link href="/dashboard" className="text-2xl font-bold">
              Lolla
            </Link>
            <div className="text-sm font-medium">
              {totalCompleted} of {totalQuestions} completed
            </div>
          </div>
        </div>
      </header>

      <main className="container-wide py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-3">Enhance Your Analysis</h1>
          <p className="text-lg text-gray-600 mb-2">
            Answer questions to improve analysis quality. Answer 2 questions to unlock the next batch.
          </p>
          <div className="text-sm text-gray-500 mt-3">
            Your question:&nbsp;&ldquo;{initialQuery || 'Should our SaaS company expand into enterprise?'}&rdquo;
          </div>
        </div>

        <div className="mb-8">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-bold">Progress</span>
            <span className="text-sm font-bold">{Math.round(qualityScore)}% Quality</span>
          </div>
          <div className="w-full bg-gray-200 h-3 border-2 border-black">
            <div
              className="bg-accent h-full transition-all duration-500"
              style={{ width: `${(totalCompleted / totalQuestions) * 100}%` }}
            ></div>
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>Baseline: 50%</span>
            <span>Target: 95%+</span>
          </div>
        </div>

        <div className="space-y-4 mb-8">
          {displayedQuestions.map((q, globalIndex) => {
            const isAnswered = answers[q.id]?.trim().length > 0;
            const isResearchRequested = researchRequests[q.id];
            const isCompleted = isAnswered || isResearchRequested;

            return (
              <div
                key={q.id}
                className={`border-2 transition-all duration-300 ${
                  isAnswered
                    ? 'border-accent-green bg-accent-green/5'
                    : isResearchRequested
                    ? 'border-accent-orange bg-white'
                    : 'border-mesh'
                }`}
              >
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <span className="text-sm font-bold text-gray-500">
                          Q{globalIndex + 1}
                        </span>
                        <span className={`text-xs font-bold px-2 py-1 rounded border bg-white ${
                          q.tier === 1 ? 'border-accent-green text-ink-1' :
                          q.tier === 2 ? 'border-accent-yellow text-ink-1' :
                          'border-mesh text-text-label'
                        }`}>
                          TIER {q.tier}: {q.tier_name.toUpperCase()}
                        </span>
                      </div>
                      <h3 className="text-base font-semibold leading-relaxed">
                        {q.question}
                      </h3>
                    </div>
                    {isCompleted && (
                      <div className={`ml-4 font-bold text-sm ${
                        isAnswered ? 'text-accent-green' : 'text-accent-orange'
                      }`}>
                        ✓
                      </div>
                    )}
                  </div>

                  <div className="mb-4 text-sm text-text-body bg-canvas p-3 border-l-2 border-mesh">
                    Why this matters: {q.reasoning}
                  </div>

                  <textarea
                    className="input text-sm min-h-[80px] resize-y w-full"
                    placeholder="Your answer..."
                    value={answers[q.id] || ''}
                    onChange={(e) => handleAnswerChange(q.id, e.target.value)}
                    disabled={isResearchRequested}
                  />

                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <label className="flex items-center gap-2 cursor-pointer text-sm">
                      <input
                        type="checkbox"
                        checked={isResearchRequested}
                        onChange={() => handleResearchToggle(q.id)}
                        className="w-4 h-4 accent-accent-orange"
                      />
                      <span className="font-medium text-text-body">
                        Ask Lolla to research this for me
                      </span>
                      <span className="text-xs text-text-label">
                        (Counts as 70% quality)
                      </span>
                    </label>
                    {isResearchRequested && (
                      <p className="text-xs text-accent-orange mt-1 ml-6">
                        This question will be researched and answered during analysis
                      </p>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {hasMoreQuestions && (
          <div className="text-center py-6 mb-8">
            <div className="border-2 border-dashed border-gray-300 p-6 bg-gray-50">
              <p className="text-sm text-gray-600">
                <span className="font-bold">{Math.max(0, UNLOCK_THRESHOLD - (totalCompleted % BATCH_SIZE))} more question(s)</span> to unlock the next batch
              </p>
              <p className="text-xs text-gray-500 mt-1">
                {questions.length - visibleQuestions} questions remaining • Answer or request research on any question
              </p>
            </div>
          </div>
        )}

        <div className="sticky bottom-0 bg-white border-t-2 border-black py-4 -mx-8 px-8">
          <div className="flex gap-4 items-center">
            <button
              onClick={handleContinue}
              disabled={totalCompleted === 0}
              className="btn btn-accent flex-1"
            >
              Continue {totalCompleted > 0 && `(${answeredCount} answered, ${researchCount} to research)`} — {Math.round(qualityScore)}% Quality
            </button>
            <button
              onClick={handleSkip}
              className="btn btn-outline border-2 border-gray-400 hover:bg-gray-100 text-gray-600"
            >
              Skip All
            </button>
          </div>
          <div className="flex justify-between items-center mt-2 text-xs text-gray-600">
            {totalCompleted === 0 ? (
              <p className="text-center w-full">
                Answer or mark for research at least one question to continue
              </p>
            ) : (
              <>
                <span>{answeredCount} answered • {researchCount} to research</span>
                <span>{Math.round(qualityScore)}% quality</span>
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
