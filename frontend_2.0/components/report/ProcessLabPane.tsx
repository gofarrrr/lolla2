import { useState } from 'react';
import clsx from 'clsx';

type DecisionWorkbenchData = {
  recommendations: Array<{
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
  }>;
  consultants: Array<{
    id: string;
    name: string;
    role: string;
    alignment: number;
    reasoning: string;
  }>;
  evidence: Array<{
    id: string;
    type: string;
    source: string;
    content: string;
    relevance: number;
  }>;
  mentalModels: string[];
  criticalPath: string[];
  riskSummary: {
    headline: string;
    risks: Array<{
      id: string;
      title: string;
      severity: string;
      mitigation: string;
    }>;
  };
};

type ContextTarget =
  | { type: 'mission' }
  | { type: 'recommendation'; recommendationId: string }
  | { type: 'risk' }
  | { type: 'evidence'; evidenceId: string }
  | { type: 'consultant'; consultantId: string };

type PaneView = 'trace' | 'debate' | 'evidence' | 'consultants';

interface ProcessLabPaneProps {
  data: DecisionWorkbenchData;
  setContext: (ctx: ContextTarget) => void;
  activeContext: ContextTarget;
  openPane: (view: PaneView) => void;
}

export function ProcessLabPane({ data, setContext, activeContext, openPane }: ProcessLabPaneProps) {
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
              <h3 className="text-xl font-bold text-warm-black">Problem Framing &amp; Decomposition</h3>
            </div>
            <p className="text-sm text-text-body ml-13">
              How we structured and understood the decision space
            </p>
          </div>
          <span className="text-2xl">{expandedSection === 'problem' ? '−' : '+'}</span>
        </button>
        {expandedSection === 'problem' && (
          <div className="mt-4 pt-4 border-t border-border-default">
            <p className="text-text-body">
              We decomposed your decision into {data.criticalPath.length} key dimensions,
              examining {data.evidence.length} pieces of evidence across {data.mentalModels.length} mental models.
            </p>
          </div>
        )}
      </section>

      <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
        <button
          onClick={() => setExpandedSection(expandedSection === 'synthesis' ? null : 'synthesis')}
          className="w-full flex items-start justify-between text-left"
        >
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="flex items-center justify-center w-10 h-10 rounded-full bg-green-100 text-green-700 font-bold">
                2
              </span>
              <h3 className="text-xl font-bold text-warm-black">Expert Synthesis</h3>
            </div>
            <p className="text-sm text-text-body ml-13">
              {data.consultants.length} expert perspectives analyzed and aligned
            </p>
          </div>
          <span className="text-2xl">{expandedSection === 'synthesis' ? '−' : '+'}</span>
        </button>
        {expandedSection === 'synthesis' && (
          <div className="mt-4 pt-4 border-t border-border-default space-y-3">
            {data.consultants.slice(0, 3).map((consultant) => (
              <div key={consultant.id} className="p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold text-warm-black">{consultant.name}</div>
                  <div className="text-xs text-text-label">{consultant.role}</div>
                </div>
                <p className="text-sm text-text-body">{consultant.reasoning}</p>
                <div className="mt-2 text-xs text-text-label">
                  Alignment: {Math.round(consultant.alignment * 100)}%
                </div>
              </div>
            ))}
            {data.consultants.length > 3 && (
              <button
                onClick={() => openPane('consultants')}
                className="text-sm text-bright-green hover:underline"
              >
                View all {data.consultants.length} consultants →
              </button>
            )}
          </div>
        )}
      </section>

      <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
        <button
          onClick={() => setExpandedSection(expandedSection === 'validation' ? null : 'validation')}
          className="w-full flex items-start justify-between text-left"
        >
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="flex items-center justify-center w-10 h-10 rounded-full bg-orange-100 text-orange-700 font-bold">
                3
              </span>
              <h3 className="text-xl font-bold text-warm-black">Challenge &amp; Validation</h3>
            </div>
            <p className="text-sm text-text-body ml-13">
              Devil&apos;s advocate checks and stress tests applied
            </p>
          </div>
          <span className="text-2xl">{expandedSection === 'validation' ? '−' : '+'}</span>
        </button>
        {expandedSection === 'validation' && (
          <div className="mt-4 pt-4 border-t border-border-default space-y-3">
            {data.recommendations[0]?.devilChecks.slice(0, 3).map((check, idx) => (
              <div key={idx} className="flex gap-3 p-3 bg-orange-50 rounded-lg">
                <span className="text-orange-500 text-lg">⚠</span>
                <p className="text-sm text-text-body flex-1">{check}</p>
              </div>
            ))}
            <button
              onClick={() => openPane('debate')}
              className="text-sm text-bright-green hover:underline"
            >
              View full debate transcript →
            </button>
          </div>
        )}
      </section>

      <section className="bg-white rounded-2xl border border-border-default p-8 shadow-sm">
        <button
          onClick={() => setExpandedSection(expandedSection === 'recommendation' ? null : 'recommendation')}
          className="w-full flex items-start justify-between text-left"
        >
          <div>
            <div className="flex items-center gap-3 mb-2">
              <span className="flex items-center justify-center w-10 h-10 rounded-full bg-purple-100 text-purple-700 font-bold">
                4
              </span>
              <h3 className="text-xl font-bold text-warm-black">Final Recommendation</h3>
            </div>
            <p className="text-sm text-text-body ml-13">
              Synthesized guidance with confidence scores and next steps
            </p>
          </div>
          <span className="text-2xl">{expandedSection === 'recommendation' ? '−' : '+'}</span>
        </button>
        {expandedSection === 'recommendation' && data.recommendations[0] && (
          <div className="mt-4 pt-4 border-t border-border-default">
            <h4 className="font-bold text-warm-black mb-2">{data.recommendations[0].title}</h4>
            <p className="text-text-body mb-4">{data.recommendations[0].summary}</p>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div className="text-text-label mb-1">Confidence</div>
                <div className="font-semibold text-bright-green">
                  {Math.round(data.recommendations[0].confidence * 100)}%
                </div>
              </div>
              <div>
                <div className="text-text-label mb-1">Evidence Support</div>
                <div className="font-semibold text-bright-green">
                  {data.recommendations[0].evidenceCount} pieces
                </div>
              </div>
            </div>
            {data.recommendations[0].howWeBuiltIt.length > 0 && (
              <div className="mt-4 pt-4 border-t border-border-default">
                <div className="text-xs uppercase tracking-wider text-text-label font-semibold mb-2">
                  How We Built This
                </div>
                <ul className="space-y-1 text-sm text-text-body">
                  {data.recommendations[0].howWeBuiltIt.map((step, idx) => (
                    <li key={idx} className="flex gap-2">
                      <span className="text-bright-green">•</span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
