'use client';

import { use } from 'react';
import Link from 'next/link';
import { useCopywriterResults, useCopywriterStatus } from '@/lib/api/hooks';
import { downloadAsFile } from '@/lib/utils';

export default function CopywriterResultsPage({ params }: { params: Promise<{ job_id: string }> }) {
  const { job_id } = use(params);
  const { data: status } = useCopywriterStatus(job_id);
  const { data: results, isLoading } = useCopywriterResults(
    status?.status === 'completed' ? job_id : null
  );

  if (isLoading || status?.status !== 'completed') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg mb-2">Crafting persuasive copy...</p>
          <p className="text-gray-600">{status?.current_stage || 'Processing'}</p>
        </div>
      </div>
    );
  }

  if (!results) return null;

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b-2 border-black">
        <div className="container-wide py-4">
          <Link href="/dashboard" className="text-2xl font-bold">Lolla</Link>
        </div>
      </header>

      <main className="container-content py-12">
        <h1 className="text-4xl font-bold mb-8">Communication Package</h1>

        <section className="mb-8 card bg-accent text-white border-accent">
          <h2 className="text-xl font-bold mb-2">Governing Thought (12 Words)</h2>
          <p className="text-2xl leading-relaxed">{results.governing_thought}</p>
        </section>

        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Polished Content</h2>
          <div className="prose prose-lg max-w-none border-2 border-black p-6">
            <div style={{ whiteSpace: 'pre-wrap' }}>{results.polished_content}</div>
          </div>
          <p className="text-sm text-gray-600 mt-2">{results.word_count} words</p>
        </section>

        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Quality Scores</h2>
          <div className="grid grid-cols-4 gap-4">
            <div className="card text-center">
              <p className="text-sm text-gray-600">Clarity</p>
              <p className="text-3xl font-bold">{Math.round(results.clarity_score * 100)}%</p>
            </div>
            <div className="card text-center">
              <p className="text-sm text-gray-600">Persuasion</p>
              <p className="text-3xl font-bold">{Math.round(results.persuasion_score * 100)}%</p>
            </div>
            <div className="card text-center">
              <p className="text-sm text-gray-600">Skim Test</p>
              <p className="text-3xl font-bold">{Math.round(results.skim_test_score * 100)}%</p>
            </div>
            <div className="card text-center">
              <p className="text-sm text-gray-600">Defensibility</p>
              <p className="text-3xl font-bold">{Math.round(results.defensibility_score * 100)}%</p>
            </div>
          </div>
        </section>

        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Anticipated Objections</h2>
          <div className="space-y-3">
            {results.anticipated_objections.map((objection, index) => (
              <div key={index} className="card">
                <div className="flex justify-between items-start mb-2">
                  <span className="font-semibold">{objection.objection_type}</span>
                  <span className={`px-2 py-1 border-2 text-xs ${
                    objection.severity === 'critical' ? 'border-error' : 'border-warning'
                  }`}>
                    {objection.severity.toUpperCase()}
                  </span>
                </div>
                <p className="text-gray-600">{objection.statement}</p>
              </div>
            ))}
          </div>
        </section>

        <button
          onClick={() => downloadAsFile(results.polished_content, `copy-${job_id}.txt`)}
          className="btn btn-accent w-full"
        >
          Download Copy →
        </button>
      </main>
    </div>
  );
}
