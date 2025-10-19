'use client';

import Link from 'next/link';
import { useState } from 'react';
import { formatDate } from '@/lib/utils';
import type { ProjectWithAnalyses, Analysis } from '@/types/database';

export default function DashboardPage() {
  // Mock data - replace with real API call to Supabase
  const [projects] = useState<ProjectWithAnalyses[]>([
    {
      id: '1',
      user_id: 'mock-user',
      name: 'European Expansion Strategy',
      description: 'Comprehensive analysis of EU market entry',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      analyses: [
        {
          id: 'a1',
          user_id: 'mock-user',
          project_id: '1',
          query: 'Market sizing for Western Europe',
          status: 'completed',
          quality_score: 0.92,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
        {
          id: 'a2',
          user_id: 'mock-user',
          project_id: '1',
          query: 'Competitive landscape analysis',
          status: 'completed',
          quality_score: 0.88,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        },
      ],
      analysis_count: 2,
    },
  ]);

  const [standaloneAnalyses] = useState<Analysis[]>([
    {
      id: 's1',
      user_id: 'mock-user',
      project_id: null,
      query: 'Quick pricing strategy check for Q4',
      status: 'completed',
      quality_score: 0.85,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
  ]);

  const hasData = projects.length > 0 || standaloneAnalyses.length > 0;

  return (
    <div className="min-h-screen bg-cream-bg">
      {/* Header - Matching Report V2 Style */}
      <header className="border-b border-border-default bg-white sticky top-0 z-20">
        <div className="container-wide py-3">
          <div className="flex justify-between items-center">
            <Link href="/" className="text-xl font-bold text-warm-black hover:text-bright-green transition">
              Lolla
            </Link>
            <div className="flex gap-6 items-center">
              <Link href="/academy" className="text-sm text-text-body hover:text-warm-black transition">
                Academy
              </Link>
              <button className="text-sm text-text-body hover:text-warm-black transition">
                Settings
              </button>
            <button className="w-8 h-8 rounded-full bg-white text-warm-black border-2 border-accent-green flex items-center justify-center font-semibold text-sm">
              M
            </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container-wide py-12">
        <div className="flex justify-between items-center mb-8">
          <div>
            <div className="text-[10px] uppercase tracking-[0.2em] text-text-label font-semibold mb-2">
              Your Work
            </div>
            <h1 className="text-3xl font-bold text-warm-black">Dashboard</h1>
          </div>
          <div className="flex gap-3">
            <Link href="/projects/new" className="bg-white text-warm-black border border-border-default px-6 py-3 font-semibold shadow-sm hover:shadow-md transition-all duration-300 rounded-2xl">
              + New Project
            </Link>
            <Link href="/analyze" className="bg-white text-warm-black px-6 py-3 font-semibold border-2 border-accent-green shadow-sm hover:shadow-md transition-all duration-300 rounded-2xl">
              + New Analysis
            </Link>
          </div>
        </div>

        {!hasData ? (
          <div className="text-center py-20">
            <div className="max-w-lg mx-auto bg-white rounded-3xl border border-border-default p-12 shadow-sm">
              <h2 className="text-3xl font-bold mb-4 text-warm-black">Welcome to Lolla</h2>
              <p className="text-base text-text-body mb-8 leading-relaxed">
                Start your first strategic analysis or create a project to organize multiple analyses.
              </p>
              <div className="flex gap-4 justify-center">
                <Link href="/projects/new" className="bg-white text-warm-black border-2 border-border-default px-8 py-4 font-semibold shadow-sm hover:shadow-md transition-all duration-300 rounded-2xl">
                  Create Project
                </Link>
                <Link href="/analyze" className="bg-white text-warm-black px-8 py-4 font-semibold border-2 border-accent-green shadow-sm hover:shadow-md transition-all duration-300 rounded-2xl">
                  Start Analysis →
                </Link>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-12">
            {/* Projects Section */}
            {projects.length > 0 && (
              <section>
                <div className="mb-6">
                  <div className="text-[10px] uppercase tracking-[0.2em] text-text-label font-semibold mb-2">
                    Projects
                  </div>
                  <h2 className="text-2xl font-bold text-warm-black">Your Strategic Projects</h2>
                </div>
                <div className="grid grid-cols-1 gap-4">
                  {projects.map((project) => (
                    <Link
                      key={project.id}
                      href={`/projects/${project.id}`}
                      className="bg-white rounded-3xl border border-border-default hover:shadow-md hover:-translate-y-0.5 transition-all duration-300 group p-6"
                    >
                      <div className="flex justify-between items-start mb-4">
                        <div className="flex-1">
                          <h3 className="text-2xl font-semibold mb-2 text-warm-black">
                            {project.name}
                          </h3>
                          {project.description && (
                            <p className="text-text-body mb-3 leading-relaxed">{project.description}</p>
                          )}
                          <div className="flex gap-4 text-sm text-text-label">
                            <span className="inline-flex items-center gap-1">
                              <span className="h-1.5 w-1.5 rounded-full bg-accent-green" />
                              {project.analysis_count} analyses
                            </span>
                            <span>Updated {formatDate(project.updated_at)}</span>
                          </div>
                        </div>
                        <div className="ml-4">
                          <div className="px-4 py-2 border border-bright-green bg-card-active-bg text-sm font-semibold text-warm-black rounded-2xl">
                            CHAT
                          </div>
                        </div>
                      </div>

                      {/* Recent Analyses Preview */}
                      {project.analyses && project.analyses.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-border-default">
                          <div className="text-[10px] uppercase tracking-[0.1em] text-text-label font-semibold mb-3">
                            Recent Analyses
                          </div>
                          <div className="space-y-2">
                            {project.analyses.slice(0, 3).map((analysis) => (
                              <div key={analysis.id} className="flex justify-between items-center text-sm">
                                <span className="text-text-body truncate">{analysis.query}</span>
                                {analysis.quality_score && (
                                  <span className="text-xs text-bright-green font-medium ml-2 flex-shrink-0">
                                    {Math.round(analysis.quality_score * 100)}%
                                  </span>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </Link>
                  ))}
                </div>
              </section>
            )}

            {/* Standalone Analyses Section */}
            {standaloneAnalyses.length > 0 && (
              <section>
                <div className="mb-6">
                  <div className="text-[10px] uppercase tracking-[0.2em] text-text-label font-semibold mb-2">
                    Standalone
                  </div>
                  <h2 className="text-2xl font-bold text-warm-black">Recent Analyses</h2>
                </div>
                <div className="grid grid-cols-1 gap-4">
                  {standaloneAnalyses.map((analysis) => (
                    <Link
                      key={analysis.id}
                      href={`/analysis/${analysis.id}/report_v2`}
                      className="bg-white rounded-3xl border border-border-default hover:shadow-md hover:-translate-y-0.5 transition-all duration-300 group p-6"
                    >
                      <div className="flex justify-between items-start">
                        <div className="flex-1 min-w-0">
                          <h3 className="text-xl font-semibold mb-2 text-warm-black group-hover:text-bright-green transition truncate">
                            {analysis.query}
                          </h3>
                          <div className="flex gap-4 text-sm text-text-label">
                            <span>{formatDate(analysis.created_at)}</span>
                            {analysis.status === 'completed' && analysis.quality_score && (
                              <span className="text-bright-green font-medium inline-flex items-center gap-1">
                                <span className="h-1.5 w-1.5 rounded-full bg-bright-green" />
                                Quality: {Math.round(analysis.quality_score * 100)}%
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="ml-4 flex-shrink-0">
                          <span
                            className={`px-4 py-2 border text-sm font-semibold rounded-2xl ${
                              analysis.status === 'completed'
                                ? 'border-accent-green text-warm-black bg-white'
                                : 'border-border-default text-text-body bg-white'
                            }`}
                          >
                            {analysis.status.toUpperCase()}
                          </span>
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
