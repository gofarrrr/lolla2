'use client';

import Link from 'next/link';
import { useState } from 'react';
import { useParams } from 'next/navigation';
import { formatDate } from '@/lib/utils';
import type { ProjectWithAnalyses, ProjectChatMessage } from '@/types/database';
import { projectChatAPI, type ChatMessage } from '@/lib/api/project-chat';

export default function ProjectDetailPage() {
  const params = useParams();
  const projectId = params.id as string;

  // Mock data - replace with real API call
  const [project] = useState<ProjectWithAnalyses>({
    id: projectId,
    user_id: 'mock-user',
    name: 'European Expansion Strategy',
    description: 'Comprehensive analysis of EU market entry opportunities and risks',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    analyses: [
      {
        id: 'a1',
        user_id: 'mock-user',
        project_id: projectId,
        query: 'Market sizing for Western Europe tech sector',
        status: 'completed',
        quality_score: 0.92,
        created_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
        updated_at: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
      },
      {
        id: 'a2',
        user_id: 'mock-user',
        project_id: projectId,
        query: 'Competitive landscape and key players analysis',
        status: 'completed',
        quality_score: 0.88,
        created_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
        updated_at: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000).toISOString(),
      },
      {
        id: 'a3',
        user_id: 'mock-user',
        project_id: projectId,
        query: 'Entry strategy options: acquisition vs organic growth',
        status: 'completed',
        quality_score: 0.90,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ],
    analysis_count: 3,
  });

  const [chatMessages, setChatMessages] = useState<ProjectChatMessage[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!chatInput.trim()) return;

    const userMessage: ProjectChatMessage = {
      id: Date.now().toString(),
      project_id: projectId,
      user_id: 'mock-user',
      role: 'user',
      content: chatInput,
      created_at: new Date().toISOString(),
    };

    const currentInput = chatInput;
    setChatMessages([...chatMessages, userMessage]);
    setChatInput('');
    setIsLoading(true);

    try {
      // Convert ProjectChatMessage[] to ChatMessage[] for API
      const conversationHistory: ChatMessage[] = chatMessages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      const response = await projectChatAPI.sendMessage({
        project_id: projectId,
        user_id: 'mock-user', // TODO: Replace with real user ID from auth
        message: currentInput,
        conversation_history: conversationHistory,
      });

      const aiMessage: ProjectChatMessage = {
        id: (Date.now() + 1).toString(),
        project_id: projectId,
        user_id: 'mock-user',
        role: 'assistant',
        content: response.answer,
        context_used: {
          analysis_ids: response.sources.map((s) => s.analysis_id),
          evidence_refs: [],
        },
        created_at: new Date().toISOString(),
      };

      setChatMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: ProjectChatMessage = {
        id: (Date.now() + 1).toString(),
        project_id: projectId,
        user_id: 'mock-user',
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        created_at: new Date().toISOString(),
      };
      setChatMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-cream-bg">
      <header className="border-b border-border-default bg-white">
        <div className="container-wide py-3 flex items-center justify-between">
          <Link href="/dashboard" className="text-xl font-bold text-warm-black hover:text-bright-green transition">
            Lolla
          </Link>
          <div className="flex items-center gap-4 text-sm">
            <Link href="/academy" className="text-text-body hover:text-warm-black transition">
              Academy
            </Link>
            <button className="text-text-body hover:text-warm-black transition">Log Out</button>
          </div>
        </div>
      </header>

      <main className="container-wide py-10 space-y-10">
        {/* Breadcrumb */}
        <div className="mb-6 text-sm">
          <Link href="/dashboard" className="text-gray-600 hover:text-accent">
            Dashboard
          </Link>
          <span className="mx-2 text-gray-400">/</span>
          <span className="font-medium">{project.name}</span>
        </div>

        {/* Project Header */}
        <div className="rounded-3xl border border-border-default bg-white px-8 py-8 shadow-sm">
          <h1 className="text-3xl font-semibold text-warm-black">{project.name}</h1>
          {project.description && (
            <p className="mt-3 text-base text-text-body leading-relaxed">{project.description}</p>
          )}
          <div className="mt-4 flex flex-wrap gap-4 text-sm text-text-label">
            <span>{project.analysis_count} analyses</span>
            <span>Updated {formatDate(project.updated_at)}</span>
          </div>
        </div>

        {/* Main Grid: Analyses + Chat */}
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,0.6fr)_minmax(0,0.4fr)]">
          {/* Left: Analyses List */}
          <div>
            <div className="mb-6 flex items-center justify-between">
              <h2 className="text-xl font-semibold text-warm-black">Analyses</h2>
              <Link
                href={`/analyze?project=${projectId}`}
                className="inline-flex items-center gap-2 rounded-2xl bg-white border-2 border-accent-green px-5 py-2 text-sm font-semibold text-warm-black shadow-sm transition hover:shadow-md"
              >
                + New Analysis
              </Link>
            </div>

            <div className="space-y-4">
              {project.analyses?.map((analysis) => (
                <Link
                  key={analysis.id}
                  href={`/analysis/${analysis.id}/report_v2`}
                  className="block rounded-3xl border border-border-default bg-white px-6 py-5 shadow-sm transition hover:-translate-y-0.5 hover:border-brand-lime/60 hover:shadow-md"
                >
                  <h3 className="text-base font-semibold text-warm-black">{analysis.query}</h3>
                  <div className="mt-2 flex flex-wrap gap-4 text-xs text-text-label">
                    <span>{formatDate(analysis.created_at)}</span>
                    {analysis.quality_score && (
                      <span className="text-success">
                        Quality: {Math.round(analysis.quality_score * 100)}%
                      </span>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          </div>

          {/* Right: Chat Interface */}
          <div className="lg:sticky lg:top-8 lg:self-start">
            <div className="overflow-hidden rounded-3xl border border-border-default bg-white shadow-sm">
              <div className="border-b border-border-default bg-brand-lime/15 px-6 py-4">
                <h2 className="text-xl font-semibold text-warm-black">Project Chat</h2>
                <p className="mt-1 text-sm text-text-body">
                  Ask anything about your {project.analysis_count} analyses.
                </p>
              </div>

              <div className="h-[500px] overflow-y-auto bg-cream-bg px-6 py-5">
                {chatMessages.length === 0 ? (
                  <div className="rounded-3xl border border-dashed border-border-default bg-white/70 px-6 py-10 text-center text-sm text-text-body">
                    <div className="text-3xl mb-3">💡</div>
                    <p className="font-semibold text-warm-black">Start a conversation</p>
                    <p className="mt-1 text-xs text-text-label">
                      Use chat to get synthesis, compare analyses, or ask follow-ups.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {chatMessages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div
                          className={`max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-sm ${
                            message.role === 'user'
                              ? 'bg-white border-2 border-accent-green text-warm-black'
                              : 'bg-white border border-border-default text-text-body'
                          }`}
                        >
                          <p className="whitespace-pre-wrap">{message.content}</p>
                          {message.context_used && (
                            <div className="mt-2 border-t border-white/30 pt-2 text-xs opacity-70">
                              Used {message.context_used.analysis_ids.length} analyses
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                    {isLoading && (
                      <div className="flex justify-start">
                        <div className="rounded-2xl border border-border-default bg-white px-4 py-3">
                          <div className="flex gap-2 text-brand-lime">
                            <span className="h-2 w-2 rounded-full bg-brand-lime animate-bounce"></span>
                            <span className="h-2 w-2 rounded-full bg-brand-lime animate-bounce delay-150"></span>
                            <span className="h-2 w-2 rounded-full bg-brand-lime animate-bounce delay-300"></span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="border-t border-border-default bg-white px-6 py-5">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="Ask a question about your analyses..."
                    className="flex-1 rounded-2xl border border-border-default px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-brand-lime/40 focus:border-brand-lime"
                    disabled={isLoading}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={isLoading || !chatInput.trim()}
                    className="inline-flex items-center rounded-2xl border border-brand-lime/60 px-4 py-3 text-sm font-semibold text-warm-black transition hover:bg-brand-lime/10 disabled:opacity-50"
                  >
                    Send
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
