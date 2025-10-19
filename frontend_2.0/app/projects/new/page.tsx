'use client';

import Link from 'next/link';
import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function NewProjectPage() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    // TODO: Replace with actual Supabase API call
    // const { data, error } = await supabase
    //   .from('projects')
    //   .insert([{ name, description, user_id: session.user.id }])
    //   .select()
    //   .single();

    // Mock success - redirect to project detail
    setTimeout(() => {
      router.push('/projects/mock-id-123');
    }, 500);
  };

  return (
    <div className="min-h-screen bg-cream-bg">
      <header className="border-b border-border-default bg-white">
        <div className="container-wide flex items-center justify-between py-3">
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

      <main className="container-wide py-12">
        <div className="grid gap-8 lg:grid-cols-[minmax(0,0.6fr)_minmax(0,0.4fr)]">
          <div className="space-y-6">
            <div className="text-sm">
              <Link href="/dashboard" className="text-text-body hover:text-bright-green transition">
                Dashboard
              </Link>
              <span className="mx-2 text-text-label">/</span>
              <span className="font-medium text-warm-black">New Project</span>
            </div>

            <div className="rounded-3xl border border-border-default bg-white px-10 py-9 shadow-sm">
              <h1 className="text-3xl font-semibold text-warm-black">Create New Project</h1>
              <p className="mt-2 text-sm text-text-body leading-relaxed">
                Organize analyses together and keep a glass-box chat across every output.
              </p>

              <form onSubmit={handleSubmit} className="mt-6 space-y-6">
                <div>
                  <label
                    htmlFor="name"
                    className="mb-2 block text-xs font-semibold uppercase tracking-[0.2em] text-text-label"
                  >
                    Project Name *
                  </label>
                  <input
                    id="name"
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="w-full rounded-2xl border border-border-default px-4 py-3 text-base transition focus:outline-none focus:ring-2 focus:ring-brand-lime/30 focus:border-brand-lime"
                    placeholder="e.g., European Expansion Strategy"
                    required
                  />
                </div>

                <div>
                  <label
                    htmlFor="description"
                    className="mb-2 block text-xs font-semibold uppercase tracking-[0.2em] text-text-label"
                  >
                    Description (optional)
                  </label>
                  <textarea
                    id="description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    className="w-full min-h-[120px] rounded-2xl border border-border-default px-4 py-3 text-base transition focus:outline-none focus:ring-2 focus:ring-brand-lime/30 focus:border-brand-lime"
                    placeholder="Describe what this project is about..."
                  />
                  <p className="mt-2 text-xs text-text-label">
                    Help your future self remember the context and goals
                  </p>
                </div>

                <div className="rounded-3xl border border-brand-lime/30 bg-brand-lime/10 px-6 py-5">
                  <h3 className="text-sm font-semibold text-warm-black mb-3">What you can do with projects:</h3>
                  <ul className="space-y-2 text-sm text-text-body">
                    {[
                      'Organize multiple analyses under one strategic initiative.',
                      'Chat with all project data—ask cross-analysis questions.',
                      'Access consultant perspectives, research, and evidence in one place.',
                      'Track progress and evolution of your strategic thinking.',
                    ].map((item) => (
                      <li key={item} className="flex items-start gap-2">
                        <span className="text-brand-lime font-bold">✓</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="flex flex-wrap gap-3">
                  <button
                    type="submit"
                    disabled={isLoading || !name.trim()}
                    className="inline-flex items-center justify-center rounded-2xl bg-white border-2 border-accent-green px-6 py-3 text-base font-semibold text-warm-black shadow-sm transition hover:shadow-md disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isLoading ? 'Creating...' : 'Create Project →'}
                  </button>
                  <Link
                    href="/dashboard"
                    className="inline-flex items-center justify-center rounded-2xl border border-border-default px-6 py-3 text-base font-semibold text-warm-black transition hover:-translate-y-0.5 hover:border-accent-orange/60"
                  >
                    Cancel
                  </Link>
                </div>
              </form>
            </div>
          </div>

          <div className="rounded-3xl border border-border-default bg-white px-10 py-10 shadow-sm">
            <h2 className="text-lg font-semibold text-warm-black">Why projects matter</h2>
            <p className="mt-3 text-sm text-text-body leading-relaxed">
              Projects bundle every analysis, chat, and evidence artifact into a living workspace. Keep track of strategic
              initiatives without hunting for context.
            </p>
            <div className="mt-6 space-y-4">
              <div className="rounded-2xl border border-border-default bg-cream-bg px-5 py-4">
                <h3 className="text-sm font-semibold text-warm-black">Stay aligned</h3>
                <p className="mt-1 text-xs text-text-body">
                  Invite stakeholders, share analyses, and collaborate directly in the glass-box workspace.
                </p>
              </div>
              <div className="rounded-2xl border border-border-default bg-cream-bg px-5 py-4">
                <h3 className="text-sm font-semibold text-warm-black">Cross-analysis insight</h3>
                <p className="mt-1 text-xs text-text-body">
                  Lolla’s project chat references every analysis so you can reason across initiatives.
                </p>
              </div>
              <div className="rounded-2xl border border-border-default bg-cream-bg px-5 py-4">
                <h3 className="text-sm font-semibold text-warm-black">Ready for teams</h3>
                <p className="mt-1 text-xs text-text-body">
                  SOC 2-ready controls, audit trails, and access management built in.
                </p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
