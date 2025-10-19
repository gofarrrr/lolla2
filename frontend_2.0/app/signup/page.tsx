'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { PermanentNav } from '@/components/PermanentNav';
import { TrustFooter } from '@/components/layout/TrustFooter';
import { PageContainer } from '@/components/layout/PageContainer';

export default function SignupPage() {
  const router = useRouter();
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError('');
    setLoading(true);

    try {
      await new Promise((resolve) => setTimeout(resolve, 500));
      localStorage.setItem('lolla_session', 'mock_session_token');
      router.push('/analyze');
    } catch (err) {
      console.error(err);
      setError('Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-canvas flex flex-col">
      <PermanentNav />

      <PageContainer className="flex w-full flex-1 flex-col gap-8 py-6 lg:flex-row lg:py-8">
        <aside className="hidden lg:flex lg:w-1/2">
          <div className="relative flex h-full w-full flex-col justify-center rounded-3xl p-12 bg-white">
            <span className="inline-flex w-max items-center gap-2 rounded-full bg-white border border-mesh px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-1">
              Free forever tier
            </span>
            <h2 className="mt-6 text-4xl font-bold leading-tight text-ink-1">
              Start Your First Analysis in Under 3 Minutes
            </h2>
            <p className="mt-4 text-lg text-ink-2 leading-relaxed">
              No credit card required. No commitment. Just exponential insights.
            </p>
            <div className="mt-8 space-y-4">
              {[
                { headline: '200+', body: 'Elite mental models from McKinsey, BCG, and Munger.' },
                { headline: '5×', body: 'Multiple consultant perspectives working simultaneously.' },
                { headline: '100%', body: 'Glass-box transparency on every reasoning step.' },
              ].map((item) => (
                <div key={item.headline} className="border-l-2 border-accent-green pl-5">
                  <div className="text-2xl font-bold text-ink-1">{item.headline}</div>
                  <div className="text-sm text-ink-2">{item.body}</div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <section className="flex w-full items-center lg:w-1/2">
          <div className="mx-auto w-full max-w-md rounded-2xl border border-mesh bg-white px-6 py-8 shadow-sm">
            <div className="mb-6">
              <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-ink-3">
                Welcome
              </span>
              <h1 className="mt-2 text-2xl font-bold text-ink-1">Get Started</h1>
              <p className="mt-1 text-xs text-ink-2 leading-relaxed">
                Join operators using Lolla for sharper decisions.
              </p>
            </div>

            {error && (
              <div className="mb-5 rounded-2xl border-2 border-accent-orange bg-white px-4 py-3">
                <p className="text-sm font-semibold text-ink-1">{error}</p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-3">
              <div>
                <label className="mb-1 block text-xs font-semibold text-ink-1 uppercase tracking-wide" htmlFor="name">
                  Name
                </label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  className="w-full rounded-xl border border-mesh px-3 py-2 text-sm transition focus:outline-none focus:ring-2 focus:ring-accent-green focus:border-transparent"
                  placeholder="Jane Smith"
                  required
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold text-ink-1 uppercase tracking-wide" htmlFor="email">
                  Email
                </label>
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  className="w-full rounded-xl border border-mesh px-3 py-2 text-sm transition focus:outline-none focus:ring-2 focus:ring-accent-green focus:border-transparent"
                  placeholder="jane@company.com"
                  required
                />
              </div>

              <div>
                <label className="mb-1 block text-xs font-semibold text-ink-1 uppercase tracking-wide" htmlFor="password">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className="w-full rounded-xl border border-mesh px-3 py-2 text-sm transition focus:outline-none focus:ring-2 focus:ring-accent-green focus:border-transparent"
                  placeholder="8+ characters"
                  minLength={8}
                  required
                />
              </div>

              <label className="flex items-start gap-2 text-[11px] text-ink-2 leading-snug">
                <input
                  type="checkbox"
                  className="mt-0.5 h-3 w-3 rounded border border-mesh flex-shrink-0"
                  required
                />
                <span>
                  I agree to the{' '}
                  <Link href="/terms" className="font-semibold text-ink-1 hover:underline hover:decoration-accent-green">
                    Terms
                  </Link>{' '}
                  &{' '}
                  <Link href="/privacy" className="font-semibold text-ink-1 hover:underline hover:decoration-accent-green">
                    Privacy
                  </Link>
                </span>
              </label>

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-2xl bg-white border-2 border-accent-green py-4 text-base font-semibold text-warm-black shadow-sm transition-all duration-300 hover:shadow-md disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? 'Creating account…' : 'Start Free Analysis →'}
              </button>
            </form>

            <div className="mt-4 text-center text-xs text-ink-2">
              Have an account?{' '}
              <Link
                href="/login"
                className="font-semibold text-ink-1 hover:underline hover:decoration-accent-green"
              >
                Log in
              </Link>
            </div>
          </div>
        </section>
      </PageContainer>

      <TrustFooter
        headline="Launch your first strategic sprint"
        subheadline="Upload context, answer a few prompts, and Lolla delivers a glass-box decision brief end to end."
        primaryCta={{ label: 'Start an analysis', href: '/analyze' }}
        secondaryCta={{ label: 'See pricing', href: '/pricing' }}
      />
    </div>
  );
}
