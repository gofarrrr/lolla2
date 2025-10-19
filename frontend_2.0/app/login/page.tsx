'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { PermanentNav } from '@/components/PermanentNav';
import { TrustFooter } from '@/components/layout/TrustFooter';
import { PageContainer } from '@/components/layout/PageContainer';

export default function LoginPage() {
  const router = useRouter();
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
      router.push('/dashboard');
    } catch (err) {
      console.error(err);
      setError('Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-canvas flex flex-col">
      <PermanentNav />

      <PageContainer className="flex w-full flex-1 flex-col gap-8 py-6 lg:flex-row lg:gap-10 lg:py-8">
        <aside className="hidden lg:flex lg:w-[46%]">
          <div className="relative flex h-full w-full flex-col justify-between overflow-hidden rounded-3xl p-12 bg-white">
            <div className="space-y-6">
              <span className="inline-flex w-max items-center gap-2 rounded-full bg-white border border-mesh px-4 py-2 text-xs font-bold uppercase tracking-wide text-ink-1">
                Cognitive Intelligence
              </span>
              <h2 className="text-4xl font-bold leading-tight text-ink-1">
                Think Like the World&rsquo;s Best Consultants
              </h2>
              <p className="text-base text-ink-2 leading-relaxed">
                Lolla stacks 200+ mental models across multiple AI consultants so you make decisions with confidence.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              {[
                { headline: '200+', body: 'Mental models embedded from McKinsey, BCG, Munger.' },
                { headline: '5×', body: 'Diverse consultant perspectives running in parallel.' },
                { headline: '24/7', body: 'Glass-box audit trail and evidence whenever you need it.' },
                { headline: '100%', body: 'SOC 2-ready, enterprise controls, and full transparency.' },
              ].map((item) => (
                <div key={item.headline} className="rounded-2xl border border-mesh px-5 py-4 bg-white">
                  <div className="text-lg font-semibold text-ink-1">{item.headline}</div>
                  <div className="mt-1 text-xs text-ink-2 leading-snug">{item.body}</div>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <section className="flex w-full items-center lg:w-[54%]">
          <div className="mx-auto w-full max-w-2xl rounded-2xl border border-mesh p-8 bg-white shadow-sm">
            <div className="mb-6">
              <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-ink-3">
                Welcome back
              </span>
              <h1 className="mt-2 text-2xl font-bold text-ink-1">Log In</h1>
              <p className="mt-1 text-xs text-ink-2 leading-relaxed">
                Access your cognitive intelligence workspace.
              </p>
            </div>

            {error && (
              <div className="mb-5 rounded-2xl border-2 border-accent-orange bg-white px-4 py-3">
                <p className="text-sm font-semibold text-ink-1">{error}</p>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
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
                  placeholder="you@company.com"
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
                  placeholder="••••••••"
                  required
                />
              </div>

              <div className="flex items-center gap-2 text-xs">
                <label className="flex items-center gap-2 cursor-pointer flex-1">
                  <input type="checkbox" className="h-3 w-3 rounded border border-mesh" />
                  <span className="text-ink-2">Remember me</span>
                </label>
                <Link
                  href="/forgot-password"
                  className="text-ink-2 hover:underline hover:decoration-accent-green"
                >
                  Forgot?
                </Link>
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full rounded-2xl bg-white border-2 border-accent-green py-4 text-base font-semibold text-warm-black shadow-sm transition-all duration-300 hover:shadow-md disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading ? 'Logging in…' : 'Log In →'}
              </button>
            </form>

            <div className="mt-4 text-center text-xs text-ink-2">
              Don&rsquo;t have an account?{' '}
              <Link
                href="/signup"
                className="font-semibold text-ink-1 hover:underline hover:decoration-accent-green"
              >
                Sign up
              </Link>
            </div>
          </div>
        </section>
      </PageContainer>

      <TrustFooter
        headline="Need a reminder of what Lolla unlocks?"
        subheadline="Spin up a fresh analysis, invite your team, and keep the glass-box evidence trail in one place."
        primaryCta={{ label: 'Start an analysis', href: '/analyze' }}
        secondaryCta={{ label: 'View sample reports', href: '/dashboard' }}
      />
    </div>
  );
}
