'use client';

import Link from 'next/link';
import { TrustFooter } from '@/components/layout/TrustFooter';
import { PageContainer } from '@/components/layout/PageContainer';

export default function Home() {
  return (
    <>
      {/* Navigation - Granola-inspired concentrated layout */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 border-b border-border-default backdrop-blur-md">
        <PageContainer className="max-w-7xl">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-6">
              <Link href="/" className="text-lg font-bold tracking-tight text-warm-black hover:text-bright-green transition-colors duration-200">
                Lolla
              </Link>

              <div className="h-4 w-px bg-border-default hidden sm:block" />

              <div className="hidden sm:flex items-center gap-1">
                <Link href="/academy" className="px-3 py-1.5 text-sm font-medium text-text-body hover:text-warm-black hover:bg-cream-bg rounded-lg transition-all duration-200">
                  Academy
                </Link>
                <Link href="/dashboard" className="px-3 py-1.5 text-sm font-medium text-text-body hover:text-warm-black hover:bg-cream-bg rounded-lg transition-all duration-200">
                  Reports
                </Link>
                <Link href="/blog" className="px-3 py-1.5 text-sm font-medium text-text-body hover:text-warm-black hover:bg-cream-bg rounded-lg transition-all duration-200">
                  Blog
                </Link>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <Link href="/login" className="hidden sm:inline-flex px-3 py-1.5 text-sm font-medium text-text-body hover:text-warm-black hover:bg-cream-bg rounded-lg transition-all duration-200">
                Sign In
              </Link>
              <Link href="/signup" className="inline-flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-white text-warm-black text-sm font-semibold border-2 border-accent-green shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-300">
                Get Started
                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M5 12h14M12 5l7 7-7 7"/>
                </svg>
              </Link>
            </div>
          </div>
        </PageContainer>
      </nav>

      {/* Hero Section - Report V2 Standard */}
      <section className="relative pt-32 pb-24 overflow-hidden bg-cream-bg">
        <div className="absolute inset-0 flex items-center justify-center" style={{isolation: 'isolate'}}>
          {/* Neutral background (no color fills) */}
          <div className="absolute inset-0 -z-10" />

          {/* Content */}
          <div className="relative aspect-square flex w-[90vw] max-w-3xl items-center justify-center">

            {/* Content */}
            <div className="relative z-20 flex flex-col text-center px-6 py-8 space-y-8 items-center">
              {/* Pill - with persimmon accent */}
              <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white backdrop-blur-sm shadow-sm border border-border-default text-xs font-semibold text-warm-black uppercase tracking-wide group hover:border-bright-green transition-colors duration-300">
                <svg className="w-3.5 h-3.5 text-bright-green animate-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z"/>
                </svg>
                AI Cognitive Intelligence
              </span>

              {/* Title - clean with subtle accent */}
              <h1 className="sm:text-6xl text-5xl font-bold text-warm-black tracking-tight leading-tight">
                Think Like<br/>
                <span className="relative inline-block text-warm-black">
                  The Best
                  <span className="absolute -bottom-2 left-0 right-0 h-1 bg-bright-green rounded-full opacity-40"></span>
                </span>
              </h1>

              {/* Subcopy */}
              <p className="mt-3 text-base sm:text-lg text-text-body max-w-2xl leading-relaxed">
                200+ mental models. 5 AI consultants. Exponential insights in under 3 minutes.
              </p>

              {/* CTAs */}
              <div className="flex gap-4 mt-8 items-center flex-wrap justify-center">
                <Link href="/signup" className="group inline-flex items-center justify-center gap-2 px-8 py-4 rounded-2xl bg-white text-warm-black text-base font-semibold border-2 border-accent-green shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all duration-300">
                  <span>Start Free Analysis</span>
                  <svg className="w-4 h-4 group-hover:translate-x-0.5 transition-transform duration-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M5 12h14M12 5l7 7-7 7"/>
                  </svg>
                </Link>
                <Link href="/academy" className="group inline-flex items-center justify-center gap-2 px-8 py-4 rounded-2xl bg-white text-warm-black text-base font-medium border border-border-default shadow-sm hover:shadow-md hover:border-accent-green/60 hover:-translate-y-0.5 transition-all duration-300">
                  <span>Learn Frameworks</span>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="bg-white py-24 sm:py-32">
        <PageContainer className="max-w-7xl">
          {/* Header */}
          <div className="text-center mb-20">
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cream-bg shadow-sm border border-border-default text-xs font-semibold text-warm-black uppercase tracking-wide">
              How It Works
            </span>
            <h2 className="sm:text-5xl text-4xl font-bold tracking-tight mt-8 text-warm-black">Three Simple Steps</h2>
            <p className="text-lg sm:text-xl text-text-body max-w-2xl mx-auto mt-4 leading-relaxed">Multiple perspectives converge for exponential insights.</p>
          </div>

          {/* Cards */}
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {/* Card 1 */}
            <div className="group relative bg-cream-bg rounded-3xl p-8 border border-border-default shadow-sm hover:shadow-md hover:border-gray-300 hover:-translate-y-1 transition-all duration-300">
              <div className="text-3xl font-bold text-bright-green/70 mb-6 group-hover:text-bright-green transition-colors duration-300">01</div>
              <h3 className="text-xl font-semibold tracking-tight text-warm-black mb-3 leading-snug">Ask Your Question</h3>
              <p className="text-text-body leading-relaxed">Enter any strategic decision. Upload documents for deeper analysis.</p>
            </div>

            {/* Card 2 */}
            <div className="group relative bg-cream-bg rounded-3xl p-8 border border-border-default shadow-sm hover:shadow-md hover:border-gray-300 hover:-translate-y-1 transition-all duration-300">
              <div className="text-3xl font-bold text-bright-green/70 mb-6 group-hover:text-bright-green transition-colors duration-300">02</div>
              <h3 className="text-xl font-semibold tracking-tight text-warm-black mb-3 leading-snug">AI Consultant Teams</h3>
              <p className="text-text-body leading-relaxed">5 perspectives analyze using 200+ McKinsey & BCG frameworks simultaneously.</p>
            </div>

            {/* Card 3 */}
            <div className="group relative bg-cream-bg rounded-3xl p-8 border border-border-default shadow-sm hover:shadow-md hover:border-gray-300 hover:-translate-y-1 transition-all duration-300">
              <div className="text-3xl font-bold text-bright-green/70 mb-6 group-hover:text-bright-green transition-colors duration-300">03</div>
              <h3 className="text-xl font-semibold tracking-tight text-warm-black mb-3 leading-snug">Actionable Report</h3>
              <p className="text-text-body leading-relaxed">Executive summary, strategic recommendations, and complete transparency.</p>
            </div>
          </div>
        </PageContainer>
      </section>

      {/* Features Section - with persimmon energy accents */}
      <section className="bg-off-white py-24 sm:py-32 relative overflow-hidden">
        {/* Ambient background */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-soft-persimmon/5 rounded-full blur-3xl -z-10"></div>
        <div className="absolute bottom-0 left-0 w-96 h-96 bg-bright-green/5 rounded-full blur-3xl -z-10"></div>

        <PageContainer className="max-w-7xl">
          <div className="text-center mb-16">
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white shadow-sm border border-border-default text-xs font-semibold text-warm-black uppercase tracking-wide mb-4">
              The Numbers
            </span>
            <h3 className="text-4xl md:text-5xl font-bold text-warm-black">Built for Serious Thinkers</h3>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-5">
            {/* Card 1 - Green accent */}
            <div className="group relative bg-white rounded-3xl border border-border-default p-6 shadow-sm hover:shadow-md hover:border-bright-green hover:-translate-y-1 transition-all duration-300 overflow-hidden">
              <div className="absolute top-0 right-0 w-20 h-20 bg-bright-green/5 rounded-full blur-2xl group-hover:bg-bright-green/10 transition-all duration-300"></div>
              <div className="relative">
                <div className="text-3xl font-bold mb-3 text-warm-black group-hover:text-bright-green transition-colors">200+</div>
                <div className="text-xs font-bold mb-2 text-text-label tracking-wider">MENTAL MODELS</div>
                <div className="text-sm text-text-body leading-relaxed">McKinsey, BCG, Munger, Kahneman frameworks</div>
              </div>
            </div>

            {/* Card 2 - Persimmon accent */}
            <div className="group relative bg-white rounded-3xl border border-border-default p-6 shadow-sm hover:shadow-md hover:border-soft-persimmon hover:-translate-y-1 transition-all duration-300 overflow-hidden">
              <div className="absolute top-0 right-0 w-20 h-20 bg-soft-persimmon/5 rounded-full blur-2xl group-hover:bg-soft-persimmon/10 transition-all duration-300"></div>
              <div className="relative">
                <div className="text-3xl font-bold mb-3 text-warm-black group-hover:text-soft-persimmon transition-colors">&lt;3min</div>
                <div className="text-xs font-bold mb-2 text-text-label tracking-wider">ANALYSIS TIME</div>
                <div className="text-sm text-text-body leading-relaxed">Full strategic report in under 3 minutes</div>
              </div>
            </div>

            {/* Card 3 - Green accent */}
            <div className="group relative bg-white rounded-3xl border border-border-default p-6 shadow-sm hover:shadow-md hover:border-bright-green hover:-translate-y-1 transition-all duration-300 overflow-hidden">
              <div className="absolute top-0 right-0 w-20 h-20 bg-bright-green/5 rounded-full blur-2xl group-hover:bg-bright-green/10 transition-all duration-300"></div>
              <div className="relative">
                <div className="text-3xl font-bold mb-3 text-warm-black group-hover:text-bright-green transition-colors">100%</div>
                <div className="text-xs font-bold mb-2 text-text-label tracking-wider">TRANSPARENT</div>
                <div className="text-sm text-text-body leading-relaxed">Glass-box AI—see every reasoning step</div>
              </div>
            </div>

            {/* Card 4 - Persimmon accent */}
            <div className="group relative bg-white rounded-3xl border border-border-default p-6 shadow-sm hover:shadow-md hover:border-soft-persimmon hover:-translate-y-1 transition-all duration-300 overflow-hidden">
              <div className="absolute top-0 right-0 w-20 h-20 bg-soft-persimmon/5 rounded-full blur-2xl group-hover:bg-soft-persimmon/10 transition-all duration-300"></div>
              <div className="relative">
                <div className="text-3xl font-bold mb-3 text-warm-black group-hover:text-soft-persimmon transition-colors">N-Way</div>
                <div className="text-xs font-bold mb-2 text-text-label tracking-wider">RELATIONS</div>
                <div className="text-sm text-text-body leading-relaxed">5 models working simultaneously</div>
              </div>
            </div>
          </div>
        </PageContainer>
      </section>

      {/* Use Cases */}
      <section className="bg-white py-24 sm:py-32">
        <PageContainer className="max-w-7xl">
          <h3 className="text-4xl md:text-5xl font-bold mb-6 text-warm-black">Who Uses Lolla</h3>
          <p className="text-xl text-text-body mb-16 max-w-3xl">
            From Fortune 500 strategy teams to solo founders making their next big bet.
          </p>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="group bg-cream-bg rounded-3xl border border-border-default p-8 shadow-sm hover:shadow-md hover:border-gray-300 transition-all duration-300">
              <h4 className="text-2xl font-semibold mb-4 text-warm-black">Executives & Leaders</h4>
              <p className="text-text-body leading-relaxed">
                Strategic decisions, market entry, M&A evaluation, organizational design
              </p>
            </div>
            <div className="group bg-cream-bg rounded-3xl border border-border-default p-8 shadow-sm hover:shadow-md hover:border-gray-300 transition-all duration-300">
              <h4 className="text-2xl font-semibold mb-4 text-warm-black">Founders & Operators</h4>
              <p className="text-text-body leading-relaxed">
                Product strategy, fundraising prep, competitive analysis, pivot decisions
              </p>
            </div>
            <div className="group bg-cream-bg rounded-3xl border border-border-default p-8 shadow-sm hover:shadow-md hover:border-gray-300 transition-all duration-300">
              <h4 className="text-2xl font-semibold mb-4 text-warm-black">Individuals</h4>
              <p className="text-text-body leading-relaxed">
                Career moves, personal investments, major life decisions, risk evaluation
              </p>
            </div>
          </div>
        </PageContainer>
      </section>

      <TrustFooter />
    </>
  );
}
