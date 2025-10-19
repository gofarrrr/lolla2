'use client';

import Link from 'next/link';
import { PermanentNav } from '@/components/PermanentNav';
import { TrustFooter } from '@/components/layout/TrustFooter';
import { NewsletterCTA } from '@/components/layout/NewsletterCTA';
import { PageContainer } from '@/components/layout/PageContainer';

const PLANS = [
  {
    name: 'Starter',
    price: 'Free',
    tagline: 'For individuals kicking off their first analysis.',
    cta: { label: 'Get started', href: '/signup' },
    features: [
      'Up to 3 analyses per month',
      'Glass-box report with confidence metrics',
      'Project workspace with evidence chat',
      'Email share links',
    ],
    badge: 'No credit card',
  },
  {
    name: 'Growth',
    price: '$89',
    cadence: 'per analyst / month',
    tagline: 'For teams running multi-analysis sprints.',
    cta: { label: 'Start a pilot', href: '/signup?plan=growth' },
    features: [
      'Unlimited analyses & projects',
      'Collaborative project chat with mentions',
      'Scenario simulator & unlockable playbooks',
      'CSV / Notion / Slides export bundle',
      'Priority support within 24 hours',
    ],
    highlighted: true,
    badge: 'Most popular',
  },
  {
    name: 'Enterprise',
    price: 'Custom',
    tagline: 'For global orgs needing governance & integrations.',
    cta: { label: 'Talk to us', href: '/contact' },
    features: [
      'SOC 2 Type I / SSO / SCIM',
      'Dedicated success architect & playbook onboarding',
      'Private model endpoints & evidence retention controls',
      'SLAs, audit trails, advanced analytics',
      'On-prem or VPC deployment options',
    ],
  },
];

const FAQ = [
  {
    q: 'What makes Lolla different from generic AI assistants?',
    a: 'Every analysis runs through five consultant archetypes with auditability. You see the exact evidence, mental models, and risk checks—not just a summary.'
  },
  {
    q: 'Can I invite teammates on the Starter plan?',
    a: 'Yes. Starter includes shared read access. For collaborative editing, upgrade to Growth and invite unlimited teammates.'
  },
  {
    q: 'Do you offer pilots or proof of value engagements?',
    a: 'Growth includes a 14-day pilot. Enterprise customers get a guided proof of value tailored to their initiative.'
  },
  {
    q: 'Where is my data stored and how is it secured?',
    a: 'We never train on your proprietary data. SOC 2-ready infrastructure, encryption at rest & in transit, plus optional data residency controls are included.'
  },
];

export default function PricingPage() {
  return (
    <div className="min-h-screen bg-cream-bg">
      <PermanentNav />

      <main className="py-16">
        <PageContainer className="space-y-16">
          <section className="text-center space-y-6">
            <span className="inline-flex items-center gap-2 rounded-full border border-brand-persimmon/30 bg-brand-persimmon/10 px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-warm-black">
              Pricing
            </span>
            <h1 className="text-4xl md:text-5xl font-bold text-warm-black">
              Pick the runway that fits your strategic cadence
            </h1>
            <p className="mx-auto max-w-2xl text-base text-text-body leading-relaxed">
              Every plan unlocks glass-box reports, evidence trails, and collaborative project chat. Start free and scale when your team needs parallel analyses.
            </p>
          </section>

          <section className="grid gap-6 md:grid-cols-3">
            {PLANS.map((plan) => (
              <div
                key={plan.name}
                className={`rounded-3xl border border-border-default bg-white px-8 py-10 shadow-sm transition hover:-translate-y-1 hover:shadow-md ${
                  plan.highlighted ? 'ring-2 ring-brand-lime/60 shadow-lg bg-brand-lime/10 border-brand-lime/40' : ''
                }`}
              >
                {plan.badge && (
                  <span className={`inline-flex rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] ${
                    plan.highlighted ? 'bg-warm-black text-white' : 'bg-brand-lime/15 text-warm-black'
                  }`}>
                    {plan.badge}
                  </span>
                )}
                <h2 className="mt-4 text-2xl font-semibold text-warm-black">{plan.name}</h2>
                <div className="mt-3 flex items-baseline gap-2">
                  <span className="text-3xl font-bold text-warm-black">{plan.price}</span>
                  {plan.cadence && <span className="text-xs text-text-label uppercase tracking-wider">{plan.cadence}</span>}
                </div>
                <p className="mt-3 text-sm text-text-body leading-relaxed">{plan.tagline}</p>
                <ul className="mt-6 space-y-2 text-sm text-text-body">
                  {plan.features.map((feature) => (
                    <li key={feature} className="flex items-start gap-2">
                      <span className="mt-1 h-1.5 w-1.5 rounded-full bg-brand-lime" />
                      <span>{feature}</span>
                    </li>
                  ))}
                </ul>
                <Link
                  href={plan.cta.href}
                  className={`mt-6 inline-flex w-full items-center justify-center rounded-2xl px-5 py-3 text-sm font-semibold transition ${
                    plan.highlighted
                      ? 'bg-warm-black text-white shadow-lg hover:-translate-y-0.5'
                      : 'border border-border-default text-warm-black hover:border-brand-lime/50 hover:bg-brand-lime/10'
                  }`}
                >
                  {plan.cta.label}
                </Link>
              </div>
            ))}
          </section>

          <section className="grid gap-8 lg:grid-cols-[minmax(0,0.55fr)_minmax(0,0.45fr)]">
            <div className="space-y-6">
              <div className="rounded-3xl border border-border-default bg-white px-8 py-8 shadow-sm">
                <h2 className="text-2xl font-semibold text-warm-black">What’s included across plans</h2>
                <div className="mt-6 grid gap-4 text-sm text-text-body md:grid-cols-2">
                  <div className="rounded-2xl border border-border-default bg-cream-bg px-4 py-4">
                    <h3 className="text-sm font-semibold text-warm-black">Glass-box reports</h3>
                    <p className="mt-2 text-xs text-text-body">
                      Every plan delivers mission, recommendations, risk summary, mental models, evidence, and scenarios with traceability.
                    </p>
                  </div>
                  <div className="rounded-2xl border border-border-default bg-cream-bg px-4 py-4">
                    <h3 className="text-sm font-semibold text-warm-black">Project workspaces</h3>
                    <p className="mt-2 text-xs text-text-body">
                      Organize analyses, track progress, and chat across evidence in one glass-box hub.
                    </p>
                  </div>
                  <div className="rounded-2xl border border-border-default bg-cream-bg px-4 py-4">
                    <h3 className="text-sm font-semibold text-warm-black">Consultant bench</h3>
                    <p className="mt-2 text-xs text-text-body">
                      Parallel consultant archetypes with mental model diversity, devil’s advocate checks, and scenario simulation.
                    </p>
                  </div>
                  <div className="rounded-2xl border border-border-default bg-cream-bg px-4 py-4">
                    <h3 className="text-sm font-semibold text-warm-black">Evidence & exports</h3>
                    <p className="mt-2 text-xs text-text-body">
                      Full citation list, CSV & slide exports, and links you can share with stakeholders.
                    </p>
                  </div>
                </div>
              </div>

              <div className="rounded-3xl border border-brand-persimmon/40 bg-brand-persimmon/10 px-8 py-8 shadow-sm">
                <h2 className="text-2xl font-semibold text-warm-black">Trusted by teams that need clarity</h2>
                <p className="mt-3 text-sm text-text-body leading-relaxed">
                  Operators at Atlas Ops, Beacon Labs, Northwind Ventures, and Helix Cloud rely on Lolla for high-stakes decisions.
                </p>
                <div className="mt-4 flex flex-wrap gap-3 text-sm text-text-label">
                  {['Atlas Ops', 'Beacon Labs', 'Northwind Ventures', 'Helix Cloud'].map((brand) => (
                    <span key={brand} className="inline-flex items-center rounded-full border border-border-default px-3 py-1">
                      {brand}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="rounded-3xl border border-border-default bg-white px-8 py-8 shadow-sm">
              <h2 className="text-2xl font-semibold text-warm-black">FAQ</h2>
              <dl className="mt-6 space-y-5">
                {FAQ.map((item) => (
                  <div key={item.q} className="border border-border-default rounded-2xl bg-cream-bg px-4 py-4">
                    <dt className="text-sm font-semibold text-warm-black">{item.q}</dt>
                    <dd className="mt-2 text-sm text-text-body">{item.a}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </section>
        </PageContainer>
      </main>

      <NewsletterCTA
        title="Wondering which plan fits your team?"
        description="Subscribe for weekly playbooks and deep dives on how teams are using Lolla across strategy, PM, and operations."
      />
      <TrustFooter
        headline="Ready to ship glass-box decisions?"
        subheadline="Run your first analysis in minutes and invite your team when you’re ready."
      />
    </div>
  );
}
