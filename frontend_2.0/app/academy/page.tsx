import Link from 'next/link';
import { PermanentNav } from '@/components/PermanentNav';
import { NewsletterCTA } from '@/components/layout/NewsletterCTA';
import { TrustFooter } from '@/components/layout/TrustFooter';
import { PageContainer } from '@/components/layout/PageContainer';

const MENTAL_MODEL_CATEGORIES = [
  {
    name: 'Strategic Frameworks',
    count: 23,
    models: ['Porter\'s Five Forces', 'McKinsey 7-S', 'BCG Matrix', 'Blue Ocean Strategy'],
  },
  {
    name: 'Decision Science',
    count: 18,
    models: ['Expected Value', 'Decision Trees', 'Bayesian Thinking', 'Opportunity Cost'],
  },
  {
    name: 'Systems Thinking',
    count: 16,
    models: ['Feedback Loops', 'Leverage Points', 'System Archetypes', 'Stock and Flow'],
  },
  {
    name: 'Behavioral Economics',
    count: 22,
    models: ['Loss Aversion', 'Anchoring', 'Confirmation Bias', 'Sunk Cost Fallacy'],
  },
  {
    name: 'Innovation & Disruption',
    count: 14,
    models: ['Jobs to be Done', 'Disruptive Innovation', 'S-Curve', 'Technology Adoption'],
  },
  {
    name: 'Financial & Investment',
    count: 19,
    models: ['DCF Analysis', 'Moat Analysis', 'Circle of Competence', 'Margin of Safety'],
  },
  {
    name: 'Organizational Design',
    count: 15,
    models: ['Conway\'s Law', 'Span of Control', 'Matrix Structure', 'Agile Organizations'],
  },
  {
    name: 'Game Theory',
    count: 10,
    models: ['Nash Equilibrium', 'Prisoner\'s Dilemma', 'Zero-Sum Games', 'Iterative Games'],
  },
];

const NWAY_RELATIONS = [
  {
    id: 'NWAY_DECISION_002',
    name: 'Strategic Decision Under Uncertainty',
    models: ['Expected Value', 'Bayesian Updating', 'Option Value', 'Scenario Planning', 'Loss Aversion'],
    strength: 0.82,
  },
  {
    id: 'NWAY_INNOVATION_001',
    name: 'Disruptive Innovation Analysis',
    models: ['Jobs to be Done', 'S-Curve', 'Crossing the Chasm', 'Platform Effects', 'Technology Adoption'],
    strength: 0.76,
  },
  {
    id: 'NWAY_COMPETITIVE_001',
    name: 'Competitive Strategy',
    models: ['Porter\'s Five Forces', 'Blue Ocean', 'Network Effects', 'Switching Costs', 'Economies of Scale'],
    strength: 0.84,
  },
];

export default function AcademyPage() {
  return (
    <div className="min-h-screen bg-cream-bg">
      <PermanentNav />

      <main className="py-12">
        <PageContainer className="max-w-6xl">
          <div className="max-w-4xl">
          <span className="inline-flex items-center gap-2 rounded-full border border-brand-lime/30 bg-brand-lime/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] text-warm-black">
            Academy
          </span>
          <h1 className="mt-4 text-4xl font-bold text-warm-black">Mental Models Library</h1>
          <p className="mt-3 text-lg leading-relaxed text-text-body">
            137 mental models from 200+ authoritative sources. Learn the frameworks that power strategic thinking.
          </p>

          {/* Mental Model Categories */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold mb-6">Browse by Category</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {MENTAL_MODEL_CATEGORIES.map((category) => (
                <div key={category.name} className="card hover:border-accent">
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="text-xl font-semibold">{category.name}</h3>
                    <span className="text-sm text-gray-600">{category.count} models</span>
                  </div>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {category.models.map((model) => (
                      <li key={model}>• {model}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>

          {/* N-Way Relations */}
          <section className="mb-16">
            <h2 className="text-3xl font-bold mb-4">N-Way Relations</h2>
            <p className="text-lg text-gray-600 mb-6">
              See how mental models combine for exponential insights (Lollapalooza Effect).
            </p>
            <div className="space-y-4">
              {NWAY_RELATIONS.map((nway) => (
                <div key={nway.id} className="card">
                  <div className="flex justify-between items-start mb-3">
                    <h3 className="text-xl font-semibold">{nway.name}</h3>
                    <span className="text-sm text-accent">
                      Strength: {Math.round(nway.strength * 100)}%
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-2">{nway.id}</p>
                  <div className="flex flex-wrap gap-2">
                    {nway.models.map((model) => (
                      <span key={model} className="px-2 py-1 border-2 border-black text-xs">
                        {model}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* CTA */}
          <div className="rounded-3xl border border-border-default bg-white/95 px-8 py-10 text-center shadow-sm">
            <h2 className="text-2xl font-bold mb-4 text-warm-black">Apply These Models to Your Business</h2>
            <p className="text-lg text-text-body mb-6">
              Get AI consultant teams trained on these frameworks to analyze your strategic questions.
            </p>
            <Link href="/analyze" className="inline-flex items-center justify-center gap-2 rounded-2xl bg-white border-2 border-accent-green px-6 py-3 text-base font-semibold text-warm-black shadow-sm hover:shadow-md transition-all duration-300">
              Start Free Analysis →
            </Link>
          </div>
          </div>
        </PageContainer>
      </main>

      <NewsletterCTA
        title="Level up your cognitive toolkit"
        description="Receive curated mental models, breakdowns, and N-way relation templates every Friday."
      />
      <TrustFooter
        headline="Turn mental models into shipped strategy"
        subheadline="Clone these frameworks into a live consultant run and get a glass-box decision report."
        primaryCta={{ label: 'Launch an analysis', href: '/analyze' }}
        secondaryCta={{ label: 'View sample reports', href: '/dashboard' }}
      />
    </div>
  );
}
