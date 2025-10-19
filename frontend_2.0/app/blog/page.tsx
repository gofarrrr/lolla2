'use client';

import Link from 'next/link';
import { PermanentNav } from '@/components/PermanentNav';
import { NewsletterCTA } from '@/components/layout/NewsletterCTA';
import { TrustFooter } from '@/components/layout/TrustFooter';
import { PageContainer } from '@/components/layout/PageContainer';

// Sample blog posts - replace with CMS or database in production
const blogPosts = [
  {
    id: 'thinking-in-mental-models',
    title: 'Thinking in Mental Models: Why Consultants See What Others Miss',
    excerpt: 'Mental models are cognitive frameworks that help us understand and navigate the world. Learn how top consultants use 200+ frameworks to analyze complex problems.',
    date: '2025-01-15',
    author: 'Lolla Team',
    category: 'Frameworks',
    readTime: '8 min read',
    image: '/blog/mental-models.jpg',
  },
  {
    id: 'lollapalooza-effect',
    title: 'The Lollapalooza Effect: When Multiple Mental Models Converge',
    excerpt: 'Charlie Munger\'s concept of the Lollapalooza Effect explains why combining multiple mental models creates exponential insights. Discover how Lolla implements this.',
    date: '2025-01-10',
    author: 'Lolla Team',
    category: 'Cognitive Science',
    readTime: '6 min read',
    image: '/blog/lollapalooza.jpg',
  },
  {
    id: 'ai-transparency-glass-box',
    title: 'Glass-Box AI: Why Transparency Matters in Strategic Analysis',
    excerpt: 'Unlike black-box AI systems, glass-box AI shows you every reasoning step. Learn why transparency is crucial for high-stakes decisions.',
    date: '2025-01-05',
    author: 'Lolla Team',
    category: 'AI & Technology',
    readTime: '7 min read',
    image: '/blog/transparency.jpg',
  },
];

export default function BlogPage() {
  return (
    <div className="min-h-screen bg-off-white">
      <PermanentNav />

      {/* Hero Section */}
      <section className="relative bg-white border-b border-border-default pt-16 pb-12">
        <PageContainer className="max-w-5xl">
          <div className="text-center">
            <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-cream-bg shadow-sm border border-border-default text-xs font-semibold text-warm-black uppercase tracking-wide mb-6">
              <svg className="w-3.5 h-3.5 text-bright-green" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 2L2 7v10c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V7l-10-5z"/>
              </svg>
              Insights & Frameworks
            </span>
            <h1 className="text-5xl md:text-6xl font-bold text-warm-black mb-4 tracking-tight">
              The Lolla Blog
            </h1>
            <p className="text-xl text-text-body max-w-2xl mx-auto leading-relaxed">
              Deep dives into mental models, cognitive science, and strategic thinking frameworks.
            </p>
          </div>
        </PageContainer>
      </section>

      {/* Blog Posts Grid */}
      <section className="py-16">
        <PageContainer className="max-w-6xl">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {blogPosts.map((post, index) => {
              // Alternate accent colors for visual interest
              const accentColor = index % 2 === 0 ? 'bright-green' : 'soft-persimmon';
              const hoverBorder = index % 2 === 0 ? 'hover:border-bright-green' : 'hover:border-soft-persimmon';

              return (
                <Link
                  key={post.id}
                  href={`/blog/${post.id}`}
                  className={`group bg-white rounded-3xl border border-border-default shadow-sm hover:shadow-md ${hoverBorder} hover:-translate-y-1 transition-all duration-300 overflow-hidden`}
                >
                  {/* Image placeholder */}
                  <div className={`relative h-48 bg-gradient-to-br ${
                    index % 2 === 0
                      ? 'from-bright-green/10 to-bright-green/5'
                      : 'from-soft-persimmon/10 to-soft-persimmon/5'
                  } flex items-center justify-center`}>
                    <svg className={`w-12 h-12 ${index % 2 === 0 ? 'text-bright-green' : 'text-soft-persimmon'} opacity-30`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z"/>
                    </svg>
                  </div>

                  {/* Content */}
                  <div className="p-6">
                    {/* Category & Read Time */}
                    <div className="flex items-center gap-3 mb-3">
                      <span className={`text-xs font-semibold uppercase tracking-wider ${
                        index % 2 === 0 ? 'text-bright-green' : 'text-soft-persimmon'
                      }`}>
                        {post.category}
                      </span>
                      <span className="text-xs text-text-label">•</span>
                      <span className="text-xs text-text-label">{post.readTime}</span>
                    </div>

                    {/* Title */}
                    <h3 className="text-xl font-bold text-warm-black mb-3 leading-tight group-hover:text-bright-green transition-colors">
                      {post.title}
                    </h3>

                    {/* Excerpt */}
                    <p className="text-sm text-text-body leading-relaxed mb-4">
                      {post.excerpt}
                    </p>

                    {/* Meta */}
                    <div className="flex items-center justify-between pt-4 border-t border-border-default">
                      <span className="text-xs text-text-label">{post.date}</span>
                      <span className="inline-flex items-center gap-1 text-xs font-medium text-warm-black group-hover:text-bright-green transition-colors">
                        Read more
                        <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <path d="M5 12h14M12 5l7 7-7 7"/>
                        </svg>
                      </span>
                    </div>
                  </div>
                </Link>
              );
            })}
          </div>
        </PageContainer>
      </section>

      {/* Newsletter CTA */}
      <NewsletterCTA />

      <TrustFooter
        headline="Apply the frameworks you&rsquo;re reading about"
        subheadline="Spin up your own Lollapalooza-style analysis and turn today&rsquo;s insights into a decision-ready brief."
        primaryCta={{ label: 'Start an analysis', href: '/analyze' }}
        secondaryCta={{ label: 'See sample reports', href: '/dashboard' }}
      />
    </div>
  );
}
