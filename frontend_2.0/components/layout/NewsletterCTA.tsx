'use client';

import { useState } from 'react';

export function NewsletterCTA({
  title = 'Get Weekly Insights',
  description = 'Mental models, strategic frameworks, and cognitive science delivered to your inbox.',
  badge = 'Stay Updated',
  onSubscribe,
  className = '',
}: {
  title?: string;
  description?: string;
  badge?: string;
  onSubscribe?: (email: string) => Promise<void> | void;
  className?: string;
}) {
  const [email, setEmail] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!email.trim()) {
      setFeedback('Add your email to subscribe.');
      return;
    }

    try {
      setIsSubmitting(true);
      setFeedback(null);
      if (onSubscribe) {
        await onSubscribe(email.trim());
      }
      setFeedback('Subscribed — check your inbox for the first digest.');
      setEmail('');
    } catch (error) {
      console.error(error);
      setFeedback('Could not subscribe right now. Please try again soon.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className={`bg-white text-ink-1 py-16 relative overflow-hidden ${className}`}>
      <div className="relative max-w-3xl mx-auto px-4 sm:px-6 lg:px-8 text-center space-y-6">
        <span className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-accent-green text-xs font-semibold text-ink-1 uppercase tracking-wide">
          {badge}
        </span>
        <h2 className="text-3xl md:text-4xl font-bold">
          {title}
        </h2>
        <p className="text-lg text-ink-2">
          {description}
        </p>
        <form
          onSubmit={handleSubmit}
          className="flex flex-col gap-3 sm:flex-row sm:items-stretch sm:justify-center max-w-xl mx-auto"
        >
          <input
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
            placeholder="your@email.com"
            className="flex-1 px-4 py-3 rounded-2xl bg-white border border-mesh text-ink-1 placeholder:text-ink-3 focus:outline-none focus:ring-2 focus:ring-accent-green focus:border-transparent"
            aria-label="Email address"
            required
          />
          <button
            type="submit"
            className="px-6 py-3 rounded-2xl bg-white text-ink-1 font-semibold border-2 border-accent-green shadow-sm hover:shadow-md transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isSubmitting}
          >
            {isSubmitting ? 'Subscribing…' : 'Subscribe'}
          </button>
        </form>
        {feedback && (
          <p className="text-sm text-ink-2">
            {feedback}
          </p>
        )}
      </div>
    </section>
  );
}
