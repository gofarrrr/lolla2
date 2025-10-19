import Link from 'next/link';
import { Button } from '@/components/ui/Button';

interface TrustFooterProps {
  headline?: string;
  subheadline?: string;
  primaryCta?: {
    label: string;
    href: string;
  };
  secondaryCta?: {
    label: string;
    href: string;
  };
  highlights?: string[];
  className?: string;
}

const defaultHighlights = [
  'SOC 2 Type I controls • opt out of model training any time',
  'Enterprise SSO & SCIM provisioning ready on day one',
  'Live status page, SLA reporting, and audit trails included',
];

export function TrustFooter({
  headline = 'Ready for calmer, more strategic working sessions?',
  subheadline = 'Lolla keeps the glass-box evidence trail while your team focuses on decisions. No bots, no mystery AI.',
  primaryCta = { label: 'Get started free', href: '/signup' },
  secondaryCta = { label: 'Talk to us', href: '/contact' },
  highlights = defaultHighlights,
  className = '',
}: TrustFooterProps) {
  return (
    <footer className={`bg-white text-ink-1 border-t border-border-default ${className}`}>
      <div className="mx-auto max-w-7xl px-6 py-16 space-y-12">
        <div className="grid gap-8 md:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)] md:items-center">
          <div className="space-y-4">
            <span className="inline-flex items-center gap-2 rounded-full border border-accent-green px-4 py-2 text-xs font-semibold uppercase tracking-[0.2em] text-ink-1 bg-white">
              Cognitive intelligence
            </span>
            <h2 className="text-3xl font-semibold leading-tight">
              {headline}
            </h2>
            <p className="text-base leading-relaxed text-ink-2">
              {subheadline}
            </p>
          </div>
          <div className="flex flex-col gap-3 md:items-end">
            <Button
              href={primaryCta.href}
              variant="primary"
              size="md"
              iconPosition="right"
            >
              {primaryCta.label}
            </Button>
            <Button
              href={secondaryCta.href}
              variant="ghost"
              size="md"
            >
              {secondaryCta.label}
            </Button>
          </div>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <div className="space-y-4">
            <div className="text-xs font-semibold uppercase tracking-[0.2em] text-ink-3">
              Trusted by operators at
            </div>
            <div className="flex flex-wrap items-center gap-3 text-sm text-ink-2">
              {['Atlas Ops', 'Beacon Labs', 'Northwind Ventures', 'Helix Cloud'].map((brand) => (
                <span
                  key={brand}
                  className="inline-flex items-center justify-center rounded-full border border-border-default px-4 py-2 text-sm text-ink-2 bg-white"
                >
                  {brand}
                </span>
              ))}
            </div>
          </div>
          <ul className="space-y-3 text-sm leading-relaxed text-ink-2">
            {highlights.map((item) => (
              <li key={item} className="flex gap-3">
                <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-accent-green" />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>

        <div className="flex flex-col gap-4 border-t border-border-default pt-6 text-xs text-ink-3 sm:flex-row sm:items-center sm:justify-between">
          <span>© {new Date().getFullYear()} Lolla. Built for serious thinkers.</span>
          <div className="flex flex-wrap gap-4">
            <Link href="/docs/security" className="hover:text-ink-1 transition-colors">
              Security
            </Link>
            <Link href="/status" className="hover:text-ink-1 transition-colors">
              Status
            </Link>
            <Link href="/legal/terms" className="hover:text-ink-1 transition-colors">
              Terms
            </Link>
            <Link href="/legal/privacy" className="hover:text-ink-1 transition-colors">
              Privacy
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
