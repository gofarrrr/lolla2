import clsx from 'clsx';

interface StageProgressProps {
  stages: string[];
  currentStage: number; // 0-indexed
  className?: string;
}

type StageState = 'complete' | 'active' | 'upcoming';

const stateStyles: Record<StageState, { card: string; badge: string; meta: string; beam?: string; badgeGlow?: string }> = {
  complete: {
    card: 'border-brand-lime/60 bg-brand-lime/10 shadow-sm stage-card-complete',
    badge: 'bg-brand-lime text-white',
    meta: 'text-brand-lime',
    beam: 'stage-beam-complete',
  },
  active: {
    card: 'border-brand-persimmon bg-white shadow-lg ring-1 ring-brand-persimmon/30 stage-card-active',
    badge: 'bg-brand-persimmon text-white',
    meta: 'text-brand-persimmon',
    beam: 'stage-beam-active',
    badgeGlow: 'stage-badge-glow',
  },
  upcoming: {
    card: 'border-dashed border-border-default bg-white',
    badge: 'bg-neutral-200 text-text-label',
    meta: 'text-text-label',
  },
};

const stateLabel: Record<StageState, string> = {
  complete: 'Complete',
  active: 'In progress',
  upcoming: 'Up next',
};

export default function StageProgress({ stages, currentStage, className }: StageProgressProps) {
  return (
    <div className={clsx('flex flex-col gap-4', className)}>
      <div className="flex items-stretch gap-3 overflow-x-auto pb-2">
        {stages.map((stage, index) => {
          const state: StageState =
            index < currentStage ? 'complete' : index === currentStage ? 'active' : 'upcoming';
          const styles = stateStyles[state];

          return (
            <div key={stage} className="relative flex min-w-[190px] flex-shrink-0 flex-col">
              <div
                className={clsx(
                  'relative overflow-hidden rounded-2xl border px-4 py-3 transition-all duration-300',
                  'backdrop-blur-sm',
                  styles.card
                )}
              >
                {styles.beam && (
                  <span className={clsx('stage-beam absolute inset-0', styles.beam)} aria-hidden />
                )}
                <div className="flex items-start justify-between gap-3">
                  <span
                    className={clsx(
                      'flex h-8 w-8 items-center justify-center rounded-full text-sm font-semibold shadow-sm stage-badge',
                      styles.badge,
                      styles.badgeGlow ?? ''
                    )}
                    aria-hidden
                  >
                    {index < currentStage ? '✓' : index + 1}
                  </span>
                  <div className="flex flex-col items-end gap-1">
                    <span className={clsx('text-[11px] font-semibold uppercase tracking-[0.2em]', styles.meta)}>
                      {stateLabel[state]}
                    </span>
                    <span className="text-xs font-medium text-text-label">
                      Stage {index + 1}
                    </span>
                  </div>
                </div>
                <div className="mt-3 text-sm font-semibold leading-tight text-warm-black">
                  {stage}
                </div>
              </div>
              {index < stages.length - 1 && (
                <div className="absolute right-[-20px] top-1/2 hidden h-px w-10 translate-y-[-50%] bg-gradient-to-r from-border-default via-border-default to-transparent md:block" />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
