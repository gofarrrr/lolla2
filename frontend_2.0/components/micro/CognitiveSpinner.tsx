import clsx from 'clsx';

type SpinnerSize = 'sm' | 'md' | 'lg';

const sizeMap: Record<SpinnerSize, { stack: string; label: string }> = {
  sm: { stack: 'h-10 w-10', label: 'text-[10px]' },
  md: { stack: 'h-12 w-12', label: 'text-[11px]' },
  lg: { stack: 'h-14 w-14', label: 'text-xs' },
};

interface CognitiveSpinnerProps {
  label?: string;
  size?: SpinnerSize;
  className?: string;
}

/**
 * Cognitive Spinner — layered shapes that lift in sequence.
 * Symbolises Lolla’s stacked reasoning and mental models.
 */
export function CognitiveSpinner({
  label = 'Assembling intelligence',
  size = 'md',
  className,
}: CognitiveSpinnerProps) {
  const dimensions = sizeMap[size];

  return (
    <div
      className={clsx(
        'cognitive-spinner',
        className
      )}
      role="status"
      aria-live="polite"
    >
      <div className={clsx('cognitive-spinner-stack', dimensions.stack)}>
        <div className="cognitive-layer cognitive-layer--base" />
        <div className="cognitive-layer cognitive-layer--middle" />
        <div className="cognitive-layer cognitive-layer--top">
          <span className="cognitive-layer-glyph" />
        </div>
      </div>
      {label && (
        <span className={clsx('cognitive-spinner-label', dimensions.label)}>
          {label}
        </span>
      )}
    </div>
  );
}
