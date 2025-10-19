'use client';

import React from 'react';

interface ConfidenceMeterProps {
  /** Confidence score from 0-100 */
  score: number;
  /** Size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Show numeric label */
  showLabel?: boolean;
  /** Label text (default: "Confidence") */
  label?: string;
}

const heightStyles = {
  sm: 'h-1.5',
  md: 'h-2',
  lg: 'h-3',
};

export function ConfidenceMeter({
  score,
  size = 'md',
  showLabel = true,
  label = 'Confidence'
}: ConfidenceMeterProps) {
  const clampedScore = Math.max(0, Math.min(100, score));

  // Color based on confidence level
  const getColor = (score: number) => {
    if (score >= 80) return 'bg-bright-green';
    if (score >= 60) return 'bg-yellow-500';
    if (score >= 40) return 'bg-soft-persimmon';
    return 'bg-red-500';
  };

  return (
    <div className="w-full">
      {showLabel && (
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-xs font-medium text-text-label uppercase tracking-wider">{label}</span>
          <span className="text-xs font-bold text-warm-black">{clampedScore}%</span>
        </div>
      )}
      <div className={`w-full bg-cream-bg rounded-full overflow-hidden ${heightStyles[size]}`}>
        <div
          className={`${heightStyles[size]} ${getColor(clampedScore)} transition-all duration-500 ease-out rounded-full`}
          style={{ width: `${clampedScore}%` }}
        />
      </div>
    </div>
  );
}

// Quality indicator with icon and meter
export function QualityIndicator({ score, label = 'Quality' }: { score: number; label?: string }) {
  const getIcon = (score: number) => {
    if (score >= 80) return '✓';
    if (score >= 60) return '~';
    return '!';
  };

  const getIconColor = (score: number) => {
    if (score >= 80) return 'text-bright-green bg-accent-success';
    if (score >= 60) return 'text-yellow-700 bg-yellow-50';
    return 'text-orange-700 bg-accent-energy';
  };

  return (
    <div className="inline-flex items-center gap-3 px-4 py-2.5 rounded-2xl bg-white border border-border-default shadow-sm">
      <div className={`w-6 h-6 rounded-full flex items-center justify-center font-bold text-sm ${getIconColor(score)}`}>
        {getIcon(score)}
      </div>
      <div className="flex-1 min-w-[120px]">
        <ConfidenceMeter score={score} size="sm" label={label} />
      </div>
    </div>
  );
}

// Simplified confidence pill
export function ConfidencePill({ score }: { score: number }) {
  const getColor = (score: number) => {
    if (score >= 80) return 'bg-accent-success text-green-700 border-bright-green';
    if (score >= 60) return 'bg-yellow-50 text-yellow-700 border-yellow-500';
    return 'bg-accent-energy text-orange-700 border-accent-energy-border';
  };

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-xs font-semibold ${getColor(score)}`}>
      <span className="w-1.5 h-1.5 rounded-full bg-current"></span>
      {score}%
    </span>
  );
}
