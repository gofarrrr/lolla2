'use client';

import React from 'react';

interface ProgressDotsProps {
  /** Number of dots to display (default: 3) */
  count?: number;
  /** Size of dots: 'sm' (4px), 'md' (6px), 'lg' (8px) */
  size?: 'sm' | 'md' | 'lg';
  /** Color variant */
  variant?: 'green' | 'persimmon' | 'gray';
  /** Animation speed in ms (default: 600) */
  speed?: number;
}

const sizeStyles = {
  sm: 'w-1 h-1',
  md: 'w-1.5 h-1.5',
  lg: 'w-2 h-2',
};

const colorStyles = {
  green: 'bg-bright-green',
  persimmon: 'bg-soft-persimmon',
  gray: 'bg-text-label',
};

export function ProgressDots({
  count = 3,
  size = 'md',
  variant = 'green',
  speed = 600
}: ProgressDotsProps) {
  return (
    <div className="inline-flex items-center gap-1.5">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className={`${sizeStyles[size]} ${colorStyles[variant]} rounded-full`}
          style={{
            animation: `pulse ${speed}ms ease-in-out infinite`,
            animationDelay: `${i * (speed / count)}ms`,
          }}
        />
      ))}
      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.9); }
          50% { opacity: 1; transform: scale(1.1); }
        }
      `}</style>
    </div>
  );
}

// Thinking indicator - three animated dots
export function ThinkingIndicator() {
  return (
    <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-cream-bg border border-border-default">
      <span className="text-xs font-medium text-text-body">Thinking</span>
      <ProgressDots size="sm" variant="green" />
    </div>
  );
}

// Processing indicator with label
export function ProcessingIndicator({ label = 'Processing' }: { label?: string }) {
  return (
    <div className="inline-flex items-center gap-2 px-4 py-2 rounded-2xl bg-white border border-border-default shadow-sm">
      <ProgressDots size="md" variant="green" />
      <span className="text-sm font-medium text-warm-black">{label}</span>
    </div>
  );
}
