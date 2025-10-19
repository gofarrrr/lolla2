import React from 'react';

type BadgeVariant = 'success' | 'pending' | 'energy' | 'info' | 'neutral';
type BadgeSize = 'sm' | 'md' | 'lg';

interface StatusBadgeProps {
  variant?: BadgeVariant;
  size?: BadgeSize;
  children: React.ReactNode;
  icon?: React.ReactNode;
  pulse?: boolean;
}

const variantStyles: Record<BadgeVariant, string> = {
  success: 'bg-white border-accent-green text-warm-black',
  pending: 'bg-white border-mesh text-ink-2',
  energy: 'bg-white border-accent-orange text-warm-black',
  info: 'bg-white border-accent-yellow text-warm-black',
  neutral: 'bg-white border-mesh text-ink-1',
};

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'px-2 py-1 text-xs',
  md: 'px-3 py-1.5 text-sm',
  lg: 'px-4 py-2 text-base',
};

export function StatusBadge({
  variant = 'neutral',
  size = 'md',
  children,
  icon,
  pulse = false
}: StatusBadgeProps) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-medium ${variantStyles[variant]} ${sizeStyles[size]}`}>
      {icon && (
        <span className={`flex-shrink-0 ${pulse ? 'animate-pulse' : ''}`}>
          {icon}
        </span>
      )}
      {children}
    </span>
  );
}

// Check mark icon for completed states
export const CheckIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
  </svg>
);

// Clock icon for pending states
export const ClockIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <circle cx="12" cy="12" r="10" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6l4 2" />
  </svg>
);

// Lightning icon for energy/premium features
export const LightningIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
    <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" />
  </svg>
);

// Sparkle icon for AI/premium features
export const SparkleIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z"/>
  </svg>
);

// Pulse dot indicator for active states
export function PulseDot({ variant = 'success' }: { variant?: 'success' | 'energy' | 'info' }) {
  const colorMap = {
    success: 'bg-accent-green',
    energy: 'bg-accent-orange',
    info: 'bg-accent-yellow',
  };

  return (
    <span className="relative flex h-2 w-2">
      <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${colorMap[variant]} opacity-75`}></span>
      <span className={`relative inline-flex rounded-full h-2 w-2 ${colorMap[variant]}`}></span>
    </span>
  );
}
