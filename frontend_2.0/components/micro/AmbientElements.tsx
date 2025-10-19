'use client';

import React from 'react';

// Soft gradient orb for hero sections (Granola-inspired)
export function FloatingOrb({ variant = 'green', position = 'center' }: {
  variant?: 'green' | 'persimmon' | 'mixed';
  position?: 'center' | 'left' | 'right';
}) {
  const gradients = {
    green: 'radial-gradient(circle at 50% 50%, rgba(104,222,124,0.08) 0%, rgba(104,222,124,0.03) 35%, rgba(0,0,0,0) 70%)',
    persimmon: 'radial-gradient(circle at 50% 50%, rgba(255,107,61,0.06) 0%, rgba(255,107,61,0.02) 35%, rgba(0,0,0,0) 70%)',
    mixed: 'radial-gradient(circle at 50% 50%, rgba(104,222,124,0.06) 0%, rgba(255,107,61,0.04) 40%, rgba(0,0,0,0) 70%)',
  };

  const positions = {
    center: 'items-center justify-center',
    left: 'items-center justify-start',
    right: 'items-center justify-end',
  };

  return (
    <div className={`absolute inset-0 -z-10 flex ${positions[position]}`} style={{ isolation: 'isolate' }}>
      <div
        className="w-[1200px] h-[1200px] rounded-full"
        style={{
          background: gradients[variant],
          filter: 'blur(40px)',
          opacity: 0.6,
        }}
      />
    </div>
  );
}

// Subtle grid overlay for sections
export function GridOverlay({ opacity = 0.03 }: { opacity?: number }) {
  return (
    <div
      className="absolute inset-0 -z-10"
      style={{
        backgroundImage: `
          linear-gradient(to right, rgba(26,26,26,${opacity}) 1px, transparent 1px),
          linear-gradient(to bottom, rgba(26,26,26,${opacity}) 1px, transparent 1px)
        `,
        backgroundSize: '32px 32px',
      }}
    />
  );
}

// Corner flourish accent
export function CornerFlourish({ position = 'top-right', variant = 'green' }: {
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  variant?: 'green' | 'persimmon';
}) {
  const colors = {
    green: '#68DE7C',
    persimmon: '#FF6B3D',
  };

  const positions = {
    'top-left': 'top-0 left-0',
    'top-right': 'top-0 right-0',
    'bottom-left': 'bottom-0 left-0',
    'bottom-right': 'bottom-0 right-0',
  };

  return (
    <div className={`absolute ${positions[position]} w-24 h-24 opacity-20`}>
      <svg viewBox="0 0 100 100" className="w-full h-full">
        <path
          d="M 0 0 Q 50 0 100 50 Q 100 0 100 0 Z"
          fill={colors[variant]}
          opacity="0.15"
        />
      </svg>
    </div>
  );
}

// Soft gradient divider between sections
export function GradientDivider({ variant = 'green-to-persimmon' }: {
  variant?: 'green-to-persimmon' | 'persimmon-to-green' | 'subtle';
}) {
  const gradients = {
    'green-to-persimmon': 'linear-gradient(to right, rgba(104,222,124,0.15), rgba(255,107,61,0.15))',
    'persimmon-to-green': 'linear-gradient(to right, rgba(255,107,61,0.15), rgba(104,222,124,0.15))',
    'subtle': 'linear-gradient(to right, rgba(229,231,235,0.5), rgba(229,231,235,0.1), rgba(229,231,235,0.5))',
  };

  return (
    <div className="w-full h-px my-12" style={{ background: gradients[variant] }} />
  );
}

// Wave pattern for section backgrounds
export function WavePattern({ variant = 'top', opacity = 0.05 }: {
  variant?: 'top' | 'bottom';
  opacity?: number;
}) {
  const path = variant === 'top'
    ? 'M0,20 Q250,0 500,20 T1000,20 L1000,0 L0,0 Z'
    : 'M0,0 Q250,20 500,0 T1000,0 L1000,20 L0,20 Z';

  return (
    <svg
      className="absolute w-full"
      style={{ [variant]: 0, opacity }}
      viewBox="0 0 1000 20"
      preserveAspectRatio="none"
    >
      <path d={path} fill="currentColor" className="text-bright-green" />
    </svg>
  );
}

// Selection bar indicator for active cards/items
export function SelectionBar({ position = 'left' }: { position?: 'left' | 'top' }) {
  if (position === 'top') {
    return (
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-bright-green to-soft-persimmon rounded-t-3xl" />
    );
  }

  return (
    <div className="absolute top-0 left-0 bottom-0 w-1 bg-gradient-to-b from-bright-green to-soft-persimmon rounded-l-3xl" />
  );
}

// Hover glow effect (apply to cards)
export function HoverGlow() {
  return (
    <div className="absolute inset-0 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 -z-10">
      <div className="absolute inset-0 bg-gradient-to-br from-bright-green/5 to-soft-persimmon/5 rounded-3xl blur-xl" />
    </div>
  );
}
