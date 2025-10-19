import React from 'react';
import Link from 'next/link';

type ButtonVariant = 'primary' | 'secondary' | 'accent' | 'ghost' | 'energy' | 'tertiary';
type ButtonSize = 'sm' | 'md' | 'lg';

interface BaseButtonProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  children: React.ReactNode;
  className?: string;
  icon?: React.ReactNode;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
}

interface ButtonAsButtonProps extends BaseButtonProps, Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, 'children'> {
  href?: never;
}

interface ButtonAsLinkProps extends BaseButtonProps {
  href: string;
  onClick?: never;
  disabled?: never;
  type?: never;
}

type ButtonProps = ButtonAsButtonProps | ButtonAsLinkProps;

const variantStyles: Record<ButtonVariant, string> = {
  // Primary CTA: 3D press effect (lift on hover, press on active)
  primary: 'bg-white text-ink-1 border-2 border-accent-green rounded-2xl font-semibold px-8 py-3 transition-all duration-200 hover:shadow-[0_8px_24px_rgba(104,222,124,0.20)] hover:-translate-y-1 active:translate-y-0 active:shadow-[0_2px_8px_rgba(104,222,124,0.12)] disabled:opacity-50 disabled:cursor-not-allowed',
  // Secondary: minimal, no 3D effect
  secondary: 'bg-white text-ink-1 border border-mesh rounded-2xl font-medium px-8 py-3 transition-colors duration-200 hover:border-accent-green disabled:opacity-50 disabled:cursor-not-allowed',
  // Accent: orange frame, 3D effect like primary
  accent: 'bg-white text-ink-1 border-2 border-accent-orange rounded-2xl font-semibold px-8 py-3 transition-all duration-200 hover:shadow-[0_8px_24px_rgba(255,107,61,0.20)] hover:-translate-y-1 active:translate-y-0 active:shadow-[0_2px_8px_rgba(255,107,61,0.12)] disabled:opacity-50 disabled:cursor-not-allowed',
  // Tertiary: no use (removed per spec)
  tertiary: 'bg-white text-ink-1 border border-mesh rounded-2xl font-medium px-8 py-3 transition-colors duration-200 hover:border-accent-green disabled:opacity-50 disabled:cursor-not-allowed',
  // Ghost: link-style only
  ghost: 'text-ink-1 bg-transparent transition-colors duration-200 hover:underline hover:underline-offset-2 hover:decoration-accent-green',
  // Energy: deprecated, same as tertiary
  energy: 'bg-white text-ink-1 border border-mesh rounded-2xl font-medium px-8 py-3 transition-colors duration-200 hover:border-accent-green disabled:opacity-50 disabled:cursor-not-allowed',
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-4 py-2 text-sm',
  md: 'px-6 py-3 text-base',
  lg: 'px-8 py-4 text-lg',
};

export function Button({
  variant = 'primary',
  size = 'md',
  children,
  className = '',
  icon,
  iconPosition = 'right',
  fullWidth = false,
  ...props
}: ButtonProps) {
  const baseStyles = 'inline-flex items-center justify-center gap-2 rounded-2xl font-semibold transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed';
  const widthStyles = fullWidth ? 'w-full' : '';
  const combinedClassName = `${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${widthStyles} ${className}`;

  const content = (
    <>
      {icon && iconPosition === 'left' && <span className="flex-shrink-0">{icon}</span>}
      <span>{children}</span>
      {icon && iconPosition === 'right' && <span className="flex-shrink-0 group-hover:translate-x-0.5 transition-transform duration-300">{icon}</span>}
    </>
  );

  if ('href' in props && props.href) {
    return (
      <Link href={props.href} className={`${combinedClassName} group`}>
        {content}
      </Link>
    );
  }

  const buttonProps = props as ButtonAsButtonProps;
  return (
    <button {...buttonProps} className={`${combinedClassName} group`}>
      {content}
    </button>
  );
}

// Arrow icon for CTAs
export const ArrowIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
    <path d="M5 12h14M12 5l7 7-7 7"/>
  </svg>
);

// Sparkle icon for energy/premium features
export const SparkleIcon = () => (
  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M11.017 2.814a1 1 0 0 1 1.966 0l1.051 5.558a2 2 0 0 0 1.594 1.594l5.558 1.051a1 1 0 0 1 0 1.966l-5.558 1.051a2 2 0 0 0-1.594 1.594l-1.051 5.558a1 1 0 0 1-1.966 0l-1.051-5.558a2 2 0 0 0-1.594-1.594l-5.558-1.051a1 1 0 0 1 0-1.966l5.558-1.051a2 2 0 0 0 1.594-1.594z"/>
  </svg>
);
