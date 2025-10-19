import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      // Neutral-first design with accent frames only
      colors: {
        // Neutrals
        'warm-black': '#0F172A',             // Primary ink
        'ink-1': '#0F172A',                  // Headings/body
        'ink-2': '#334155',                  // Muted
        'ink-3': '#64748B',                  // Subtle
        'canvas': '#F3F4F6',                 // Page background
        'surface': '#FFFFFF',                // Cards
        'mesh': '#C9CED6',                   // Dividers/wireframe

        // Legacy aliases (kept for gradual migration)
        'cream-bg': '#F3F4F6',
        'off-white': '#FAFAF8',
        'white': '#FFFFFF',
        'border-default': '#C9CED6',
        'border-focus': '#68DE7C',
        'text-heading': '#0F172A',
        'text-body': '#334155',
        'text-label': '#64748B',

        // Accents (frames only)
        'accent-green': '#68DE7C',
        'accent-orange': '#FF6B3D',
        'accent-yellow': '#EAE45B',
        // Backward-compat tokens
        'bright-green': '#68DE7C',
        'green-hover': '#5ACC6E',
        'soft-persimmon': '#FF6B3D',
        'brand-lime': '#68DE7C',
        'brand-persimmon': '#FF6B3D',
        'brand-yellow': '#EAE45B',
        'accent-primary': '#68DE7C',

        // Utility scales
        neutral: {
          50: '#F9FAFB',
          100: '#F3F4F6',
          200: '#E5E7EB',
          300: '#D1D5DB',
          400: '#9CA3AF',
          500: '#6B7280',
          600: '#4B5563',
          700: '#374151',
          800: '#1F2937',
          900: '#111827',
        },
        gray: {
          50: '#F9FAFB',
          100: '#F3F4F6',
          200: '#E5E7EB',
          300: '#D1D5DB',
          400: '#9CA3AF',
          500: '#6B7280',
          600: '#4B5563',
          700: '#374151',
          800: '#1F2937',
          900: '#0F172A',
        },
        success: '#22C55E',
        warning: '#F59E0B',
        error: '#EF4444',
      },

      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        mono: ['IBM Plex Mono', 'SF Mono', 'Menlo', 'monospace'],
      },

      fontSize: {
        xs: '12px',
        sm: '14px',
        base: '16px',
        lg: '18px',
        xl: '24px',
        '2xl': '32px',
        '3xl': '48px',
        '4xl': '64px',
      },

      spacing: {
        '0': '0',
        '1': '8px',
        '2': '16px',
        '3': '24px',
        '4': '32px',
        '5': '40px',
        '6': '48px',
        '8': '64px',
        '10': '80px',
        '12': '96px',
        '16': '128px',
        '20': '160px',
      },

      borderWidth: {
        DEFAULT: '2px',
        '0': '0',
        '1': '1px',
        '2': '2px',
        '3': '3px',
        '4': '4px',
      },

      borderRadius: {
        none: '0',
        sm: '8px',
        DEFAULT: '12px',
        md: '16px',
        lg: '20px',
        xl: '24px',
        '2xl': '28px',
        '3xl': '32px',
        full: '9999px',
      },

      maxWidth: {
        'content': '720px',
        'reading': '680px',
        'container': '1200px',
        'wide': '1440px',
      },

      lineHeight: {
        tight: '1.25',
        normal: '1.6',
        relaxed: '1.8',
        loose: '2.0',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),
  ],
};

export default config;
