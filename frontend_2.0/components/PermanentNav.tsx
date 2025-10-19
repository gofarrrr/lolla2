'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useGlassBoxStore } from '@/lib/state/glassBox';

export function PermanentNav() {
  const pathname = usePathname();
  const { glassBox, toggle } = useGlassBoxStore();

  const isActive = (path: string) => {
    if (path === '/dashboard') {
      return pathname === '/dashboard' || pathname?.startsWith('/analysis/');
    }
    return pathname?.startsWith(path);
  };

  return (
    <header className="border-b border-border-default bg-white/95 backdrop-blur-md sticky top-0 z-50">
      <div className="container-wide">
        <div className="flex items-center justify-between h-14">
          {/* Logo and primary nav - concentrated layout */}
          <div className="flex items-center gap-6">
            <Link
              href="/dashboard"
              className="text-lg font-bold text-ink-1 transition-colors duration-200 hover:underline hover:decoration-accent-green"
            >
              lolla
            </Link>

            <div className="h-4 w-px bg-border-default hidden sm:block" />

            <nav className="hidden sm:flex items-center gap-1">
              <Link
                href="/dashboard"
                className={`px-3 py-1.5 text-sm font-medium transition-all duration-200 border-b-2 ${
                  isActive('/dashboard')
                    ? 'text-ink-1 border-accent-green'
                    : 'text-ink-2 border-transparent hover:text-ink-1 hover:border-accent-green'
                }`}
              >
                Reports
              </Link>
              <Link
                href="/academy"
                className={`px-3 py-1.5 text-sm font-medium transition-all duration-200 border-b-2 ${
                  isActive('/academy')
                    ? 'text-ink-1 border-accent-green'
                    : 'text-ink-2 border-transparent hover:text-ink-1 hover:border-accent-green'
                }`}
              >
                Academy
              </Link>
              <Link
                href="/blog"
                className={`px-3 py-1.5 text-sm font-medium transition-all duration-200 border-b-2 ${
                  isActive('/blog')
                    ? 'text-ink-1 border-accent-green'
                    : 'text-ink-2 border-transparent hover:text-ink-1 hover:border-accent-green'
                }`}
              >
                Blog
              </Link>
            </nav>
          </div>

          {/* Right side actions - include Glass Box toggle */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              aria-label="Toggle Glass Box"
              onClick={toggle}
              className={`flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg transition-all duration-200 border ${
                glassBox
                  ? 'bg-[#1E1E1E] text-[#D4D4D4] border-[#404040]'
                  : 'bg-cream-bg text-text-body hover:text-warm-black border-border-default'
              }`}
              title={`Glass Box: ${glassBox ? 'ON' : 'OFF'}`}
            >
              <span>Glass Box:</span>
              <span className={`inline-flex items-center justify-center w-5 h-5 rounded ${glassBox ? 'bg-[#4EC9B0]' : 'bg-gray-300'}`} />
              <span className="hidden sm:inline">{glassBox ? 'ON' : 'OFF'}</span>
            </button>
            <button className="px-3 py-1.5 text-sm font-medium text-text-body hover:text-warm-black hover:bg-cream-bg rounded-lg transition-all duration-200">
              Settings
            </button>
            <button className="w-8 h-8 rounded-full bg-gradient-to-br from-bright-green to-green-hover text-white flex items-center justify-center font-semibold text-sm shadow-sm hover:shadow-md transition-shadow duration-200">
              M
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
