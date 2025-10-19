'use client';

import React from 'react';

export type ZoomLevel = 'L0' | 'L1' | 'L2' | 'L3';

interface ZoomControlProps {
  zoom: ZoomLevel;
  setZoom: (z: ZoomLevel) => void;
}

// Map technical levels to user-friendly perspective names
const PERSPECTIVE_LABELS: Record<ZoomLevel, { name: string; icon: string; description: string }> = {
  'L0': { name: 'Decision', icon: '🎯', description: 'Executive summary and key decisions' },
  'L1': { name: 'Strategy', icon: '📊', description: 'Strategic recommendations and roadmap' },
  'L2': { name: 'Process', icon: '🔍', description: 'How we analyzed this problem' },
  'L3': { name: 'Evidence', icon: '📚', description: 'Research sources and citations' },
};

export default function ZoomControl({ zoom, setZoom }: ZoomControlProps) {
  const levels: ZoomLevel[] = ['L0', 'L1', 'L2', 'L3'];

  return (
    <div className="pointer-events-auto fixed bottom-4 left-4 z-20">
      <div className="inline-flex overflow-hidden rounded-xl border border-gray-200 bg-white shadow-[0_6px_20px_rgba(0,0,0,0.08)]">
        {levels.map((l) => {
          const perspective = PERSPECTIVE_LABELS[l];
          return (
            <button
              key={l}
              onClick={() => setZoom(l)}
              className={`px-4 py-2 text-sm font-bold border-r last:border-r-0 border-gray-200 transition-all ${
                zoom === l ? 'bg-gray-900 text-white' : 'bg-white text-gray-900 hover:bg-gray-50'
              }`}
              aria-pressed={zoom === l}
              title={perspective.description}
            >
              <div className="flex items-center gap-2">
                <span className="text-base">{perspective.icon}</span>
                <span>{perspective.name}</span>
              </div>
            </button>
          );
        })}
      </div>
      <div className="mt-2 text-xs text-gray-600 text-center max-w-md">
        {PERSPECTIVE_LABELS[zoom].description}
      </div>
    </div>
  );
}
