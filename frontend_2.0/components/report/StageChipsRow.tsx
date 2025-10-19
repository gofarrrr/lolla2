'use client';

import React from 'react';

export type StageId =
  | 'all'
  | 'executive'
  | 'recommendations'
  | 'consultants'
  | 'synthesis'
  | 'research';

interface StageDef {
  id: StageId;
  label: string;
  count?: number;
}

interface StageChipsRowProps {
  stages: StageDef[];
  activeStage: StageId;
  onStageClick: (id: StageId) => void;
}

export default function StageChipsRow({ stages, activeStage, onStageClick }: StageChipsRowProps) {
  return (
    <div
      className="sticky top-0 z-30 border-b border-gray-200 bg-white/95 backdrop-blur-md"
      style={{ top: '68px' }}
    >
      <div className="mx-auto flex w-full max-w-[1400px] items-center gap-2 overflow-x-auto px-4 py-3 scrollbar-hide">
        {stages.map((s) => (
          <button
            key={s.id}
            onClick={() => onStageClick(s.id)}
            className={`shrink-0 whitespace-nowrap rounded-full border px-4 py-2 text-sm font-medium transition-all ${
              activeStage === s.id
                ? 'border-black/5 bg-gray-900 text-white shadow-[0_6px_20px_rgba(0,0,0,0.12)]'
                : 'border-black/5 bg-white text-gray-700 shadow-sm hover:bg-gray-50 hover:text-gray-900'
            }`}
          >
            {s.label}
            {typeof s.count === 'number' && s.count > 0 && (
              <span className="ml-2 inline-flex items-center rounded-full bg-gray-800 px-2 py-0.5 text-[10px] font-semibold text-white">
                {s.count}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
