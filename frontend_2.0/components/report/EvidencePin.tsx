'use client';

import React, { useState } from 'react';

interface EvidencePinProps {
  count: number;
  onClick: (opts?: { multi?: boolean }) => void;
  preview?: string;
}

export default function EvidencePin({ count, onClick, preview }: EvidencePinProps) {
  const [hover, setHover] = useState(false);

  if (count === 0) return null;

  return (
    <div className="relative inline-block">
      <button
        onClick={(e) => onClick({ multi: e.metaKey || e.ctrlKey })}
        onMouseEnter={() => setHover(true)}
        onMouseLeave={() => setHover(false)}
        className="border border-gray-200 rounded bg-white px-2 py-0.5 text-xs font-bold hover:bg-gray-900 hover:text-white transition-colors"
        title="Open evidence"
      >
        E:{count}
      </button>
      {hover && preview && (
        <div className="absolute left-0 top-full z-10 mt-1 w-72 max-w-[80vw] border border-gray-200 rounded-xl bg-white p-2 text-[12px] shadow-[0_6px_20px_rgba(0,0,0,0.08)]">
          {preview}
        </div>
      )}
    </div>
  );
}
