'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import { X } from 'lucide-react';

interface EvidenceItem {
  id: string;
  type: 'quote' | 'figure' | 'research' | 'model' | 'citation';
  content: string;
  source: string;
  provenance?: 'real' | 'derived';
  metadata?: Record<string, any>;
}

interface EvidencePaneProps {
  isOpen: boolean;
  width: number;
  setWidth: (w: number) => void;
  onClose: () => void;
  evidence: EvidenceItem[];
  selectedIds: string[];
  setSelectedIds: (ids: string[]) => void;
}

export default function EvidencePane({
  isOpen,
  width,
  setWidth,
  onClose,
  evidence,
  selectedIds,
  setSelectedIds,
}: EvidencePaneProps) {
  const paneRef = useRef<HTMLDivElement | null>(null);
  const dragging = useRef(false);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    dragging.current = true;
    e.preventDefault();
  }, []);

  const onMouseMove = useCallback((e: MouseEvent) => {
    if (!dragging.current || !paneRef.current) return;
    const vw = window.innerWidth;
    const newW = Math.max(420, Math.min(vw * 0.7, vw - e.clientX));
    setWidth(newW);
  }, [setWidth]);

  const onMouseUp = useCallback(() => {
    dragging.current = false;
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [onMouseMove, onMouseUp]);

  const selectedEvidence = selectedIds.length > 0
    ? evidence.filter((e) => selectedIds.includes(e.id))
    : evidence;

  const [activeTab, setActiveTab] = useState(0);

  useEffect(() => {
    setActiveTab(0);
  }, [selectedIds]);

  return (
    <div
      ref={paneRef}
      className={`fixed right-0 top-[104px] bottom-0 z-20 border-l border-gray-200 bg-gray-50 shadow-xl transition-transform ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}
      style={{ width }}
      role="complementary"
      aria-label="Evidence pane"
    >
      {/* Resize handle */}
      <div
        onMouseDown={onMouseDown}
        className="absolute left-0 top-0 z-30 -ml-1 h-full w-2 cursor-col-resize bg-transparent hover:bg-gray-300"
        title="Drag to resize"
      />

      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-gray-200 bg-white/90 px-4 py-3 backdrop-blur">
          <div className="flex items-center gap-2 text-sm font-semibold text-gray-900">
            Evidence {selectedIds.length > 0 ? `(${selectedIds.length} selected)` : `(${evidence.length} total)`}
          </div>
          <button
            onClick={onClose}
            className="rounded-full border border-gray-200 p-1.5 text-gray-700 transition hover:bg-gray-100"
            title="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Tabs for multi-select compare */}
        {selectedIds.length > 1 && (
          <div className="flex gap-2 overflow-x-auto border-b border-gray-200 bg-white px-4 py-2 scrollbar-hide">
            {selectedEvidence.map((e, idx) => (
              <button
                key={e.id}
                onClick={() => setActiveTab(idx)}
                className={`shrink-0 whitespace-nowrap rounded-full border px-3 py-1.5 text-xs font-medium transition-all ${
                  activeTab === idx
                    ? 'border-black/5 bg-gray-900 text-white shadow-sm'
                    : 'border-black/5 bg-white text-gray-700 shadow-sm hover:bg-gray-50'
                }`}
              >
                {e.id}
              </button>
            ))}
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto bg-gray-50 p-4">
          {selectedIds.length === 1 || selectedIds.length > 1 ? (
            <EvidenceCard evidence={selectedEvidence[activeTab]} />
          ) : (
            <EvidenceList evidence={evidence} setSelectedIds={setSelectedIds} />
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 bg-white/90 px-4 py-3 text-xs text-gray-700">
          <span className="mr-2 inline-flex items-center rounded-full border border-gray-200 px-2 py-0.5 font-semibold">
            Provenance
          </span>
          <span className="mr-3 text-gray-900">Real Data</span> ·{' '}
          <span className="ml-3 text-gray-700">Derived</span>
        </div>
      </div>
    </div>
  );
}

function EvidenceCard({ evidence }: { evidence: EvidenceItem }) {
  if (!evidence) return <div className="text-sm text-gray-600">No evidence selected</div>;

  return (
    <div className="rounded-3xl border border-gray-200 bg-white shadow-[0_1px_0_rgba(0,0,0,0.04),_0_12px_30px_rgba(0,0,0,0.06)]">
      <div className="border-b border-gray-200 bg-gray-50 p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2 min-w-0">
            <span className="rounded-full bg-gray-900 px-2 py-1 text-[11px] font-semibold uppercase text-white">
              {evidence.type}
            </span>
            <span className="text-xs font-mono text-gray-700 truncate max-w-[200px]" title={evidence.source}>
              {evidence.source}
            </span>
          </div>
          {evidence.provenance && (
            <span
              className={`shrink-0 rounded-full border px-2 py-1 text-xs font-semibold ${
                evidence.provenance === 'real'
                  ? 'border-black/5 bg-gray-900 text-white'
                  : 'border-black/5 bg-gray-100 text-gray-900'
              }`}
            >
              {evidence.provenance === 'real' ? 'Real' : 'Derived'}
            </span>
          )}
        </div>
      </div>
      <div className="p-4">
        <p className="whitespace-pre-wrap break-words text-sm leading-relaxed text-gray-700">{evidence.content}</p>
      </div>
    </div>
  );
}

function EvidenceList({
  evidence,
  setSelectedIds,
}: {
  evidence: EvidenceItem[];
  setSelectedIds: (ids: string[]) => void;
}) {
  return (
    <div className="space-y-3">
      {evidence.map((e) => (
        <button
          key={e.id}
          onClick={() => setSelectedIds([e.id])}
          className="w-full rounded-3xl border border-gray-200 bg-white text-left shadow-sm transition-all hover:border-gray-300 hover:shadow-[0_6px_20px_rgba(0,0,0,0.08)]"
        >
          <div className="border-b border-gray-200 bg-gray-50 p-2">
            <div className="flex flex-wrap items-center gap-2 min-w-0">
              <span className="rounded-full bg-gray-900 px-2 py-0.5 text-[11px] font-semibold uppercase text-white">
                {e.type}
              </span>
              <span className="text-xs font-mono text-gray-700 truncate" title={e.source}>{e.source}</span>
            </div>
          </div>
          <div className="p-3">
            <p className="line-clamp-3 break-words text-sm leading-relaxed text-gray-700">{e.content}</p>
          </div>
        </button>
      ))}
      {evidence.length === 0 && (
        <div className="text-sm text-gray-600 text-center py-8">No evidence available</div>
      )}
    </div>
  );
}
