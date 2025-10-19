'use client';

import { use } from 'react';
import Link from 'next/link';
import { usePitchResults, usePitchStatus } from '@/lib/api/hooks';

export default function PitchDeckPage({ params }: { params: Promise<{ pitch_id: string }> }) {
  const { pitch_id } = use(params);
  const { data: status } = usePitchStatus(pitch_id);
  const { data: results, isLoading } = usePitchResults(
    status?.status === 'completed' ? pitch_id : null
  );

  if (isLoading || status?.status !== 'completed') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg mb-2">Generating pitch deck...</p>
          <p className="text-gray-600">{status?.current_phase || 'Processing'}</p>
        </div>
      </div>
    );
  }

  if (!results) return null;

  return (
    <div className="min-h-screen bg-white">
      <header className="border-b-2 border-black">
        <div className="container-wide py-4">
          <Link href="/dashboard" className="text-2xl font-bold">Lolla</Link>
        </div>
      </header>

      <main className="container-content py-12">
        <h1 className="text-4xl font-bold mb-2">{results.pitch_deck.title}</h1>
        <p className="text-lg text-gray-600 mb-8">{results.pitch_deck.executive_summary}</p>

        <h2 className="text-2xl font-bold mb-6">Slide Deck</h2>
        <div className="space-y-6">
          {results.pitch_deck.slides.map((slide, index) => (
            <div key={index} className="border-4 border-black p-8 bg-gray-100">
              <div className="flex justify-between items-start mb-4">
                <span className="text-sm text-gray-600">Slide {slide.slide_number}</span>
                <span className="px-2 py-1 border-2 border-black text-xs font-bold">
                  {slide.slide_type.toUpperCase()}
                </span>
              </div>
              <h3 className="text-3xl font-bold mb-4">{slide.headline}</h3>
              <ul className="space-y-2 mb-6">
                {slide.key_points.map((point, i) => (
                  <li key={i} className="flex items-start">
                    <span className="text-accent mr-2">▪</span>
                    <span className="text-lg">{point}</span>
                  </li>
                ))}
              </ul>
              <details className="border-t-2 border-black pt-4">
                <summary className="cursor-pointer text-sm text-gray-600">Speaker Notes</summary>
                <p className="mt-2 text-sm">{slide.speaker_notes}</p>
              </details>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
