'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { useStartAnalysis } from '@/lib/api/hooks';
import { PermanentNav } from '@/components/PermanentNav';
import { TrustFooter } from '@/components/layout/TrustFooter';
import { PageContainer } from '@/components/layout/PageContainer';

export default function AnalyzePage() {
  const router = useRouter();
  const [query, setQuery] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);

  const startAnalysis = useStartAnalysis();

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (uploadedFiles.length + files.length > 5) {
      alert('Maximum 5 files allowed');
      return;
    }
    setUploadedFiles((prev) => [...prev, ...files]);
  };

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleContinue = async () => {
    if (!query.trim()) {
      alert('Please enter a query');
      return;
    }

    try {
      const response = await startAnalysis.mutateAsync({
        query,
        interactive_mode: true, // Enable ULTRATHINK pause for user answers
      });

      router.push(`/analysis/${response.trace_id}`);
    } catch (error) {
      console.error('Error starting analysis:', error);
      alert('Failed to start analysis. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-cream-bg">
      <PermanentNav />

      {/* Main Content */}
      <main className="py-10">
        <PageContainer className="max-w-5xl">
          <div className="max-w-3xl mx-auto">
          {/* Header */}
          <div className="mb-8">
            <div className="text-[10px] uppercase tracking-[0.2em] text-text-label font-semibold mb-2">
              Start New
            </div>
            <h1 className="text-3xl font-bold mb-2 text-warm-black">Strategic Analysis</h1>
            <p className="text-base text-text-body leading-relaxed">
              Ask your strategic question. Multiple consultant perspectives in minutes.
            </p>
          </div>

          {/* Main Form Card */}
          <div className="bg-white rounded-3xl border border-border-default p-8 shadow-sm space-y-6">
            {/* Query Input */}
            <div>
              <label className="block text-sm font-semibold mb-3 text-warm-black">Your Strategic Question</label>
              <textarea
                className="w-full px-4 py-3 border border-border-default rounded-2xl bg-white text-warm-black font-normal min-h-[120px] resize-y text-sm focus:outline-none focus:ring-2 focus:ring-bright-green focus:border-transparent transition"
                placeholder="Example: Should we expand our SaaS product into the enterprise market? What are the strategic considerations?"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                maxLength={2000}
              />
              <p className="text-xs text-text-label mt-2 flex justify-between items-center">
                <span>Be specific about the decision context and constraints</span>
                <span>{query.length} / 2000</span>
              </p>
            </div>

            {/* PDF Upload */}
            <div>
              <label className="block text-sm font-semibold mb-3 text-warm-black">Supporting Documents (Optional)</label>
              <div className="flex items-center gap-3">
                <input
                  type="file"
                  id="file-upload"
                  className="hidden"
                  accept=".pdf,.docx"
                  multiple
                  onChange={handleFileUpload}
                />
                <label
                  htmlFor="file-upload"
                  className="cursor-pointer flex items-center gap-2 px-4 py-2 border border-border-default rounded-2xl hover:bg-gray-50 hover:shadow-sm transition-all duration-300 text-sm font-medium text-warm-black"
                >
                  <span className="text-lg">+</span>
                  <span>Add files</span>
                </label>
                <span className="text-xs text-text-label">
                  PDF or DOCX • Max 5 files • 10MB each
                </span>
              </div>

              {/* Uploaded Files */}
              {uploadedFiles.length > 0 && (
                <div className="mt-4 space-y-2">
                  {uploadedFiles.map((file, index) => (
                    <div
                      key={index}
                      className="flex justify-between items-center border border-border-default rounded-2xl p-3 text-sm bg-cream-bg"
                    >
                      <div>
                        <p className="font-medium text-warm-black">{file.name}</p>
                        <p className="text-xs text-text-label">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                      <button
                        onClick={() => removeFile(index)}
                        className="text-red-600 hover:underline text-xs font-medium"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Action Button */}
            <button
              onClick={handleContinue}
              disabled={!query.trim() || startAnalysis.isPending}
              className="bg-white text-warm-black w-full py-4 font-semibold border-2 border-accent-green shadow-sm hover:shadow-md transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed rounded-2xl"
            >
              {startAnalysis.isPending ? 'Starting Analysis...' : 'Start Analysis →'}
            </button>

            <p className="text-xs text-text-label text-center">
              Multiple consultant perspectives with evidence in minutes
            </p>
          </div>
          </div>
        </PageContainer>
      </main>

      <TrustFooter
        headline="Ready to test this with your own question?"
        subheadline="Answer a few prompts, attach context, and Lolla spins up multi-consultant analysis in minutes."
        primaryCta={{ label: 'Start analysis now', href: '/analysis' }}
        secondaryCta={{ label: 'See how it works', href: '/analysis/demo' }}
      />
    </div>
  );
}
