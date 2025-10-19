'use client';

import { use } from 'react';
import Link from 'next/link';
import { useIdeaflowResults, useIdeaflowStatus } from '@/lib/api/hooks';
import { downloadAsFile } from '@/lib/utils';

export default function IdeaflowResultsPage({ params }: { params: Promise<{ sprint_id: string }> }) {
  const { sprint_id } = use(params);
  const { data: status } = useIdeaflowStatus(sprint_id);
  const { data: results, isLoading } = useIdeaflowResults(
    status?.status === 'completed' ? sprint_id : null
  );

  const handleExport = () => {
    if (!results) return;
    const markdown = `# Ideaflow Sprint Results\n\n**Problem**: ${results.problem_statement}\n\n**Generated Ideas**: ${results.generated_ideas_count}\n\n${results.clusters.map((cluster, i) => `## Cluster ${i + 1}: ${cluster.theme}\n\n${cluster.description}\n\n**Selected Idea**: ${cluster.selected_idea}\n\n### Experiment Design\n- **Hypothesis**: ${cluster.experiment.hypothesis}\n- **Test**: ${cluster.experiment.test_design}\n`).join('\n\n')}`;
    downloadAsFile(markdown, `ideaflow-${sprint_id}.md`, 'text/markdown');
  };

  if (isLoading || status?.status !== 'completed') {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-lg mb-2">Generating ideas...</p>
          <p className="text-gray-600">{status?.progress_percentage || 0}% complete</p>
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
        <h1 className="text-4xl font-bold mb-2">Ideaflow Sprint Results</h1>
        <p className="text-lg text-gray-600 mb-2">{results.problem_statement}</p>
        <p className="text-success mb-8">✓ Generated {results.generated_ideas_count} ideas</p>

        {results.clusters.map((cluster, index) => (
          <section key={index} className="mb-12 card">
            <h2 className="text-2xl font-bold mb-2">{cluster.theme}</h2>
            <p className="text-gray-600 mb-4">{cluster.description}</p>

            <div className="border-t-2 border-black pt-4 mb-4">
              <h3 className="font-semibold mb-2">Selected Idea</h3>
              <p className="text-lg">{cluster.selected_idea}</p>
            </div>

            <details className="border-t-2 border-black pt-4">
              <summary className="cursor-pointer font-semibold mb-2">Experiment Design</summary>
              <div className="space-y-2 mt-3">
                <p><strong>Hypothesis:</strong> {cluster.experiment.hypothesis}</p>
                <p><strong>Test Design:</strong> {cluster.experiment.test_design}</p>
                <p><strong>Duration:</strong> {cluster.experiment.duration}</p>
              </div>
            </details>
          </section>
        ))}

        <button onClick={handleExport} className="btn btn-accent w-full">
          Export Full Report →
        </button>
      </main>
    </div>
  );
}
