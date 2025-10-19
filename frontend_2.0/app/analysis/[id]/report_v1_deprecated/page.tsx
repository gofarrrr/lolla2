'use client';

import { use, useState } from 'react';
import Link from 'next/link';
import { useFinalReport, useGeneratePitch, useStartIdeaflowSprint, useStartCopywriterJob } from '@/lib/api/hooks';
import { downloadAsFile, copyToClipboard } from '@/lib/utils';
import { useRouter } from 'next/navigation';

export default function ReportPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [viewMode, setViewMode] = useState<'report' | 'process'>('report');

  // For demo, bypass API call and use mock data
  const { data: reportData, isLoading: isLoadingReal } = useFinalReport(id === 'demo-with-research' ? null : id);
  const isLoading = id === 'demo-with-research' ? false : isLoadingReal;

  // Enhancement research answers for demo mode
  const enhancementResearchAnswers = (id === 'demo-with-research') ? [
    {
      question_id: 'q-5',
      question_text: 'What is the competitive landscape for enterprise SaaS in Western Europe?',
      answer: `The competitive landscape in Western Europe is highly fragmented with both global players and regional specialists competing across multiple segments.

Key findings:
• Market leaders (SAP, Salesforce, Microsoft) hold approximately 45% combined market share across 5 major countries
• Regional players like Sage, Unit4, and Exact focus heavily on compliance-heavy industries (finance, healthcare, government)
• Pricing strategies vary 30-40% across markets due to purchasing power differences and local competition

Market context: Western European enterprise software spending reached €85B in 2024, growing at 12% CAGR. The market shows strong preference for data sovereignty and GDPR-compliant solutions, creating opportunities for EU-based providers. However, sales cycles are 40% longer than North America due to consensus-based decision making and stricter procurement processes.`,
      citations: [
        {
          url: 'https://www.gartner.com/eu-enterprise-saas-2024',
          title: 'Gartner Enterprise Software Market Report 2024',
          relevance: 0.94
        },
        {
          url: 'https://www.idc.com/european-saas-trends',
          title: 'IDC European SaaS Market Trends Q4 2024',
          relevance: 0.89
        },
        {
          url: 'https://techcrunch.com/eu-market-analysis',
          title: 'TechCrunch European Enterprise Software Analysis',
          relevance: 0.82
        }
      ],
      quality_indicator: 'high'
    },
    {
      question_id: 'q-8',
      question_text: 'What are the typical customer acquisition costs for B2B SaaS in European markets?',
      answer: `Customer acquisition costs (CAC) in European B2B SaaS markets vary significantly by segment and geography, but average 15-25% higher than North American equivalents.

Key findings:
• SMB segment: €3,000-€8,000 per customer (18-24 month payback period)
• Mid-market: €15,000-€45,000 per customer (24-30 month payback period)
• Enterprise: €80,000-€250,000 per customer (30-42 month payback period)

Market context: Higher CAC driven by longer sales cycles, need for localized content in multiple languages, and relationship-based selling culture. However, European customers show 20% higher retention rates once acquired, offsetting higher initial costs. Channel partnerships and local resellers can reduce CAC by 30-40% but require margin sharing of 15-25%.`,
      citations: [
        {
          url: 'https://www.saastr.com/european-cac-benchmarks',
          title: 'SaaStr European CAC Benchmarks 2024',
          relevance: 0.91
        },
        {
          url: 'https://www.mckinsey.com/b2b-saas-europe',
          title: 'McKinsey B2B SaaS Growth Study - Europe',
          relevance: 0.87
        }
      ],
      quality_indicator: 'high'
    }
  ] : [];

  // Complete mock report for demonstration
  const mockReport = {
    trace_id: 'demo-with-research',
    query: 'Should our SaaS company expand into the European enterprise market?',
    executive_summary: 'Based on comprehensive analysis of market dynamics, competitive landscape, and operational requirements, European enterprise expansion presents a compelling strategic opportunity with manageable risks. The market shows strong fundamentals with €85B in enterprise software spending and 12% CAGR, though success requires careful navigation of longer sales cycles, localization requirements, and regulatory compliance. A phased approach starting with Western Europe, leveraging channel partnerships, and building GDPR-compliant infrastructure positions the company for sustainable growth while mitigating execution risks.',
    strategic_recommendations: [
      {
        recommendation: 'Launch phased European expansion starting with UK, Germany, and Netherlands',
        priority: 'critical',
        rationale: 'These three markets represent 55% of Western European enterprise SaaS spend and share common language/business practices that reduce localization costs',
        implementation_guidance: 'Establish EU entity in Netherlands (tax efficiency), hire country managers, build channel partner network. Timeline: 6-9 months. Budget: €2.5-3.5M first year.',
        expected_impact: '€8-12M ARR within 18 months',
        risks: ['Longer sales cycles', 'Channel partner dependencies', 'Regulatory compliance complexity'],
        dependencies: ['GDPR-compliant infrastructure', 'Localized product', 'European bank account']
      },
      {
        recommendation: 'Build GDPR-compliant data infrastructure before market entry',
        priority: 'critical',
        rationale: 'Data sovereignty concerns are deal-breakers for 70% of European enterprise buyers. EU-based data centers required.',
        implementation_guidance: 'Partner with AWS Frankfurt/Dublin or Azure Netherlands regions. Implement data residency controls and audit logging. Obtain ISO 27001 certification.',
        expected_impact: 'Eliminates primary objection in sales process',
        risks: ['Infrastructure costs', 'Migration complexity'],
        dependencies: ['Technical architecture review', 'Security audit']
      },
      {
        recommendation: 'Establish channel partner network for mid-market penetration',
        priority: 'high',
        rationale: 'Channel partners reduce CAC by 35% and provide local market knowledge. Essential for mid-market success.',
        implementation_guidance: 'Recruit 3-5 strategic resellers per country. Provide 20-25% margin, co-marketing budget, and technical training.',
        expected_impact: '40% of European revenue through partners by year 2',
        risks: ['Partner quality', 'Margin pressure'],
        dependencies: ['Partner program design', 'Channel enablement materials']
      }
    ],
    quality_metrics: {
      overall_quality: 0.89,
      cognitive_diversity_index: 0.76,
      evidence_strength: 0.84,
      recommendation_clarity: 0.91,
      execution_time_ms: 145000,
      consultant_count: 5,
      mental_models_applied: 12,
      nway_relations_activated: 3
    },
    consultant_analyses: [
      {
        consultant_name: 'Market Entry Strategist',
        consultant_type: 'strategic',
        perspective: 'European market entry requires balancing speed with localization quality',
        key_insights: [
          'UK provides English-language beachhead with familiar business culture',
          'DACH region offers highest enterprise spend but requires German localization',
          'Nordics show strong SaaS adoption but smaller market size'
        ],
        recommendations: ['Phased entry by market maturity', 'Invest in local partnerships'],
        concerns: ['Sales cycle extension', 'Localization costs'],
        confidence_score: 0.87
      },
      {
        consultant_name: 'International Expansion CFO',
        consultant_type: 'financial',
        perspective: 'European expansion financially viable with 18-24 month payback',
        key_insights: [
          'CAC 20% higher than US but offset by better retention',
          'Channel partners reduce upfront investment',
          'Currency risk manageable through hedging'
        ],
        recommendations: ['Establish EU entity for tax efficiency', 'Partner-heavy go-to-market'],
        concerns: ['Working capital requirements', 'FX exposure'],
        confidence_score: 0.82
      }
    ],
    evidence_trail: [
      {
        source_type: 'web_research',
        source_name: 'Gartner European Enterprise Software Report 2024',
        content_snippet: 'Western European enterprise software spending reached €85B in 2024, growing at 12% CAGR...',
        credibility_score: 0.94,
        relevance_score: 0.91,
        url: 'https://www.gartner.com/eu-software-2024'
      },
      {
        source_type: 'web_research',
        source_name: 'IDC SaaS Market Analysis',
        content_snippet: 'European B2B SaaS buyers prioritize data sovereignty, with 70% requiring EU-based infrastructure',
        credibility_score: 0.89,
        relevance_score: 0.88,
        url: 'https://www.idc.com/saas-europe'
      }
    ],
    created_at: new Date().toISOString(),
    enhancement_research_answers: enhancementResearchAnswers
  };

  // Use mock report for demo, otherwise use API data
  const report: any = id === 'demo-with-research' ? mockReport : (reportData as any);

  const generatePitch = useGeneratePitch();
  const startIdeaflow = useStartIdeaflowSprint();
  const startCopywriter = useStartCopywriterJob();

  const handleExportMarkdown = () => {
    if (!report) return;
    const markdown = `# ${report.query}\n\n${report.executive_summary}\n\n## Strategic Recommendations\n\n${(report.strategic_recommendations || []).map((r: any, i: number) => `${i + 1}. ${r.recommendation}\n\n${r.rationale}`).join('\n\n')}`;
    downloadAsFile(markdown, `analysis-${id}.md`, 'text/markdown');
  };

  const handleGeneratePitch = async () => {
    if (!report) return;
    try {
      const response = await generatePitch.mutateAsync({
        strategic_content: report.executive_summary,
      });
      router.push(`/pitch/${response.pitch_id}`);
    } catch (error) {
      console.error('Failed to generate pitch:', error);
    }
  };

  const handleStartIdeaflow = async () => {
    if (!report) return;
    try {
      const response = await startIdeaflow.mutateAsync({
        problem_statement: report.query,
      });
      router.push(`/ideaflow/${response.sprint_id}`);
    } catch (error) {
      console.error('Failed to start ideaflow:', error);
    }
  };

  const handleTransformToCopy = async () => {
    if (!report) return;
    try {
      const response = await startCopywriter.mutateAsync({
        trace_id: id,
      });
      router.push(`/copywriter/${response.job_id}`);
    } catch (error) {
      console.error('Failed to start copywriter:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-lg">Loading report...</p>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-lg">Report not found</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b-2 border-black">
        <div className="container-wide py-4">
          <div className="flex justify-between items-center">
            <Link href="/dashboard" className="text-2xl font-bold">
              Lolla
            </Link>
            <Link href="/dashboard" className="hover:text-accent">
              ← Dashboard
            </Link>
          </div>
        </div>
      </header>

      {/* Tab Toggle */}
      <div className="border-b-2 border-black">
        <div className="container-wide flex gap-4">
          <button
            onClick={() => setViewMode('report')}
            className={`px-6 py-3 border-r-2 border-black font-semibold ${
              viewMode === 'report' ? 'bg-black text-white' : 'bg-white hover:bg-gray-100'
            }`}
          >
            Report Mode
          </button>
          <button
            onClick={() => setViewMode('process')}
            className={`px-6 py-3 font-semibold ${
              viewMode === 'process' ? 'bg-black text-white' : 'bg-white hover:bg-gray-100'
            }`}
          >
            Process / Glass-Box
          </button>
        </div>
      </div>

      {/* Main Content */}
      <main className="container-content py-12">
        {viewMode === 'report' ? (
          <>
            {/* Executive Summary */}
            <section className="mb-12">
              <div className="flex justify-between items-start mb-4">
                <h1 className="text-4xl font-bold">Strategic Analysis</h1>
                <button
                  onClick={() => copyToClipboard(report.executive_summary)}
                  className="btn text-sm"
                >
                  [▼MD]
                </button>
              </div>
              <p className="text-lg text-gray-600 mb-6">{report.query}</p>
              <div className="prose prose-lg max-w-none">
                <p className="leading-relaxed">{report.executive_summary}</p>
              </div>
            </section>

            {/* Quality Metrics */}
            {report.quality_metrics && (
            <section className="mb-12 border-2 border-black p-6">
              <h2 className="text-2xl font-bold mb-4">Quality Metrics</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Overall Quality</p>
                  <p className="text-2xl font-bold">{Math.round((report.quality_metrics?.overall_quality || 0) * 100)}%</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Cognitive Diversity</p>
                  <p className="text-2xl font-bold">{(report.quality_metrics?.cognitive_diversity_index || 0).toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Consultants</p>
                  <p className="text-2xl font-bold">{report.quality_metrics?.consultant_count || 0}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Mental Models</p>
                  <p className="text-2xl font-bold">{report.quality_metrics?.mental_models_applied || 0}</p>
                </div>
              </div>
            </section>
            )}

            {/* Strategic Recommendations */}
            <section className="mb-12">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold">Strategic Recommendations</h2>
                <button onClick={() => {}} className="btn text-sm">[▼MD]</button>
              </div>
              <div className="space-y-6">
                {(report.strategic_recommendations || []).map((rec: any, index: number) => (
                  <div key={index} className="card">
                    <div className="flex items-start gap-3 mb-3">
                      <span className={`px-2 py-1 border-2 text-xs font-bold ${
                        rec.priority === 'critical' ? 'border-error text-error' :
                        rec.priority === 'high' ? 'border-warning text-warning' :
                        'border-black'
                      }`}>
                        {rec.priority.toUpperCase()}
                      </span>
                      <h3 className="text-lg font-semibold flex-1">{rec.recommendation}</h3>
                    </div>
                    <p className="text-gray-600 mb-3">{rec.rationale}</p>
                    <p className="text-sm"><strong>Implementation:</strong> {rec.implementation_guidance}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* Enhancement Research Answers */}
            {(report as any).enhancement_research_answers && (report as any).enhancement_research_answers.length > 0 && (
              <section className="mb-12">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold">Enhancement Research</h2>
                  <span className="text-sm text-gray-600">Questions you flagged for research</span>
                </div>
                <div className="space-y-6">
                  {(report as any).enhancement_research_answers.map((research: any, index: number) => (
                    <div key={index} className="card bg-blue-50 border-blue-500">
                      <div className="flex items-start gap-3 mb-3">
                        <span className="px-2 py-1 border-2 border-blue-600 bg-blue-600 text-white text-xs font-bold">
                          RESEARCHED
                        </span>
                        <h3 className="text-lg font-semibold flex-1">{research.question_text}</h3>
                      </div>
                      <div className="prose prose-sm max-w-none mb-4">
                        <p className="text-gray-700 whitespace-pre-line">{research.answer}</p>
                      </div>
                      {research.citations && research.citations.length > 0 && (
                        <div className="border-t-2 border-blue-200 pt-3">
                          <p className="text-sm font-semibold mb-2">Sources:</p>
                          <div className="space-y-1">
                            {(research.citations || []).slice(0, 3).map((citation: any, cidx: number) => (
                              <div key={cidx} className="text-sm">
                                <a
                                  href={citation.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:underline"
                                >
                                  {citation.title || 'Source'}
                                </a>
                                <span className="text-gray-500 ml-2">
                                  (Relevance: {Math.round(citation.relevance * 100)}%)
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            )}

            {/* Take This Further */}
            <section className="mb-12 border-t-2 border-black pt-12">
              <h2 className="text-2xl font-bold mb-6">Take This Further</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button
                  onClick={handleGeneratePitch}
                  disabled={generatePitch.isPending}
                  className="card hover:border-accent text-left"
                >
                  <h3 className="text-xl font-semibold mb-2">Generate Pitch Deck</h3>
                  <p className="text-gray-600 mb-3">
                    Turn this analysis into an investor-ready pitch with objection playbook.
                  </p>
                  <span className="text-accent">→</span>
                </button>
                <button
                  onClick={handleStartIdeaflow}
                  disabled={startIdeaflow.isPending}
                  className="card hover:border-accent text-left"
                >
                  <h3 className="text-xl font-semibold mb-2">Ideaflow Sprint</h3>
                  <p className="text-gray-600 mb-3">
                    Generate 50-200 solution ideas with experimental designs.
                  </p>
                  <span className="text-accent">→</span>
                </button>
                <button
                  onClick={handleTransformToCopy}
                  disabled={startCopywriter.isPending}
                  className="card hover:border-accent text-left"
                >
                  <h3 className="text-xl font-semibold mb-2">Copywriter Transform</h3>
                  <p className="text-gray-600 mb-3">
                    Transform into persuasive communication with defensive strategies.
                  </p>
                  <span className="text-accent">→</span>
                </button>
              </div>
            </section>
          </>
        ) : (
          <>
            {/* Process Mode */}
            <h1 className="text-4xl font-bold mb-8">Glass-Box Process View</h1>

            {/* Consultant Analyses */}
            <section className="mb-12">
              <h2 className="text-2xl font-bold mb-4">Consultant Perspectives</h2>
              <div className="space-y-4">
                {(report.consultant_analyses || []).map((analysis: any, index: number) => (
                  <details key={index} className="card">
                    <summary className="cursor-pointer font-semibold text-lg">
                      {analysis.consultant_name} ({analysis.consultant_type})
                    </summary>
                    <div className="mt-4 space-y-3">
                      <div>
                        <h4 className="font-semibold mb-1">Perspective</h4>
                        <p className="text-gray-600">{analysis.perspective}</p>
                      </div>
                      <div>
                        <h4 className="font-semibold mb-1">Key Insights</h4>
                        <ul className="list-disc list-inside text-gray-600">
                          {(analysis.key_insights || []).map((insight: any, i: number) => (
                            <li key={i}>{insight}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </details>
                ))}
              </div>
            </section>

            {/* Evidence Trail */}
            <section className="mb-12">
              <h2 className="text-2xl font-bold mb-4">Evidence Trail</h2>
              <div className="space-y-3">
                {(report.evidence_trail || []).slice(0, 10).map((evidence: any, index: number) => (
                  <div key={index} className="border-2 border-black p-4">
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-semibold">{evidence.source_name}</span>
                      <span className="text-sm text-gray-600">
                        Credibility: {Math.round(evidence.credibility_score * 100)}%
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm">{evidence.content_snippet}</p>
                    {evidence.page_number && (
                      <p className="text-xs text-gray-500 mt-1">Page {evidence.page_number}</p>
                    )}
                  </div>
                ))}
              </div>
            </section>
          </>
        )}

        {/* Action Bar */}
        <div className="border-t-2 border-black pt-6 flex gap-4">
          <button onClick={handleExportMarkdown} className="btn flex-1">
            Export Full Markdown
          </button>
          <button onClick={() => window.print()} className="btn flex-1">
            Export PDF
          </button>
        </div>
      </main>
    </div>
  );
}
