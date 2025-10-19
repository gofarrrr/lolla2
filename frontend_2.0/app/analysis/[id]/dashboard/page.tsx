"use client";

import { use, useState } from 'react'
import { DecisionDashboardLayout } from '@/components/dashboard/layout/DecisionDashboardLayout'
import { TabBar } from '@/components/dashboard/nav/TabBar'
import { PageContainer } from '@/components/layout/PageContainer'
import { useFinalReport } from '@/lib/api/hooks'
import { useDashboardStore } from '@/lib/state/dashboardStore'
import { mapMeceNodes, mapSocraticQuestions, mapConsultants, mapEvidenceTrail, mapResearchProviders, mapDissentSignals } from '@/lib/mappers/report'
import { MECEFrameworkVisualizer } from '@/components/dashboard/process/MECEFrameworkVisualizer'
import { SocraticQuestionsList } from '@/components/dashboard/process/SocraticQuestionsList'
import { ConsultantList } from '@/components/dashboard/team/ConsultantList'
import { ConsultantMemoViewer } from '@/components/dashboard/team/ConsultantMemoViewer'
import { EvidenceLedger } from '@/components/dashboard/evidence/EvidenceLedger'
import { ResearchProviderTrail } from '@/components/dashboard/evidence/ResearchProviderTrail'
import { UCSViewerPlaceholder } from '@/components/dashboard/raw/UCSViewer'
import { DissentPanel } from '@/components/dashboard/dissent/DissentPanel'
import { TriggersTimeline } from '@/components/dashboard/dissent/TriggersTimeline'

// New Decision Dashboard route; initially focuses on Process (Slice 1)

type Params = Promise<{ id: string }>

export default function DashboardPage({ params }: { params: Params }) {
  const { id } = use(params)
  const { data, isLoading, error } = useFinalReport(id)
  const { activeTab } = useDashboardStore()

  // Minimal loading states; reuse existing spinner if desired
  if (isLoading) {
    return (
      <DecisionDashboardLayout>
        <TabBar />
        <PageContainer className="max-w-7xl">
          <div className="py-8 text-sm text-text-label">Loading analysis…</div>
        </PageContainer>
      </DecisionDashboardLayout>
    )
  }

  if (error || !data) {
    return (
      <DecisionDashboardLayout>
        <TabBar />
        <PageContainer className="max-w-7xl">
          <div className="py-8 text-sm text-text-label">We couldn't load this analysis.</div>
        </PageContainer>
      </DecisionDashboardLayout>
    )
  }

  // Process tab view models (Slice 1)
  const mece = mapMeceNodes(data)
  const questions = mapSocraticQuestions(data)
  const consultants = mapConsultants(data)
  const evidenceItems = mapEvidenceTrail(data)
  const providerEvents = mapResearchProviders(data)
  const dissentSignals = mapDissentSignals(data)

  // Local selection state for Team tab
  // Using a client component already; safe to use React hooks here
  // Note: minimal inline state to avoid expanding global store footprint at this stage
  const selectedConsultantIdDefault = consultants[0]?.id

  return (
    <DecisionDashboardLayout>
      <TabBar />
      <PageContainer className="max-w-7xl">
        {activeTab === 'process' && (
          <div className="py-8 space-y-6">
            <MECEFrameworkVisualizer nodes={mece} />
            <SocraticQuestionsList questions={questions} />
          </div>
        )}

        {activeTab === 'team' && (
          <TeamTab consultants={consultants} defaultSelectedId={selectedConsultantIdDefault} />
        )}

        {activeTab === 'evidence' && (
          <div className="py-8 space-y-6">
            <EvidenceLedger items={evidenceItems} />
            <ResearchProviderTrail events={providerEvents} />
          </div>
        )}

        {activeTab === 'raw' && (
          <div className="py-8 space-y-6">
            <UCSViewerPlaceholder />
          </div>
        )}

        {activeTab === 'strategy' && (
          <div className="py-8 space-y-6">
            <DissentPanel signals={dissentSignals} />
            <TriggersTimeline signals={dissentSignals} />
          </div>
        )}
      </PageContainer>
    </DecisionDashboardLayout>
  )
}

function TeamTab({ consultants, defaultSelectedId }: { consultants: ReturnType<typeof mapConsultants>, defaultSelectedId?: string }) {
  const [selected, setSelected] = useState<string | undefined>(defaultSelectedId)
  const active = consultants.find((c) => c.id === selected)
  return (
    <div className="py-8 space-y-6">
      <ConsultantList consultants={consultants} selectedId={selected} onSelect={setSelected} />
      <ConsultantMemoViewer consultant={active} />
    </div>
  )
}
