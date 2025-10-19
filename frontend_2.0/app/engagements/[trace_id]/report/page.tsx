import { redirect } from 'next/navigation';

type Params = Promise<{ trace_id: string }>;

export default async function EngagementReportPage({ params }: { params: Params }) {
  const { trace_id } = await params;
  redirect(`/analysis/${trace_id}/report_v2`);
}
