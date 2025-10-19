export interface Project {
  id: string;
  user_id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface Analysis {
  id: string;
  user_id: string;
  project_id?: string | null; // null = standalone
  query: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: {
    consultants: any[];
    research: any[];
    evidence: any[];
    synthesis: any;
  };
  quality_score?: number;
  created_at: string;
  updated_at: string;
}

export interface ProjectChatMessage {
  id: string;
  project_id: string;
  user_id: string;
  role: 'user' | 'assistant';
  content: string;
  context_used?: {
    analysis_ids: string[];
    evidence_refs: string[];
  };
  created_at: string;
}

export interface ProjectWithAnalyses extends Project {
  analyses: Analysis[];
  analysis_count?: number;
}
