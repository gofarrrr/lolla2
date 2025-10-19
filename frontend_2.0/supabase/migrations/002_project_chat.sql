-- Add additional columns to project_chat_messages (from migration 001)
ALTER TABLE project_chat_messages
  ADD COLUMN IF NOT EXISTS tokens_used INTEGER,
  ADD COLUMN IF NOT EXISTS model_used TEXT;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON project_chat_messages(created_at DESC);

-- Drop existing RAG tables if they exist
DROP TABLE IF EXISTS rag_text_chunks CASCADE;
DROP TABLE IF EXISTS rag_documents CASCADE;

-- RAG Documents Table
CREATE TABLE rag_documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  user_id UUID NOT NULL,
  source_type TEXT NOT NULL, -- 'analysis', 'web_extraction', 'upload'
  source_id TEXT, -- engagement_id or URL
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- RAG Text Chunks Table (for vector search)
CREATE TABLE rag_text_chunks (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding vector(1536), -- OpenAI embedding dimension
  metadata JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for RAG
CREATE INDEX IF NOT EXISTS idx_rag_documents_project_id ON rag_documents(project_id);
CREATE INDEX IF NOT EXISTS idx_rag_documents_user_id ON rag_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id ON rag_text_chunks(document_id);

-- Vector search index (requires pgvector extension)
-- Note: Ensure pgvector extension is enabled first: CREATE EXTENSION IF NOT EXISTS vector;
CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding ON rag_text_chunks
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- RLS for RAG Documents
ALTER TABLE rag_documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own RAG documents"
  ON rag_documents FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own RAG documents"
  ON rag_documents FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- RLS for RAG Chunks (inherit from documents)
ALTER TABLE rag_text_chunks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view chunks from their documents"
  ON rag_text_chunks FOR SELECT
  USING (
    document_id IN (
      SELECT id FROM rag_documents WHERE user_id = auth.uid()
    )
  );

CREATE POLICY "Users can insert chunks for their documents"
  ON rag_text_chunks FOR INSERT
  WITH CHECK (
    document_id IN (
      SELECT id FROM rag_documents WHERE user_id = auth.uid()
    )
  );

-- Update trigger for rag_documents
CREATE OR REPLACE FUNCTION update_rag_documents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_rag_documents_updated_at
  BEFORE UPDATE ON rag_documents
  FOR EACH ROW
  EXECUTE FUNCTION update_rag_documents_updated_at();
