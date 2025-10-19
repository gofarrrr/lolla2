-- Drop existing tables if they exist (clean slate)
DROP TABLE IF EXISTS project_chat_messages CASCADE;
DROP TABLE IF EXISTS analyses CASCADE;
DROP TABLE IF EXISTS projects CASCADE;

-- Create Projects Table
CREATE TABLE projects (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create Analyses Table (updated to support projects)
CREATE TABLE analyses (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  project_id UUID REFERENCES projects(id) ON DELETE CASCADE,  -- NULL = standalone
  query TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  result JSONB,
  quality_score FLOAT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create Project Chat Messages Table
CREATE TABLE project_chat_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  user_id UUID NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content TEXT NOT NULL,
  context_used JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable Row Level Security
alter table projects enable row level security;
alter table analyses enable row level security;
alter table project_chat_messages enable row level security;

-- RLS Policies for Projects
create policy "Users can view their own projects"
  on projects for select
  using (auth.uid() = user_id);

create policy "Users can insert their own projects"
  on projects for insert
  with check (auth.uid() = user_id);

create policy "Users can update their own projects"
  on projects for update
  using (auth.uid() = user_id);

create policy "Users can delete their own projects"
  on projects for delete
  using (auth.uid() = user_id);

-- RLS Policies for Analyses
create policy "Users can view their own analyses"
  on analyses for select
  using (auth.uid() = user_id);

create policy "Users can insert their own analyses"
  on analyses for insert
  with check (auth.uid() = user_id);

create policy "Users can update their own analyses"
  on analyses for update
  using (auth.uid() = user_id);

create policy "Users can delete their own analyses"
  on analyses for delete
  using (auth.uid() = user_id);

-- RLS Policies for Chat Messages
create policy "Users can view their own chat messages"
  on project_chat_messages for select
  using (auth.uid() = user_id);

create policy "Users can insert their own chat messages"
  on project_chat_messages for insert
  with check (auth.uid() = user_id);

-- Indexes for performance
create index if not exists projects_user_id_idx on projects(user_id);
create index if not exists analyses_user_id_idx on analyses(user_id);
create index if not exists analyses_project_id_idx on analyses(project_id);
create index if not exists chat_messages_project_id_idx on project_chat_messages(project_id);
create index if not exists chat_messages_user_id_idx on project_chat_messages(user_id);

-- Function to update updated_at timestamp
create or replace function update_updated_at_column()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

-- Triggers for updated_at
create trigger update_projects_updated_at before update on projects
  for each row execute function update_updated_at_column();

create trigger update_analyses_updated_at before update on analyses
  for each row execute function update_updated_at_column();
