# Project Chat Implementation Guide

## ✅ What's Been Built

### 1. Database Schema (`supabase/migrations/002_project_chat.sql`)
- ✅ `project_chat_messages` table with RLS
- ✅ `rag_documents` table for project knowledge
- ✅ `rag_text_chunks` table for vector search
- ✅ Indexes and policies for performance & security

### 2. Backend API (`/src/api/routes/project_chat.py`)
- ✅ `POST /api/v1/projects/{project_id}/chat` - Main chat endpoint
- ✅ Uses Grok-4-Fast via OpenRouter ($0.20/1M input, $0.50/1M output)
- ✅ Integrates with ProjectRAGPipeline for semantic search
- ✅ Returns answers with sources and context metadata

### 3. FastAPI Integration (`/src/main.py`)
- ✅ Router imported and registered
- ✅ Available at `/api/v1/projects/{project_id}/chat`

### 4. Frontend UI (Already Built)
- ✅ Dashboard with Projects + Standalone Analyses
- ✅ Project Detail Page with Chat Interface
- ✅ TypeScript types for Project, Analysis, ChatMessage

## 🔧 Steps to Complete

### Step 1: Run Database Migration

```bash
# Connect to your Supabase instance and run:
cd /Users/marcin/lolla_v6/frontend_2.0
psql <YOUR_SUPABASE_CONNECTION_STRING> < supabase/migrations/002_project_chat.sql

# Or via Supabase Dashboard:
# 1. Go to SQL Editor
# 2. Paste contents of 002_project_chat.sql
# 3. Run
```

### Step 2: Update Frontend API Client

Create `/Users/marcin/lolla_v6/frontend_2.0/lib/api/project-chat.ts`:

```typescript
import axios from 'axios';

const METIS_API_URL = process.env.NEXT_PUBLIC_METIS_API_URL || 'http://localhost:8000';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ProjectChatRequest {
  project_id: string;
  user_id: string;
  message: string;
  conversation_history: ChatMessage[];
}

export interface ProjectChatResponse {
  answer: string;
  sources: Array<{
    analysis_id: string;
    confidence: number;
  }>;
  context_used: {
    relevant_docs_count: number;
    confidence_score: number;
    context_available: boolean;
  };
  model_used: string;
  tokens_used: number;
}

export const projectChatAPI = {
  async sendMessage(request: ProjectChatRequest): Promise<ProjectChatResponse> {
    const response = await axios.post(
      `${METIS_API_URL}/api/v1/projects/${request.project_id}/chat`,
      request
    );
    return response.data;
  },

  async getHistory(projectId: string, userId: string, limit = 50) {
    const response = await axios.get(
      `${METIS_API_URL}/api/v1/projects/${projectId}/chat/history`,
      { params: { user_id: userId, limit } }
    );
    return response.data;
  },
};
```

### Step 3: Update Project Detail Page

Edit `/app/projects/[id]/page.tsx` - replace mock chat with real API:

```typescript
import { projectChatAPI, ChatMessage } from '@/lib/api/project-chat';

// Replace handleSendMessage function:
const handleSendMessage = async () => {
  if (!chatInput.trim()) return;

  const userMessage: ChatMessage = {
    role: 'user',
    content: chatInput,
  };

  setChatMessages([...chatMessages, userMessage]);
  setChatInput('');
  setIsLoading(true);

  try {
    const response = await projectChatAPI.sendMessage({
      project_id: projectId,
      user_id: 'mock-user', // Replace with real user ID from auth
      message: chatInput,
      conversation_history: chatMessages,
    });

    const aiMessage: ChatMessage = {
      role: 'assistant',
      content: response.answer,
    };

    setChatMessages((prev) => [...prev, aiMessage]);

    // Optional: Show sources to user
    console.log('Sources:', response.sources);
    console.log('Context:', response.context_used);

  } catch (error) {
    console.error('Chat error:', error);
    const errorMessage: ChatMessage = {
      role: 'assistant',
      content: 'Sorry, I encountered an error. Please try again.',
    };
    setChatMessages((prev) => [...prev, errorMessage]);
  } finally {
    setIsLoading(false);
  }
};
```

### Step 4: Test the Integration

1. **Start Backend:**
   ```bash
   cd /Users/marcin/lolla_v6
   python3 src/main.py
   # Should see: "✅ Specialized workflow APIs registered (ideaflow, copywriter, pitch, documents, project_chat)"
   ```

2. **Start Frontend:**
   ```bash
   cd /Users/marcin/lolla_v6/frontend_2.0
   npm run dev
   ```

3. **Test Flow:**
   - Go to http://localhost:3001/dashboard
   - Click on a project
   - Type a question in the chat: "What are the key findings?"
   - Should get AI response using Grok-4-Fast + project knowledge

### Step 5: Add Supabase Persistence (Optional)

To store conversations in Supabase, implement the `store_chat_message()` function in `/src/api/routes/project_chat.py`:

```python
from supabase import create_client

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

async def store_chat_message(...):
    # Insert user message
    supabase.table("project_chat_messages").insert({
        "project_id": project_id,
        "user_id": user_id,
        "role": "user",
        "content": user_message,
        "created_at": datetime.now().isoformat()
    }).execute()

    # Insert assistant message
    supabase.table("project_chat_messages").insert({
        "project_id": project_id,
        "user_id": user_id,
        "role": "assistant",
        "content": assistant_message,
        "context_used": {"sources": sources},
        "tokens_used": tokens_used,
        "model_used": "grok-4-fast",
        "created_at": datetime.now().isoformat()
    }).execute()
```

## 🎯 Architecture Overview

```
User Question
    ↓
Frontend (Next.js)
    ↓
POST /api/v1/projects/{id}/chat
    ↓
ProjectRAGPipeline
    ├── Semantic search across project analyses
    ├── Vector similarity (pgvector)
    └── Returns relevant context
    ↓
OpenRouter Client (Grok-4-Fast)
    ├── System prompt + project context
    ├── Conversation history
    └── User question
    ↓
AI Response
    ├── Answer with citations
    ├── Source analyses used
    └── Confidence scores
    ↓
Store in Supabase (optional)
    ↓
Return to Frontend
```

## 💰 Cost Efficiency

- **Grok-4-Fast**: $0.20/1M input + $0.50/1M output tokens
- **Typical chat**: ~500 input + 300 output = **$0.00025 per message**
- **1000 messages**: ~$0.25
- **10,000 messages**: ~$2.50

Compare to Claude Sonnet 3.5:
- $3/1M input + $15/1M output
- Same chat: **$0.0060 per message** (24x more expensive!)

## 🔐 Security

All tables have Row Level Security (RLS) enabled:
- Users only see their own projects
- Users only see their own chat messages
- Users only query their own RAG documents

## 📝 Next Steps

1. ⏸️ Run SQL migration (manual - user will execute in Supabase)
2. ✅ Create frontend API client (`/lib/api/project-chat.ts`)
3. ✅ Update project detail page (integrated with real API)
4. ⏸️ Test end-to-end (requires SQL migration + backend running)
5. ⏸️ Add Supabase persistence (optional)
6. ⏸️ Add authentication (replace mock-user with real auth)
7. ⏸️ Add chat history loading from Supabase
