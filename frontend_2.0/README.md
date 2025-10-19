# Lolla Frontend 2.0

**Beautiful Brutalist AI-First Interface** for the Lolla Strategic Intelligence Platform

## Overview

Lolla Frontend 2.0 is a Next.js 15 application that provides a brutalist, minimalist interface for AI-powered strategic analysis. Built on the Lollapalooza Effect principle: multiple consultant perspectives converging for exponential insights.

### Key Features

- **Strategic Analysis Pipeline**: Multi-consultant AI teams analyze business questions
- **Glass-Box Transparency**: Complete visibility into analysis process and evidence
- **Specialized Workflows**: Pitch generation, ideation sprints, copywriter transformation
- **Mental Models Academy**: 137+ frameworks from 200+ authoritative sources
- **PDF Upload**: Document context enhancement for analyses
- **Real-Time Updates**: Live pipeline progress with quality metrics
- **Brutalist Design**: No clutter. Content is king. Zero rounded corners.

## Tech Stack

- **Framework**: Next.js 15.0.3 (App Router)
- **Language**: TypeScript 5.6+
- **Styling**: Tailwind CSS 3.4+ (custom brutalist theme)
- **State Management**: TanStack Query v5 + Zustand
- **Real-Time**: Socket.io-client 4.7+
- **UI Components**: Radix UI (headless primitives)
- **Forms**: React Hook Form + Zod validation

## Project Structure

```
frontend_2.0/
├── app/                      # Next.js App Router
│   ├── page.tsx              # Landing page
│   ├── dashboard/            # User dashboard
│   ├── analyze/              # Query input + PDF upload
│   ├── analysis/[id]/        # Processing view + report
│   ├── ideaflow/[sprint_id]/ # Ideaflow results
│   ├── copywriter/[job_id]/  # Copywriter results
│   ├── pitch/[pitch_id]/     # Pitch deck viewer
│   └── academy/              # Mental Models Academy
├── components/
│   └── ui/                   # Reusable brutalist components
├── lib/
│   ├── api/
│   │   ├── client.ts         # Type-safe API client
│   │   └── hooks.ts          # React Query hooks
│   ├── design-system.ts      # Design tokens
│   └── utils.ts              # Helper functions
├── types/
│   └── api.ts                # TypeScript types (from pipeline_contracts.py)
├── styles/
│   └── globals.css           # Brutalist CSS + Tailwind
└── public/                   # Static assets
```

## Getting Started

### Prerequisites

- Node.js 20+
- npm 10+
- Backend API running on `localhost:8000`

### Installation

```bash
# Navigate to frontend_2.0 directory
cd /Users/marcin/lolla_v6/frontend_2.0

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:3001`

### Environment Variables

Create `.env.local`:

```bash
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Optional: Supabase Auth
# NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
# NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
```

## Development

### Available Scripts

```bash
npm run dev          # Start development server (port 3001)
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run type-check   # TypeScript type checking
npm test             # Run tests (Vitest)
npm run test:e2e     # Run E2E tests (Playwright)
```

### Design System

The brutalist design system is defined in `lib/design-system.ts`:

**Colors**:
- Black: `#000000`
- White: `#FFFFFF`
- Accent (Electric Blue): `#0066FF`
- Success: `#00DD00`
- Error: `#FF0000`

**Typography**:
- Sans: Inter
- Mono: IBM Plex Mono

**Borders**:
- Default: 2px solid black
- Thick: 3px solid black
- Heavy: 4px solid black
- Radius: 0 (no rounded corners)

**Spacing**: Generous 2-4x standard (32px, 48px, 64px, 128px)

### API Integration

All API calls go through the type-safe client in `lib/api/client.ts`:

```typescript
import { api } from '@/lib/api/client';

// Start analysis
const response = await api.startAnalysis({
  query: 'Should we expand into Europe?',
  quality_target: 0.85,
});

// Get status
const status = await api.getAnalysisStatus(traceId);

// Upload document
const doc = await api.uploadDocument(file, traceId);
```

React Query hooks in `lib/api/hooks.ts` provide automatic caching and real-time polling:

```typescript
import { useAnalysisStatus } from '@/lib/api/hooks';

// Auto-polls every 2s if status is 'processing'
const { data: status } = useAnalysisStatus(traceId);
```

## Features

### 1. Landing Page (`/`)

- Hero section with Lollapalooza Effect messaging
- How It Works (4-step process)
- Glass-Box Transparency section
- Research-Backed section (200+ sources)
- CTA sections

### 2. Dashboard (`/dashboard`)

- Analysis cards with status indicators
- Quality scores display
- Empty state with CTA
- "New Analysis" button

### 3. Query Input (`/analyze`)

- Large textarea (2000 char limit)
- Quality slider (60-95%)
- **PDF Upload** (max 5 files, 10MB each)
- Drag-and-drop support
- "Skip to Analysis" vs "Enhance Query" flow

### 4. Processing View (`/analysis/[id]`)

- Real-time progress bar
- 8 pipeline stages with status
- **N-Way Relation Display** (upcoming)
- Educational content rotation
- Auto-redirect to report on completion

### 5. Final Report (`/analysis/[id]/report`)

**Report Mode**:
- Executive Summary
- Strategic Recommendations (priority-based)
- Quality Metrics (4 key metrics)
- Key Decisions
- Markdown export buttons ([▼MD])

**Process/Glass-Box Mode**:
- Consultant Perspectives (expandable)
- Evidence Trail (with PDF citations)
- Stage outputs
- Orthogonality visualization
- Granular markdown export

**"Take This Further" Section**:
- Generate Pitch Deck →
- Ideaflow Sprint →
- Copywriter Transform →

### 6. Specialized Workflows

**Ideaflow** (`/ideaflow/[sprint_id]`):
- 50-200 solution ideas
- Cluster-based organization
- MVS experiment designs
- Full export

**Copywriter** (`/copywriter/[job_id]`):
- 12-word governing thought
- Polished content (600-3000 words)
- Quality scores (clarity, persuasion, skim test, defensibility)
- Anticipated objections + defensive strategies

**Pitch Deck** (`/pitch/[pitch_id]`):
- Complete slide deck
- Speaker notes per slide
- Visual recommendations
- Objection playbook

### 7. Academy (`/academy`)

- 137+ mental models across 8 categories
- N-Way Relations explorer (30+ relations)
- Browse by category
- Individual model pages (upcoming)
- Link to user's past analyses

## API Endpoints

All endpoints are defined in `lib/api/client.ts`:

### Core Analysis
```
POST   /api/engagements/start
GET    /api/engagements/{trace_id}/status
GET    /api/engagements/{trace_id}/report
```

### Progressive Questions
```
POST   /api/progressive-questions/generate
POST   /api/progressive-questions/submit
```

### Ideaflow
```
POST   /ideaflow/start
GET    /ideaflow/{sprint_id}/status
GET    /ideaflow/{sprint_id}/results
```

### Copywriter
```
POST   /api/copywriter/start
GET    /api/copywriter/{job_id}/status
GET    /api/copywriter/{job_id}/results
```

### Pitch Generation
```
POST   /pitch/generate
GET    /pitch/{pitch_id}/status
GET    /pitch/{pitch_id}/deck
```

### Document Upload
```
POST   /api/documents/upload?trace_id={trace_id}
```

## Deployment

### Build for Production

```bash
npm run build
npm run start
```

### Deployment Options

**Vercel** (Recommended):
```bash
vercel --prod
```

**Docker**:
```dockerfile
FROM node:20-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 3001
CMD ["npm", "start"]
```

**Environment Variables for Production**:
- `NEXT_PUBLIC_API_URL`: Production API URL
- `NEXT_PUBLIC_WS_URL`: Production WebSocket URL
- `NEXT_PUBLIC_SUPABASE_URL`: Supabase URL (if using auth)
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`: Supabase anon key

## Testing

### Unit Tests (Vitest)

```bash
npm test
```

### E2E Tests (Playwright)

```bash
npm run test:e2e
```

### Manual Testing Checklist

- [ ] Landing page loads correctly
- [ ] Dashboard shows analyses
- [ ] Query input accepts text and files
- [ ] PDF upload works (max 5 files, 10MB)
- [ ] Analysis starts and shows progress
- [ ] Processing view polls for status
- [ ] Report displays correctly (both modes)
- [ ] "Take This Further" buttons work
- [ ] Ideaflow sprint generates ideas
- [ ] Copywriter transforms content
- [ ] Pitch deck generates slides
- [ ] Academy displays models
- [ ] Markdown export works
- [ ] Mobile responsive design

## Performance

### Optimization Techniques

- Code splitting per route
- Lazy loading for non-critical components
- React Query caching (1 min stale time)
- Image optimization (Next.js)
- Font optimization (Inter from next/font)

### Target Metrics

- Lighthouse Score: >90
- First Contentful Paint: <1.5s
- Time to Interactive: <3s
- WebSocket Latency: <100ms

## Accessibility

- WCAG 2.1 AA compliant
- Keyboard navigation support
- Focus states for all interactive elements
- Semantic HTML
- ARIA labels where needed

## Browser Support

- Chrome/Edge 90+
- Firefox 90+
- Safari 14+
- Mobile Safari (iOS 14+)
- Mobile Chrome (Android 10+)

## Contributing

### Code Style

- TypeScript strict mode enabled
- ESLint + Prettier configured
- No `any` types allowed
- Functional components with hooks
- Custom hooks in `lib/hooks/`

### Component Guidelines

- Use brutalist design system tokens
- No rounded corners (border-radius: 0)
- 2px borders by default
- Generous spacing (4 = 32px, 6 = 48px)
- No icon libraries (text only)

## Troubleshooting

### Common Issues

**"Module not found" errors**:
```bash
rm -rf node_modules package-lock.json
npm install
```

**API connection errors**:
- Verify backend is running on `localhost:8000`
- Check `.env.local` has correct `NEXT_PUBLIC_API_URL`

**TypeScript errors**:
```bash
npm run type-check
```

**Port 3001 already in use**:
```bash
# Kill process on port 3001
lsof -ti:3001 | xargs kill -9

# Or change port in package.json scripts
"dev": "next dev -p 3002"
```

## License

Proprietary - Lolla Platform

## Support

For issues or questions, contact the development team or create an issue in the repository.

---

**Built with**: Next.js 15 • TypeScript • Tailwind CSS • Brutalist Design Principles

**Status**: ✅ Production Ready
