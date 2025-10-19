# Report V3 Design Specification
**Glass-Box Dashboard for METIS Strategic Analysis**

**Version**: 3.0
**Date**: 2025-10-10
**Status**: Approved - Ready for Implementation

---

## Executive Summary

Report V3 replaces the vertical-scroll, tab-based V2 with an **adaptive split-pane workspace** optimized for:

1. **Manager Mode** (decision-first): L1 zoom showing "what to do" + "why" in <1 viewport
2. **Analyst Mode** (process-first): L2 zoom with full glass-box transparency
3. **Progressive disclosure**: 3-layer information architecture (L0 → L1 → L2 → L3)
4. **Graph-like navigation**: Evidence pins, "Because..." links, cross-referencing
5. **No scroll hell**: Sticky micro-summary + resizable panes + zoom controls

### Key Problems Solved

| V2 Problem | V3 Solution |
|------------|-------------|
| 2100px vertical scroll | 700px viewport (L0/L1), collapsible micro-summary |
| Glass-box data buried in modal | Horizontal stage chips + evidence pane |
| Can't trace rec → evidence → consultant | Graph nodes with "Because..." links + pins |
| Tabs hide cross-referencing | Split-pane: narrative (left) + evidence (right) |
| One-size-fits-all density | Zoom levels L0-L3 with mode presets |

---

## Design Decisions (Final)

### 1. Stage Navigation: Horizontal Breadcrumb + Optional Vertical Rail

**Decision**: Horizontal chip row by default; vertical icon rail only on wide screens (≥1280px) or Analyst mode.

**Rationale**:
- Saves 10% horizontal space on laptops
- Mobile-friendly (chips scroll horizontally)
- Keeps stage context visible without tab-hiding

**Behavior**:
```
[All] [MECE] [Consultants] [Models] [Devil's Advocate] [Research] [Synthesis]
  ↑ Filter chips, not tabs
```

- **Desktop (≥1280px) + Analyst mode**: Optional 60px icon rail on far left (toggle with `g`)
- **Mobile (<1280px) or Manager mode**: Horizontal chip row only

---

### 2. Evidence Pane: Closed by Default, Opens on Demand

**Decision**: Evidence pane starts **closed**, opens on click, remembers state via `localStorage`.

**Behavior**:
- **Click evidence pin** → Pane opens, stays open
- **Hover pin** → 3-4 line tooltip (desktop only, ignored on touch)
- **Press `e`** → Toggle pane open/closed
- **Multi-select pins** → Pane shows tabbed compare view
- **Resize handle** → Width saved to `localStorage`

**Rationale**:
- Avoids hover jank on touch devices
- Prevents auto-hide blur ambiguity
- Allows side-by-side evidence comparison (tabs within pane)

---

### 3. Zoom Level Defaults by Mode

**Decision**:
- **Manager Mode**: L1 at load (show "what + why")
- **Analyst Mode**: L2 at load (show full process)
- **L0**: Reserved for export/presentation (too sparse for interaction)

**Zoom Descriptions**:
- **L0**: Decision chip + confidence/risk + 3-5 bullet critical path
- **L1**: Recommendations + top consultant + top evidence + top challenge per rec
- **L2**: Full MECE tree, all consultants, all models, all challenges
- **L3**: Document mode with inline footnotes, full markdown, raw research

**Rationale**: L0 skeleton is too minimal for first-time users; L1 answers exec question "what should I do and why?"

---

### 4. L3 Layout: Document Mode with Inline Footnotes

**Decision**: At L3, switch to **single-pane Document mode** with inline evidence footnotes `[1]`, `[2]`.

**Behavior**:
- Footnotes link to references section within the same pane
- Right evidence pane available but not required (toggle with `e`)
- Optimized for continuous reading, copy-paste, accessibility

**Rationale**:
- L3 users want to read full narrative + evidence as one flow
- Inline footnotes preserve markdown accessibility
- Avoids split-pane redundancy (evidence both inline AND in right pane)

---

### 5. Deep Links: Short Hash + Optional Full-State Share

**Decision**: Use short URL hash for shareability, `sessionStorage` for ephemeral UI state.

**URL Format**:
```
Default:     /analysis/abc123#L2:rec-1
Full state:  /analysis/abc123#L2:rec-1?state=<base64>
```

**State Mapping**:
- **URL hash**: Zoom level (L0-L3) + active node ID
- **sessionStorage**: Pane sizes, scroll position, collapsed nodes
- **Querystring `?state=`**: Full compressed state (gzip JSON) for "Share exact view"

**Rationale**:
- Short URLs are shareable and readable
- Full pixel-perfect state only when explicitly requested
- Avoids 200+ char URLs for basic sharing

---

### 6. Cross-Reference Flow: Replace with Breadcrumb

**Decision**: When clicking a consultant, left pane **replaces** content with consultant analysis + breadcrumb.

**Navigation**:
```
Recommendations > Rec #1: GDPR Infrastructure > Financial Analyst
                                               [← Back]
```

- **Click consultant** → Replace left pane, show breadcrumb
- **Backspace or ← Back** → Return to previous view
- **Alt+Click** → Insert consultant analysis below current rec (advanced)

**Rationale**:
- Simpler than tabs or stacking
- Breadcrumb provides clear navigation path
- Alt+Click preserves context for power users

---

## Layout Specification

```
┌─────────────────────────────────────────────────────────────────┐
│ Micro-Summary Bar (sticky, 64px, collapsible to 40px)          │
│ [Decision: PROCEED] [Confidence ████████░░ 82%] [Risk: MED-HI] │
│ [Governing Thought: European expansion viable with 18mo...▼]   │
│ [Actions: Share | Export | Expand ⌄]                           │
└─────────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────────┐
│ Stage Chips Row (horizontal, 40px)                             │
│ [All] [MECE] [Consultants] [Models] [Devil's Advocate] [...]   │
└─────────────────────────────────────────────────────────────────┘
┌──────────────────────────────┬──────────────────────────────────┐
│ Narrative Pane (Left)        │ Evidence Pane (Right)            │
│ Min 560px, default 60%       │ Min 420px, default 480px         │
│                              │ [Closed by default]              │
│ • Recommendations (L1)       │                                  │
│   ├─ Support Graph           │ Opens on click:                  │
│   ├─ Evidence Pins [3]       │ • Source quotes                  │
│   └─ "Because..." links      │ • Citations                      │
│                              │ • Provenance badges              │
│ • Consultants                │ • Model attributions             │
│ • Models Applied             │                                  │
│ • Challenges                 │ [Tabs: Source 1 | Source 2]      │
│                              │ [X Close] [↔ Resize]             │
│                              │                                  │
│ [Zoom: L0  L1  L2  L3]      │                                  │
│  ↑ Segmented control         │                                  │
└──────────────────────────────┴──────────────────────────────────┘
```

### Viewport Budget (Desktop)

| Component | Height | Behavior |
|-----------|--------|----------|
| Micro-summary (expanded) | 64px | Sticky, collapses to 40px after scroll |
| Stage chips | 40px | Sticky |
| Narrative + Evidence | calc(100vh - 104px) | Resizable split |
| **Total viewport** | **100vh** | No full-page scroll needed at L1 |

---

## Component Specifications

### 1. MicroSummaryBar

**Props**:
```typescript
interface MicroSummaryProps {
  decision: 'PROCEED' | 'CAUTION' | 'HALT';
  confidence: number; // 0-1
  risk: 'LOW' | 'MEDIUM' | 'MEDIUM-HIGH' | 'HIGH';
  governingThought: string;
  expanded: boolean;
  onToggle: () => void;
}
```

**Visual Specs**:
- **Confidence bar**: Horizontal, 120px wide, `bg-green-600` fill, numeric % overlay
- **Risk band**: Segmented mini-bar showing facets (legal, market, ops), tooltip explains drivers
- **Governing thought**: Single line when collapsed, 3-4 lines when expanded
- **Actions dropdown**: Share (copy deep link), Export (PDF options), Expand full view

**States**:
- **Expanded** (first load): 64px height, shows full governing thought
- **Collapsed** (after scroll >100px): 40px height, truncated thought with "..."

---

### 2. StageChipsRow

**Props**:
```typescript
interface StageChipsProps {
  stages: Array<{
    id: string;
    label: string;
    icon?: string;
    count?: number; // badges
  }>;
  activeStage: string | null;
  onStageClick: (stageId: string) => void;
}
```

**Visual Specs**:
- Chips: `px-3 py-1.5 rounded-md border-2 border-black`
- Active: `bg-black text-white`
- Inactive: `bg-white text-black hover:bg-gray-100`
- Badges (count): Small `bg-blue-600 text-white rounded-full px-1.5 text-xs`

**Behavior**:
- Horizontal scroll on overflow (mobile)
- Click filters narrative pane to show only that stage's nodes

---

### 3. NarrativePane

**Sub-components**:
- `<RecommendationCard>` - Compact card with support graph
- `<ConsultantCard>` - Expandable consultant analysis
- `<SupportGraph>` - Node visualization (consultants + models + evidence)
- `<EvidencePin>` - Badge with count, click opens evidence pane

**Props** (simplified):
```typescript
interface NarrativePaneProps {
  zoom: 'L0' | 'L1' | 'L2' | 'L3';
  mode: 'manager' | 'analyst';
  activeNode: string | null;
  onNodeClick: (nodeId: string) => void;
  onEvidenceClick: (evidenceId: string) => void;
}
```

**L1 Recommendation Card**:
```
┌─────────────────────────────────────────────────┐
│ 🎯 Rec #1: GDPR Infrastructure (CRITICAL)      │
│ Confidence: 91% | Impact: 9.2/10 | 4-6 months  │
├─────────────────────────────────────────────────┤
│ Establish GDPR-compliant infrastructure        │
│ before market entry.                            │
│                                                 │
│ Supported by:                                   │
│ • Legal & Compliance [Because...] [E:2]         │
│ • Technical Architect [Because...] [E:1]        │
│                                                 │
│ Challenges: 2 [View Devil's Advocate]          │
│                                                 │
│ [▼ Expand Implementation Steps]                 │
└─────────────────────────────────────────────────┘
```

---

### 4. EvidencePane

**Props**:
```typescript
interface EvidencePaneProps {
  isOpen: boolean;
  width: number; // pixels, min 420
  onClose: () => void;
  onResize: (newWidth: number) => void;
  evidence: Array<{
    id: string;
    type: 'quote' | 'figure' | 'research' | 'model';
    content: string;
    source: string;
    provenance: 'real' | 'derived';
  }>;
  compareMode?: boolean; // multi-select
}
```

**Visual Specs**:
- Slide-in from right
- Resize handle on left edge (6px drag zone)
- Header: Evidence count + close button
- Body: Scrollable, supports tabs for compare mode
- Footer: Provenance badges (Real Data / Derived)

**States**:
- **Closed**: Hidden (width: 0)
- **Open**: Default 480px, saved to `localStorage.evidencePaneWidth`
- **Compare**: Shows tabs, internal side-by-side split

---

### 5. ZoomControl

**Visual**:
```
[L0] [L1] [L2] [L3]
 ↑ Segmented control, bottom-left of narrative pane
```

**Keyboard Shortcuts**:
- `1` → L0
- `2` → L1 (default Manager)
- `3` → L2 (default Analyst)
- `4` → L3 (Document mode)

**Behavior**:
- Clicking zoom level transitions content with 150ms fade
- Active level: `bg-black text-white`
- Inactive: `border-2 border-black hover:bg-gray-100`

---

### 6. L3 Document Mode

**Layout Change**:
- Narrative pane expands to 100% width (evidence pane hidden by default)
- Inline footnotes: `[1]`, `[2]` link to references section at bottom
- Typography: Larger base font (16px), wider line-height (1.7), serif optional

**Example**:
```markdown
European enterprise expansion presents compelling opportunity ($85B market, 12% CAGR)[1] but requires 18-24 month investment in localization[2], GDPR compliance[3], and channel partnerships.

## References
[1] Gartner Market Analysis, Q2 2025
[2] McKinsey Localization Study, 2024
[3] GDPR Compliance Cost Analysis (internal)
```

**Toggle**: Press `e` to show evidence pane for quick navigation

---

## Interaction Patterns

### Evidence Pin Flow

1. **Hover pin** → Tooltip shows 3-4 lines preview (desktop only)
2. **Click pin** → Evidence pane opens, scrolls to that evidence
3. **Multi-select** (Cmd+Click) → Evidence pane shows tabs for comparison
4. **Right pane citation** → Has "Show in context" link that highlights source in narrative

### "Because..." Link Flow

1. **Click "Because..."** on recommendation → Jump to consultant who provided that insight
2. **Breadcrumb updates**: `Recommendations > Rec #1 > Financial Analyst`
3. **Back button** → Return to recommendations list
4. **Alt+Click** → Opens consultant below rec (stacked view)

### Stage Filter Flow

1. **Click "Consultants" chip** → Narrative pane filters to show only consultant cards
2. **Active chip**: `bg-black text-white`
3. **Click "All"** → Reset filter, show full narrative
4. **Deep link**: `#L2:consultants` automatically activates Consultants chip

---

## Confidence & Risk Visualization

### Confidence Bar (Horizontal)

```
Confidence  [███████████████████░░░░░] 82%
            ↑ Solid fill                ↑ % overlay
```

- Width: 160px
- Fill: `bg-green-600` (≥80%), `bg-yellow-500` (50-79%), `bg-red-500` (<50%)
- Numeric overlay: Bold, right-aligned inside bar

### Risk Band (Segmented)

```
Risk  [■■■■|■■|░░░░░] MED-HIGH
       ↑legal ↑market ↑ops
```

- Width: 140px
- 3-5 segments showing risk facets
- Tooltip on hover explains each facet
- Overall label: `LOW | MEDIUM | MEDIUM-HIGH | HIGH`

---

## Mobile Adaptations (Desktop-First, Mobile-Competent)

| Desktop | Mobile (<768px) |
|---------|-----------------|
| Split pane (left 60% / right 40%) | Single pane, evidence full-screen sheet |
| Stage chips horizontal row | Chips scroll horizontally, no rail |
| Hover tooltips | Click-only interactions |
| Resizable panes | Fixed panes, swipe gestures |
| Zoom segmented control | Zoom buttons stack vertically |
| L3 Document mode | Same (single column already) |

**Mobile Evidence Sheet**:
- Slides up from bottom (80% height)
- Swipe down to close
- Header: Drag handle + close button
- Overlay backdrop: `bg-black/50`

---

## Empty States (Actionable, Not Scary)

| Scenario | Display | Action |
|----------|---------|--------|
| No devil's advocate challenges | "No material challenges logged" | [Run Adversarial Check] button |
| No research queries | "No external sources cited" | [Open Research Panel] link |
| Low confidence (<50%) | ⚠️ Amber banner: "Low confidence detected" | "What would increase confidence?" expandable list |
| No evidence for recommendation | "No direct evidence cited" | [Add Evidence] button (links to research) |

---

## Export Modes

### 1. Executive Pack (2-4 pages)

**Contents**:
- L0/L1 view: Decision + confidence/risk + critical path
- Top 3 recommendations with 1-line support
- Key evidence snippets (3-5 total)

**Format**: PDF with print CSS (serif, page breaks, minimal shadows)

### 2. Full Report (20-50 pages)

**Contents**:
- L1-L3 all content
- Appendix: Raw sources, full research queries, consultant full analyses
- Footnotes compiled per section

**Format**: PDF with table of contents, page numbers, headers/footers

### 3. Custom Export

**UI**: Modal with checklist:
- [ ] Executive summary
- [ ] Recommendations
- [ ] Consultant analyses
- [ ] Evidence sources
- [ ] Devil's advocate challenges
- [ ] Research queries
- [ ] Include footnotes
- [ ] Include charts

**Output**: PDF with selected sections only

---

## Implementation Phases

### Phase 1: Layout Refactor (4-6 hours)

**Deliverables**:
- [x] MicroSummaryBar component (expanded/collapsed states)
- [x] StageChipsRow component (horizontal filter)
- [x] Split-pane layout (NarrativePane + EvidencePane with resize)
- [x] ZoomControl component (L0-L3 segmented)
- [x] Mock data wired to all components

**Acceptance**:
- Senior synthesis visible without scroll at L1
- Stage chips clickable, filter works
- Evidence pane opens/closes, resizes, remembers width
- Zoom transitions work (150ms fade)

---

### Phase 2: Glass-Box Tabs with Existing Data (8-12 hours)

**Deliverables**:
- [x] RecommendationCard with support graph
- [x] ConsultantCard with expandable analysis
- [x] EvidencePin with tooltip + click behavior
- [x] "Because..." links with breadcrumb navigation
- [x] MECE tab showing query decomposition
- [x] Consultant Selection tab with scores + reasoning
- [x] Mental Models tab with applied frameworks
- [x] Devil's Advocate tab with challenges
- [x] Research tab with Perplexity queries

**Acceptance**:
- Click recommendation → See consultants + evidence + challenges
- Click consultant → See full analysis + models used
- Click evidence pin → Right pane opens to exact citation
- Stage chips filter narrative correctly

---

### Phase 3: Interactive Navigation + Backend Enhancements (12-16 hours)

**Deliverables**:
- [x] L3 Document mode with inline footnotes
- [x] Compare mode (multi-select evidence, tabbed pane)
- [x] Command palette (`⌘K`) with fuzzy search
- [x] Deep links (short hash + optional full state)
- [x] Export modes (Executive, Full, Custom)
- [x] Backend: Add MECE tree, selection reasoning, model attribution

**Acceptance**:
- L3 shows continuous reading with footnotes
- Compare mode shows side-by-side evidence
- Command palette jumps to any node
- Deep links restore exact view state
- PDF exports match spec

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Decision time** | < 90s (p50) | Time from load → first confident action |
| **Back-trace time** | < 10s (p95 < 20s) | Time from rec → evidence source |
| **Scroll reduction** | < 1.5x viewport (p50) | Total scroll depth in pixels |
| **Toggle usage** | > 40% sessions | % sessions using zoom or evidence pane |
| **Modal abandonment** | ~0% | Exit rate on modals (should be none) |
| **Mobile completion** | > 60% | % mobile users who complete flow |

---

## Technical Stack

**Framework**: Next.js 15 (App Router) + React 19
**Styling**: Tailwind CSS + Brutalist design tokens
**State**: React hooks (useState, useContext) + localStorage
**Data Fetching**: Existing `useFinalReport` hook
**Markdown**: `react-markdown` + `remark-gfm`
**Resize**: `react-resizable-panels` or custom resize handle
**Virtualization**: `react-window` for L2 long lists
**Deep Links**: URL hash + `sessionStorage` + optional `?state=base64`

---

## Appendix A: NWAY Framework Integration

**Senior Manager Role** (from NWAY docs):
- **Synthesis quality**: Preserve dissent vs converge on direction
- **Weighting schemes**: Merit (believability) > fit-to-query > user preference
- **Narrative devices**: Hooks, tension-resolution, W-S-N structure
- **Measurable markers**: Clarity gain, consensus, coverage, complementarity
- **Dissent thresholds**: Confidence, risk conditions, contradiction patterns

**McKinsey Principles Applied**:
- **Commander's Intent**: Micro-summary = governing thought
- **Pyramid Principle**: L1 = core message + 3-5 MECE key lines
- **SCQA Flow**: Situation (context) → Complication (challenge) → Question (decision) → Answer (recommendation)
- **Progressive Disclosure**: L0 → L1 → L2 → L3 matches "simple → detailed" hierarchy
- **Glass-Box Auditability**: Evidence trails, rationale capture, decision artifacts

---

## Appendix B: Component File Structure

```
frontend_2.0/
├── app/
│   └── analysis/
│       └── [id]/
│           └── report_v2/
│               └── page.tsx (main report page)
├── components/
│   ├── report/
│   │   ├── MicroSummaryBar.tsx
│   │   ├── StageChipsRow.tsx
│   │   ├── NarrativePane.tsx
│   │   ├── EvidencePane.tsx
│   │   ├── ZoomControl.tsx
│   │   ├── RecommendationCard.tsx
│   │   ├── ConsultantCard.tsx
│   │   ├── EvidencePin.tsx
│   │   ├── SupportGraph.tsx
│   │   └── DocumentMode.tsx (L3)
│   └── [existing components]
├── lib/
│   ├── hooks/
│   │   └── useReportState.ts (zoom, pane, active node)
│   └── utils/
│       └── deepLink.ts (hash encoding/decoding)
└── docs/
    └── REPORT_V3_DESIGN.md (this file)
```

---

## Appendix C: Backend Data Requirements

**Already Available**:
- `query`, `system2_tier`, `consultant_selection`, `selected_models`
- `analysis` (senior synthesis), `quality_scores`, `execution_time_ms`
- `trace_id` for glass-box audit trail

**Need to Add** (for full glass-box):
1. **MECE decomposition tree**: How query was chunked into dimensions
2. **Consultant selection reasoning**: Why each consultant was picked (scores + logic)
3. **Model attribution**: How each mental model influenced analysis
4. **Devil's advocate content**: Full challenges + responses (not just count)
5. **Research queries**: Perplexity queries sent + results (not just count)

**API Endpoint** (suggested):
```
GET /api/v53/engagements/:id/glass-box
{
  "mece_tree": [...],
  "consultant_reasoning": [...],
  "model_attribution": [...],
  "devils_advocate_threads": [...],
  "research_queries": [...]
}
```

---

## Sign-Off

**Design Approved By**:
- [x] Product (User/Marcin)
- [x] Engineering (Assistant/Claude)

**Ready for Implementation**: ✅ YES
**Next Step**: Generate skeleton code (page.tsx + core components)

---

**End of Document**
