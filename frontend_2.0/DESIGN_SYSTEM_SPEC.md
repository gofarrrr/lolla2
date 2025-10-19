# Cognitive Prism 2.0 – Design System Spec

**Last Updated:** Oct 15, 2025  
**Status:** APPROVED & ACTIVE  
**Color Strategy:** Neutral-first with accent frames only

---

## 1. COLOR PALETTE

### Primary Neutrals (Typography & Backgrounds)
- **ink-1** `#0F172A` – Headings, primary text (warm-black)
- **ink-2** `#334155` – Body copy, secondary text
- **ink-3** `#64748B` – Subtle text, labels
- **canvas** `#F3F4F6` – Page background (cream-bg)
- **surface** `#FFFFFF` – Card backgrounds
- **mesh** `#C9CED6` – Dividers, wireframe lines (border-default)

### Accent Colors (Frames & Interaction Only)
- **accent-green** `#68DE7C` – Primary CTA, focus rings, active states
- **accent-orange** `#FF6B3D` – Tertiary CTA, high-priority alerts
- **accent-yellow** `#EAE45B` – Medium-priority alerts, secondary accents

---

## 2. FRAME RULES (Borders & Radii)

### Frame Thickness
- **1px mesh border** – Standard for inputs, cards, containers, dividers
- **2px accent border** – Primary CTAs (green), accent CTAs (orange), active navigation items
- **No border** – Typography-only sections, empty space-based layouts

### Border Radius
- **rounded-2xl** `28px` – Buttons, primary cards, hero sections
- **rounded-xl** `24px` – Secondary cards, metric containers
- **rounded-lg** `20px` – Badge containers, small UI elements
- **rounded** `12px` – Inputs, search boxes
- **rounded-sm** `8px` – Micro UI, dropdown items

### Shadow Depth
- **shadow-sm** `0 2px 4px rgba(0,0,0,0.06)` – Card defaults
- **hover:shadow-md** `0 4px 12px rgba(0,0,0,0.08)` – Card hover
- **Accent glow** `0 0 6px rgba([accent],0.25)` – Active buttons/frames on hover
- **No shadow** – Form inputs at rest; only on focus/error

---

## 3. ACCENT PLACEMENT RULES

### Green Accent (Primary)
**Must appear on:**
- Primary CTAs (white bg + 2px green border)
- Focus rings (2px ring-offset)
- Active navigation items (2px left border)
- Hover state glows (6px soft shadow)

**Never on:**
- Text fills (except interaction cues)
- Filled backgrounds
- Badges (use borders only)

### Orange Accent (Urgent)
**Must appear on:**
- Accent/secondary CTAs (white bg + 2px orange border)
- High/critical priority badges (1px orange border, white bg)
- Error states (input focus ring if error)
- Hover glows on accent buttons

**Never on:**
- Default text
- Filled elements
- Navigation (only green for active)

### Yellow Accent (Moderate)
**Must appear on:**
- Medium-priority badges (1px yellow border, white bg)
- Secondary alerts/warnings
- Informational accents (never fills)

**Never on:**
- CTAs (green or orange only)
- Typography fills
- Filled backgrounds

---

## 4. COMPONENT SPECIFICATIONS

### Buttons

#### Primary Button
```css
bg-white text-ink-1 border-2 border-accent-green rounded-2xl
px-8 py-3 font-semibold text-base
hover:shadow-[0_0_0_6px_rgba(104,222,124,0.25)] hover:-translate-y-0.5
active:translate-y-0 active:shadow-[0_2px_8px_rgba(104,222,124,0.12)]
```

#### Secondary Button
```css
bg-white text-ink-1 border border-mesh rounded-2xl
px-8 py-3 font-medium text-base
hover:border-accent-green hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)]
```

#### Accent Button
```css
bg-white text-ink-1 border-2 border-accent-orange rounded-2xl
px-8 py-3 font-semibold text-base
hover:shadow-[0_0_0_6px_rgba(255,107,61,0.25)] hover:-translate-y-0.5
active:translate-y-0 active:shadow-[0_2px_8px_rgba(255,107,61,0.12)]
```

### Inputs
```css
w-full px-4 py-3 border border-mesh bg-white rounded-lg
text-text-body placeholder:text-text-label
focus:outline-none focus:ring-2 focus:ring-accent-green/20 focus:border-accent-green
transition-all duration-300
```
- Error state: `focus:ring-accent-orange/20 focus:border-accent-orange`
- No colored fills; accent only on focus/error

### Cards
```css
border border-mesh bg-white rounded-2xl p-6 or p-8
shadow-[0_1px_3px_rgba(0,0,0,0.06)]
hover:shadow-[0_4px_12px_rgba(0,0,0,0.08)] hover:-translate-y-1
```
- Metric cards: `rounded-xl border-mesh p-4` (tighter spacing)
- No accent-colored card borders (only mesh)

### Badges & Tags
```css
bg-white border px-2 py-0.5 rounded text-xs font-medium
/* High priority */ border-accent-orange text-ink-1
/* Medium priority */ border-accent-yellow text-ink-1
/* Default/info */ border-mesh text-text-label
/* Active/success */ border-accent-green text-ink-1
```

### Navigation Items (Active)
```css
border-l-2 border-accent-green text-warm-black
py-2 px-3 rounded-lg transition-colors
/* Inactive */ border-transparent text-text-body hover:bg-cream-bg/40
```

### Focus Rings (Global)
```css
button:focus-visible, input:focus-visible, a:focus-visible {
  outline-none ring-2 ring-accent-green ring-offset-2 ring-offset-canvas
}
```

---

## 5. TYPOGRAPHY HIERARCHY

| Element | Size | Weight | Color | Line Height |
|---------|------|--------|-------|-------------|
| H1 | 32px | Bold (700) | ink-1 | 1.2 |
| H2 | 24px | Bold (700) | ink-1 | 1.3 |
| H3 | 20px | Semibold (600) | ink-1 | 1.4 |
| H4 | 16px | Semibold (600) | ink-1 | 1.5 |
| Body Large | 16px | Regular (400) | text-body | 1.6 |
| Body | 14px | Regular (400) | text-body | 1.6 |
| Label | 12px | Semibold (600) | text-label | 1.4 |
| Caption | 12px | Regular (400) | text-label | 1.5 |

- **No color fills for text** (neutral inks only)
- **Accent colors for interaction cues**: underlines, borders, badges only
- **Typography weight** creates hierarchy, not color

---

## 6. AUDIT CHECKLIST

### Forms (Login, Signup, Analyze, Projects)
- [ ] All inputs use 1px mesh border at rest
- [ ] Focus state: 2px accent-green ring + green border (no fills)
- [ ] Error state: focus ring/border = orange
- [ ] Buttons above/below forms: primary (green) or secondary (mesh)
- [ ] Labels use text-label color
- [ ] No color-filled backgrounds for form sections

### Cards & Containers
- [ ] All cards: 1px mesh border, white bg, rounded-2xl
- [ ] Metric containers: rounded-xl with mesh border
- [ ] Active/selected items: 2px green left border (nav) or 2px green frame (cards)
- [ ] No filled backgrounds (white only)
- [ ] Hover shadow present, no color change

### Navigation
- [ ] Active items: 2px green left border, text = ink-1
- [ ] Inactive items: transparent border, text = text-body
- [ ] Hover state: subtle bg-cream-bg/40 (not border preview)
- [ ] No random accent colors in nav

### Buttons & CTAs
- [ ] Primary CTAs: green border, white bg, 3D lift on hover
- [ ] Accent CTAs: orange border, white bg, 3D lift on hover
- [ ] Secondary/ghost: no color frames (mesh only)
- [ ] All buttons rounded-2xl
- [ ] Focus rings apply globally

### Badges & Status Indicators
- [ ] High priority: orange border, white bg
- [ ] Medium priority: yellow border, white bg
- [ ] Default/info: mesh border, white bg
- [ ] Active/success: green border, white bg
- [ ] No filled background colors

### Asides & Sidebars
- [ ] Premium content zones: use spacing/typography only
- [ ] No flashy colored frames
- [ ] Dividers: 1px mesh
- [ ] Background: canvas or surface (white for cards)

---

## 7. WHAT NOT TO DO

❌ **Never:**
- Use solid color fills for buttons (white + border only)
- Apply accent colors to filled backgrounds
- Mix multiple accent colors in one component
- Color text with accent colors (exceptions: underlines, badges, focus)
- Use orange or yellow as primary CTA color (green only)
- Add random borders to non-interactive elements
- Use 3px+ borders for UI elements
- Apply drop-shadows to inputs at rest
- Style cards with colored borders (mesh only)
- Fill badges or tags with background color

---

## 8. IMPLEMENTATION WORKFLOW

1. **Start with neutrals**: canvas, surface, ink colors
2. **Add frames**: 1px mesh by default
3. **Mark interactions**: 2px green for primary, orange for alerts
4. **Layer typography**: hierarchy via weight/size, not color
5. **Test focus states**: all interactive elements have green ring
6. **Verify no fills**: all colored elements are borders/frames only
7. **Check shadows**: only on hover, subtle (shadow-sm → hover:shadow-md)

---

## 9. TAILWIND CONFIG TOKENS

```typescript
colors: {
  'warm-black': '#0F172A',   // ink-1
  'ink-1': '#0F172A',
  'ink-2': '#334155',
  'ink-3': '#64748B',
  'canvas': '#F3F4F6',       // page bg
  'surface': '#FFFFFF',      // cards
  'mesh': '#C9CED6',         // borders
  
  'accent-green': '#68DE7C',   // primary
  'accent-orange': '#FF6B3D',  // urgent
  'accent-yellow': '#EAE45B',  // moderate
  
  'text-heading': '#0F172A',
  'text-body': '#334155',
  'text-label': '#64748B',
  
  'border-default': '#C9CED6', // alias for mesh
  'border-focus': '#68DE7C',   // alias for accent-green
}
```

---

**Version:** 1.0  
**Maintained by:** Design System Team  
**Last Review:** Oct 15, 2025
