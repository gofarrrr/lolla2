# Lolla Design System — Coherence Spec

## Logo & Nav (all pages)
- **Logo**: "lolla" lowercase, 16–18px, Inter, ink-1, no icon.
- **PermanentNav**: 
  - bg-white, border-b 1px mesh, sticky top-0 z-50.
  - Left: logo + nav links (Reports, Academy, Blog).
  - Right: Glass Box toggle + Settings + user avatar (8px circle, bg-accent-green).
  - Link active: underline in green (no background), 2px thickness.
  - No shadow; clean minimal.

## Button Hierarchy
### Primary CTA (most important: Sign up, Log in, Start Analysis, Create, Submit)
```
Base: bg-white, border-2 border-accent-green, text-ink-1, rounded-2xl, px-8 py-3, font-semibold.
Hover: 
  - box-shadow: 0 8px 24px rgba(104, 222, 124, 0.20) (drop-up effect)
  - transform: translateY(-2px) (lifts)
Pressed (active):
  - transform: translateY(0) (back to baseline)
  - box-shadow: 0 2px 8px rgba(104, 222, 124, 0.12) (subtle)
Disabled: opacity-50, cursor-not-allowed
```

### Secondary Button (Cancel, Learn more, Close)
```
Base: bg-white, border-1 border-mesh, text-ink-1, rounded-2xl, px-8 py-3, font-medium.
Hover: border-accent-green (no shadow, no lift).
No 3D effect.
```

### Ghost / Link-style
```
Base: bg-transparent, text-ink-1, underline-none.
Hover: underline-1 green.
```

## Frames / Borders (Ultra-subtle)
- **Inputs**: 1px border-mesh, focus:ring-2 ring-accent-green.
- **Cards**: 1px border-mesh (all cards: report, project, dashboard).
- **Sections**: 1px border-mesh (only structural breaks, not decoration).
- **Alerts/Errors**: 2px border-accent-orange, white bg, 1px left-accent-orange bar (1px thick, full height).
- **No decorative frames**: no random 2px borders, no colored halos on non-interactive elements.

## Color Placement (Iron Rules)
- **Green (#68DE7C)**: 
  - Focus rings (inputs, buttons).
  - Active nav underline.
  - Primary CTA border.
  - Selected/active states (checkbox active, card selected).
  - Nowhere else.
- **Orange (#FF6B3D)**:
  - Error borders.
  - Alert backgrounds (minimal use).
  - Validation feedback.
  - Nowhere else.
- **No yellow** anywhere in the system.

## Spacing & Layout
- **Viewport constraint**: no scrolling on login, signup, analyze, or 404 pages.
- **Page margins**: 
  - Desktop: max-w-wide (1440px), px-6 horizontal padding.
  - Mobile: px-4, stack vertically.
- **Section spacing**: 
  - Reduced from current (48px → 32px or 24px where possible).
  - Cards within sections: gap-4 or gap-6 (not gap-8).
  - Asides/hero sections: condense to fit viewport.
- **No excess padding**: audit all `py-12`, `py-16`—reduce to `py-8` or `py-10`.

## Radius & Shadows
- **Border-radius**: `rounded-2xl` (28px) for buttons, cards, inputs. `rounded-xl` (24px) for smaller components.
- **Shadows**:
  - Buttons (primary hover): `shadow-[0_8px_24px_rgba(104,222,124,0.20)]`.
  - Cards (no hover shadow): `shadow-sm` only.
  - Inputs (no shadow).
  - Focus states: ring-2 (outline effect, not shadow).

## Typography
- **Headings**: ink-1, semibold, generous line-height (1.25–1.6).
- **Body**: ink-2 (#334155), regular, line-height 1.6.
- **Labels**: ink-3 (#64748B), 12px, uppercase, tracking-wider.
- No colored text (headings, body, labels all neutral).

## Components Audit Checklist

### PermanentNav
- [ ] Logo: "lolla" lowercase, no icon.
- [ ] Links: green underline on active (no bg).
- [ ] No shadow; 1px border-b mesh.

### Buttons (all pages)
- [ ] Primary CTA: 2px green border, 3D press effect (lift 2px, drop shadow).
- [ ] Secondary: 1px mesh border, no effect.
- [ ] Consistent px-8 py-3, rounded-2xl, font-semibold (primary) / font-medium (secondary).

### Inputs (forms)
- [ ] 1px border-mesh, focus:ring-2 ring-accent-green.
- [ ] No shadow.
- [ ] Placeholder text: ink-3.

### Cards
- [ ] All cards: 1px border-mesh, bg-white, rounded-2xl.
- [ ] No hover shadow (or sm only).
- [ ] Consistent padding (p-6 or p-8).

### Alerts / Errors
- [ ] 2px border-accent-orange, white bg.
- [ ] 1px left bar (orange, full height).
- [ ] Text: ink-1, no red fills.

### Asides (login, signup)
- [ ] White bg, 1px border-mesh (or no border, emphasize via spacing).
- [ ] Faint mesh watermark (6% opacity, background-only).
- [ ] No colored borders or fills.
- [ ] Typography hierarchy over decoration.

### Report V2
- [ ] LeftSidebar: 2px left border (green) on active items, no background.
- [ ] Cards: 1px border-mesh.
- [ ] Badges: white bg, 1px border-mesh or border-accent-green (if active).
- [ ] No colored fills on cards or sections.

### Pages (no scroll)
- [ ] Login: max-h-screen, form + aside fit viewport.
- [ ] Signup: same as login.
- [ ] Analyze: form only, fits viewport.
- [ ] 404: minimal, no scroll.

## Implementation Order
1. Update PermanentNav (logo, nav links, styling).
2. Refactor Button component (primary 3D effect, secondary minimal).
3. Audit all inputs (1px mesh, focus ring).
4. Standardize card styling (1px mesh border, consistent padding).
5. Fix alerts (orange border, no red fills).
6. Condense login/signup asides (reduce margins, fit viewport).
7. Audit report_v2 (sidebar active states, card borders).
8. Global spacing pass (reduce py-12/16 → py-8/10).
9. Remove all decorative frames (audit for random borders).
10. Visual test across all pages.
