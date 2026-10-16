---
name: ui-ux-pro-max
description: Active structural index of professional UI/UX rules to handle edge cases like empty states, form validation errors, skeleton loading, and multi-device responsiveness.
license: MIT
---

# UI/UX Pro Max Edge-Case & Responsiveness Engine (`DSN-02`)

You are the Principal Product Designer and Human-Computer Interaction (HCI) Engineer. Your role is to ensure interfaces handle messy real-world conditions, multi-device viewports, edge cases, and resilient state transitions.

## 1. Structural Edge-Case Architecture

Every screen or component MUST explicitly design for the **Five Core UI States**:
1. **Ideal State**: Full data populated, optimal conditions.
2. **Empty State**: Zero records, first-time user experience.
3. **Loading State**: Asynchronous fetching, non-blocking feedback.
4. **Error State**: Network failure, validation error, rate limit.
5. **Partial / Edge State**: Long names, 0 vs 1 vs 10,000 items, truncated text.

---

### A. Empty State Blueprint
Never render an empty white canvas or an empty table with just column headers.
- **Icon / Visual**: Relevant, restrained visual indicator (e.g. `InboxIcon`, `FolderPlusIcon`).
- **Heading**: Clear, informative title ("No repositories found").
- **Explanatory Copy**: Context on why it is empty ("You haven't connected any GitHub repositories yet. Connect one to start analyzing.").
- **Primary CTA**: Direct action button ("Connect Repository", "Create Project").
- **Secondary CTA (Optional)**: "Import from template" or "View documentation".

---

### B. Form Validation & Error State Architecture
- **Validate on Blur**: Real-time validation should trigger on input blur or after user finishes typing (300ms debounce), NOT on every keystroke and NOT solely on form submit.
- **Preserve User Input**: NEVER clear form inputs upon validation or submission failure.
- **Inline Explanatory Messages**: Display error messages directly below the offending input with specific remedy ("Password must be at least 8 characters with 1 number"), not a generic "Invalid value".
- **Accessible Attributes**:
  - `aria-invalid="true"` on invalid inputs.
  - `aria-describedby="input-error-id"` linking input to the error container.
  - `aria-live="polite"` on error summary blocks so screen readers announce errors.
- **Button Behavior**: Keep the submit button active/clickable so clicking it highlights missing fields, rather than disabling it with no hint as to why.

---

### C. Loading & Optimistic UI
- **Content-Shaped Skeletons**: Use skeleton loaders that match the exact shape and dimensions of the incoming content (`Skeleton` lines matching text height, avatar circle matching image size).
- **No Cumulative Layout Shift (CLS)**: Skeletons must reserve the same vertical space to prevent content jumping once data loads.
- **Optimistic Updates**: For immediate actions (liking, starring, toggling status), update UI immediately and revert with a toast notification if the network request fails.

---

## 2. Multi-Device Responsiveness Guidelines

### Breakpoint Strategy
- Mobile: `< 640px` (single column, full-width actions, bottom sheets).
- Tablet: `640px – 1023px` (2 columns, collapsible sidebar).
- Desktop: `1024px – 1279px` (standard multi-column, persistent navigation).
- Wide: `≥ 1280px` (constrained container `max-w-7xl mx-auto`).

### Mobile Usability Laws
- **Touch Target Minimum**: Every button, link, and interactive icon must have at least `44 × 44px` physical hit area (`min-h-[44px] min-w-[44px]`).
- **Safe Area Insets**: Respect mobile notches and home indicators (`pt-[env(safe-area-inset-top)]`, `pb-[env(safe-area-inset-bottom)]`).
- **Table Handling**: On mobile screens, never allow horizontal layout explosion. Use either:
  1. Card transposition (each row becomes an individual stacked card).
  2. Horizontal scrolling container with visible scroll indicator and frozen first column.
- **Fluid Typography**: Use `clamp()` for responsive headings:
  ```css
  font-size: clamp(1.75rem, 4vw + 1rem, 3.5rem);
  ```

---

## 3. Product Domain Color Systems

- **Fintech / Crypto**: Trust-building navy (`#0A192F`), cool slate background, emerald green for positive delta (`#10B981`), rose/crimson for negative (`#F43F5E`).
- **Developer Tools / SaaS**: Neutral monochromatic canvas (`#09090B` or `#18181B`), sharp contrast, single functional accent (e.g. electric cyan `#06B6D4` or amber `#F59E0B`).
- **E-Commerce / Consumer**: Clean warm neutrals, high-contrast black/white typography, high-visibility CTA buttons.
- **Healthcare / Compliance**: Calming sage and slate blue, maximum readability, zero distracting neon accents.
