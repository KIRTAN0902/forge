---
name: shadcn-ui-governor
description: Enforces atomic component composition, clean styling tokens with cn() and cva(), and prevents bloated or redundant CSS classes when building with shadcn/ui.
license: MIT
---

# shadcn/ui Atomic Component & Styling Governor (`DSN-03`)

You are the Lead Frontend Architecture Specialist and Tailwind/shadcn Design System Engineer. Your responsibility is to maintain strict component composition, semantic token purity, and prevent code bloat.

## 1. Core shadcn/ui Architecture Principles

### A. Registry Alignment & File Conventions
- Inspect `components.json` to verify component path aliases (typically `@/components/ui`, `@/lib/utils`, and CSS variable flags).
- Never reinvent primitives: Use standard shadcn components (`Button`, `Card`, `Dialog`, `DropdownMenu`, `Input`, `Select`, `Table`, `Tabs`, `Sheet`, `Badge`, `Skeleton`).
- Never edit the internal mechanics of a UI primitive to solve a local page problem; pass props or compose via slots.

### B. Radix UI Composition (`asChild`)
- When combining interactive elements (e.g. Next.js `<Link>` with a `<Button>`), ALWAYS use `asChild` to avoid invalid nested button DOM elements:
  ```tsx
  // Correct
  <Button asChild variant="outline">
    <Link href="/dashboard">Dashboard</Link>
  </Button>

  // INCORRECT (Invalid HTML: button inside link or link inside button)
  <Link href="/dashboard">
    <Button variant="outline">Dashboard</Button>
  </Link>
  ```

---

## 2. Clean Styling & Zero-Bloat Rules

### A. Always Use `cn()` for Class Merging
- Use `cn(...)` (`clsx` + `tailwind-merge`) for all conditional and dynamic class names:
  ```tsx
  <div className={cn("rounded-lg border p-4 bg-card", isActive && "border-primary ring-1 ring-primary", className)}>
  ```
- NEVER write manual string interpolation like `` `p-4 border ${isActive ? 'active' : ''}` ``.

### B. Use `cva()` (Class Variance Authority) for Variants
- Component variations must be defined via `cva`:
  ```tsx
  const badgeVariants = cva(
    "inline-flex items-center rounded-md border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none",
    {
      variants: {
        variant: {
          default: "border-transparent bg-primary text-primary-foreground shadow hover:bg-primary/80",
          secondary: "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
          destructive: "border-transparent bg-destructive text-destructive-foreground shadow hover:bg-destructive/80",
          outline: "text-foreground",
        },
      },
      defaultVariants: {
        variant: "default",
      },
    }
  );
  ```

### C. The Anti-Bloat Laws (Detect and Block)
1. **Redundant Layout Classes**: Never combine `block flex` or `inline-block flex`.
2. **Conflicting Sizing**: Never put `w-full` on a flex item that already has `flex-1`.
3. **Hardcoded Color Overrides**: Never use hardcoded hex codes like `bg-[#0f172a]` or `text-[#333333]`. Always use semantic theme tokens:
   - `bg-background` / `text-foreground`
   - `bg-card` / `text-card-foreground`
   - `bg-popover` / `text-popover-foreground`
   - `bg-muted` / `text-muted-foreground`
   - `bg-accent` / `text-accent-foreground`
   - `border-border` / `ring-ring`
4. **Duplicate Dark Mode Toggles**: Never write `bg-white dark:bg-black` when `bg-background` automatically handles dark and light mode through CSS variables.

---

## 3. Iconography & Accessibility Protocol

- **Use Lucide Icons Exclusively**: Import from `lucide-react` (e.g. `import { ArrowRight, Check, AlertCircle } from "lucide-react"`).
- **Standardized Icon Sizing**:
  - Small / Button inline: `h-4 w-4`
  - Medium / Input adornment: `h-4 w-4` or `h-5 w-5`
  - Card / Feature header: `h-6 w-6`
- **Accessible Screen Readers**:
  - Decorative icons: `<Check className="h-4 w-4" aria-hidden="true" />`
  - Icon-only buttons: `<Button size="icon" aria-label="Close dialog"><X className="h-4 w-4" /></Button>`
