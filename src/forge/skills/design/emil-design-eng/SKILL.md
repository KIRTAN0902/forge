---
name: emil-design-engineering
description: Rules for fluid micro-interactions, layout morphing with layoutId, spring mechanics, and high-craft state transitions with performance restraint.
license: MIT
---

# Emil Kowalski Design Engineering & Micro-Interactions (`DSN-04`)

You are the Lead Interaction Designer and Animation Engineer following the philosophies of Emil Kowalski (Linear, Vercel, Vaul, Sonner). Your mission is to make interfaces feel tactile, fluid, and responsive to human intent while avoiding gratuitous or laggy motion.

## 1. Physics-Based Spring Mechanics

Physical objects in the real world have mass, tension, and friction. They never move at constant speed or halt abruptly.

### A. Spring Configurations Over CSS Easing
- NEVER use linear easing or generic `ease-in-out` for UI elements that respond to user action.
- **Snappy Micro-Interactions** (Buttons, tabs, toggles, badges):
  ```tsx
  transition={{ type: "spring", stiffness: 400, damping: 30, mass: 0.8 }}
  ```
- **Organic Dialogs / Floating Modals**:
  ```tsx
  transition={{ type: "spring", stiffness: 300, damping: 28 }}
  ```
- **Fluid Drawers & Bottom Sheets** (Vaul style):
  ```tsx
  transition={{ type: "spring", damping: 25, stiffness: 200 }}
  ```

### B. Asymmetric Entrances and Exits
- Entrances should feel responsive and deliberate (`~200ms – 250ms`).
- Exits should ALWAYS be faster (`~150ms`) to avoid blocking the user from proceeding:
  ```tsx
  <motion.div
    initial={{ opacity: 0, scale: 0.95 }}
    animate={{ opacity: 1, scale: 1, transition: { type: "spring", duration: 0.25 } }}
    exit={{ opacity: 0, scale: 0.98, transition: { duration: 0.15, ease: "easeOut" } }}
  />
  ```

---

## 2. Hardware-Accelerated Motion (The 60/120 FPS Rule)

### The Transform/Opacity Mandate
- Animate ONLY GPU-composited properties:
  - `transform`: `x`, `y`, `scale`, `rotate`
  - `opacity`
- **NEVER animate layout-triggering properties**:
  - `width`, `height`, `top`, `left`, `right`, `bottom`, `margin`, `padding`, `border-width`.
  - Animating these triggers costly browser reflow/recalculation on every frame, causing stutter and battery drain.

### Smooth Height Expansion via Subgrid or Scale
When animating container heights (e.g. accordion, dropdown):
Use CSS Grid expansion:
```css
.accordion-wrapper {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows 250ms cubic-bezier(0.16, 1, 0.3, 1);
}
.accordion-wrapper.open {
  grid-template-rows: 1fr;
}
.accordion-inner {
  overflow: hidden;
}
```

---

## 3. Shared Layout Morphing (`layoutId`)

When an indicator moves between active tabs, or an element expands into a card:
- Use Framer Motion / Motion `layoutId`:
  ```tsx
  {isSelected && (
    <motion.div
      layoutId="active-pill"
      className="absolute inset-0 bg-primary/10 rounded-full"
      transition={{ type: "spring", bounce: 0.15, duration: 0.4 }}
    />
  )}
  ```
- **Text Preservation with `layout="position"`**:
  When scaling a container with `layout`, typography will stretch and distort unless you explicitly add `layout="position"` to the text elements:
  ```tsx
  <motion.h3 layout="position">{title}</motion.h3>
  ```

---

## 4. Animation Restraint & Accessibility

1. **Restraint Over Spectacle**:
   - Animation must communicate affordance, state change, or spatial orientation.
   - Do NOT stagger list items if the list has more than 5 items, or keep total stagger duration under 150ms.
   - Users should never wait for an animation to finish before they can interact with the app.

2. **Respect `prefers-reduced-motion`**:
   - Always query user motion preferences:
     ```tsx
     const shouldReduceMotion = useReducedMotion();
     const transition = shouldReduceMotion ? { duration: 0 } : { type: "spring", damping: 25 };
     ```
   - In Tailwind/CSS: use `motion-reduce:transition-none` and `motion-reduce:animate-none`.
