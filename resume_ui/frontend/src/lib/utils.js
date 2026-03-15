/**
 * Merges class names, filtering out falsy values.
 * Lightweight cn() utility (no Tailwind merge needed for this project).
 */
export function cn(...classes) {
  return classes.filter(Boolean).join(" ");
}
