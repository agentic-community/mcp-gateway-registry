// humanize("trigger_type")   → "Trigger Type"    (split on _ and -)
// humanize("owner-team")     → "Owner Team"
// humanize("workflowBody")   → "Workflow Body"   (split on camelCase boundary)
// humanize("url")            → "Url"             (single word, title-case only — no acronym logic in v1)
export function humanize(name: string): string {
  return name
    .replace(/([a-z])([A-Z])/g, '$1 $2') // camelCase split
    .replace(/[_-]+/g, ' ') // underscore/dash → space
    .replace(/\b\w/g, (c) => c.toUpperCase()) // title-case each word
    .trim();
}

export function labelFor(field: { label?: string | null; name: string }): string {
  return field.label ?? humanize(field.name);
}
