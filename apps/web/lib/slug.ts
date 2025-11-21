const NON_LATIN_REGEX = /[\u0300-\u036f]/g;
const NON_ALPHANUMERIC_REGEX = /[^a-z0-9]+/g;

/**
 * Convert arbitrary text into a URL-safe slug.
 */
export function toSlug(value: string, fallback: string = 'project'): string {
  if (!value) {
    return fallback;
  }

  const normalized = value
    .toLowerCase()
    .normalize('NFKD')
    .replace(NON_LATIN_REGEX, '')
    .replace(NON_ALPHANUMERIC_REGEX, '-')
    .replace(/^-+|-+$/g, '');

  return normalized || fallback;
}

/**
 * Determines whether an identifier consists purely of digits.
 */
export function isNumericIdentifier(value: string): boolean {
  return /^\d+$/.test(value);
}
