// Shared helpers for path-based deployments (e.g. ROOT_PATH=/registry).
// The server injects <base href="{ROOT_PATH}/"> into index.html; both
// the router basename and the API base URL are derived from it.

export const getBaseURL = (): string => {
  const baseTag = document.querySelector('base');
  if (baseTag && baseTag.href) {
    const url = new URL(baseTag.href);
    return url.pathname.replace(/\/$/, '');
  }
  return '';
};

export const getBasename = (): string => {
  return getBaseURL() || '/';
};

export const apiUrl = (path: string): string => {
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${getBaseURL()}${normalized}`;
};
