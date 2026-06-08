import '@testing-library/jest-dom';

// jsdom doesn't implement URL.createObjectURL/revokeObjectURL.
// SkillResources tests exercise blob-download paths that touch these.
if (typeof window.URL.createObjectURL === 'undefined') {
  window.URL.createObjectURL = jest.fn(() => 'blob:mock');
  window.URL.revokeObjectURL = jest.fn();
}
