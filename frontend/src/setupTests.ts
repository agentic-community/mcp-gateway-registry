import '@testing-library/jest-dom';
import { TextDecoder, TextEncoder } from 'util';

// react-router 7 needs TextEncoder at import time; jsdom does not provide it.
if (typeof globalThis.TextEncoder === 'undefined') {
  Object.assign(globalThis, { TextEncoder, TextDecoder });
}

// jsdom doesn't implement URL.createObjectURL/revokeObjectURL.
// SkillResources tests exercise blob-download paths that touch these.
if (typeof window.URL.createObjectURL === 'undefined') {
  window.URL.createObjectURL = jest.fn(() => 'blob:mock');
  window.URL.revokeObjectURL = jest.fn();
}
