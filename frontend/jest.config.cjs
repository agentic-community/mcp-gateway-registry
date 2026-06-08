/** @type {import('jest').Config} */
module.exports = {
  testEnvironment: 'jsdom',
  testMatch: ['<rootDir>/src/**/__tests__/**/*.{ts,tsx}'],
  setupFilesAfterEnv: ['<rootDir>/src/setupTests.ts'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json'],
  moduleNameMapper: {
    '\\.(css|less|scss|sass)$': 'identity-obj-proxy',
    '\\.(png|jpg|jpeg|gif|svg)$': '<rootDir>/src/__mocks__/fileMock.js',
  },
  transform: {
    '^.+\\.(ts|tsx)$': [
      'ts-jest',
      {
        tsconfig: '<rootDir>/tsconfig.test.json',
        useESM: false,
        diagnostics: false,
      },
    ],
  },
  // axios v1 ships ESM-only entrypoints; react-markdown / remark-gfm and
  // their unist/* deps are also ESM. Allow Jest's transformer to process
  // them (default ignores all of node_modules).
  transformIgnorePatterns: [
    '/node_modules/(?!(axios|react-markdown|remark-.*|rehype-.*|unified|bail|is-plain-obj|trough|vfile.*|unist-.*|mdast-.*|micromark.*|decode-named-character-reference|character-entities.*|property-information|hast-util.*|space-separated-tokens|comma-separated-tokens|web-namespaces|zwitch|html-void-elements|ccount|escape-string-regexp|markdown-table|trim-lines)/)',
  ],
};
