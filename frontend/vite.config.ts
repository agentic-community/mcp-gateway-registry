import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Output layout is constrained by two consumers:
//   1. registry/main.py rewrites `="/static/` and `="/favicon.ico"` in index.html
//      when ROOT_PATH is set (path-based routing).
//   2. nginx serves /app/frontend/build/static/ and /app/frontend/build/favicon.ico.
// Keep assetsDir="static" and the rollup output names so this contract holds.
//
// index.html keeps absolute /static/ URLs for that rewrite. URLs Vite writes into
// JS (lazy-route chunks and their CSS, preloaded at runtime) and into CSS (fonts)
// are emitted relative to the file that references them, so they follow
// whatever prefix the entry script was loaded from. With absolute URLs a lazy
// page under ROOT_PATH=/registry would request /static/... and miss the ingress.
// Do not switch `base` to './': the rewrite and deep links need absolute URLs.
export default defineConfig({
  plugins: [react()],
  base: '/',
  experimental: {
    renderBuiltUrl(_filename, { hostType }) {
      return hostType === 'html' ? undefined : { relative: true };
    },
  },
  build: {
    outDir: 'build',
    assetsDir: 'static',
    sourcemap: true,
    rollupOptions: {
      output: {
        entryFileNames: 'static/js/[name].[hash].js',
        chunkFileNames: 'static/js/[name].[hash].js',
        assetFileNames: (assetInfo) => {
          const name = assetInfo.name ?? '';
          if (name.endsWith('.css')) {
            return 'static/css/[name].[hash][extname]';
          }
          return 'static/media/[name].[hash][extname]';
        },
      },
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': 'http://localhost:7860',
      '/auth': 'http://localhost:7860',
      '/oauth2': 'http://localhost:7860',
      '/.well-known': 'http://localhost:7860',
      // Deployment branding (manifest.json, theme.css, logos) comes from the registry.
      '/branding': 'http://localhost:7860',
    },
  },
});
