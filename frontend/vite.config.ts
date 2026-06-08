import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Output layout is constrained by two consumers:
//   1. registry/main.py rewrites `="/static/` and `="/favicon.ico"` in index.html
//      when ROOT_PATH is set (path-based routing).
//   2. nginx serves /app/frontend/build/static/ and /app/frontend/build/favicon.ico.
// Keep assetsDir="static" and the rollup output names so this contract holds.
export default defineConfig({
  plugins: [react()],
  base: '/',
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
    },
  },
});
