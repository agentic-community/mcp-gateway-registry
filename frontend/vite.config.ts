import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    port: 3000,
    proxy: {
      // Registry API
      '/api': {
        target: 'http://localhost:7860',
        changeOrigin: true,
        secure: false,
      },
      // EnforceAI management API
      '/enforceai': {
        target: 'http://localhost:7860',
        changeOrigin: true,
        secure: false,
      },
      // OAuth2 endpoints
      '/oauth2': {
        target: 'http://localhost:7860',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'build',
    sourcemap: true,
  },
});
