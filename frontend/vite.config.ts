import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// Prefer going through the gateway nginx (port 80) so /enforceai and /mcpgw work.
// Override with VITE_GATEWAY_PROXY_TARGET (e.g. http://localhost:80).
//
// eslint-disable-next-line no-process-env
const gatewayTarget = process.env.VITE_GATEWAY_PROXY_TARGET || 'http://localhost';

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
        target: gatewayTarget,
        changeOrigin: true,
        secure: false,
      },
      // EnforceAI management API
      '/enforceai': {
        target: gatewayTarget,
        changeOrigin: true,
        secure: false,
      },
      // OAuth2 endpoints
      '/oauth2': {
        target: gatewayTarget,
        changeOrigin: true,
        secure: false,
      },
      // MCP Gateway (streamable HTTP/SSE)
      '/mcpgw': {
        target: gatewayTarget,
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
