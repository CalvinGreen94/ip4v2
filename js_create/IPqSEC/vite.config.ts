import { defineConfig } from 'vite';

export default defineConfig({
  optimizeDeps: {
    esbuildOptions: {
      define: {
        'crypto.hash': 'crypto.createHash',
      },
    },
  },
});