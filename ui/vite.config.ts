import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwind from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwind()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  server: {
    port: 5173,
    proxy: {
      // The FastAPI backend runs on :8000 in dev — proxy /api so we don't
      // need to think about CORS during local development.
      "/api": "http://127.0.0.1:8000",
    },
  },
});
