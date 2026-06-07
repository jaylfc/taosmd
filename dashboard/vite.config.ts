import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: path.resolve(__dirname, "../taosmd/webui"),
    emptyOutDir: true,
    assetsDir: "assets",
  },
});
