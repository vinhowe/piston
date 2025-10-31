import tsconfigPaths from "vite-tsconfig-paths";
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    projects: [
      {
        plugins: [tsconfigPaths()],
        test: {
          name: "client",
          include: ["src/**/*.browser.test.ts"],
          setupFiles: ["./vitest-client-setup.ts"],
          browser: {
            enabled: true,
            provider: "playwright",
            instances: [
              {
                browser: "chromium",
              },
            ],
          },
        },
      },
      {
        plugins: [tsconfigPaths()],
        test: {
          name: "server",
          environment: "node",
          include: ["src/**/*.test.ts"],
          exclude: ["src/**/*.browser.test.ts"],
          setupFiles: ["./vitest-server-setup.ts"],
        },
      },
    ],
  },
});
