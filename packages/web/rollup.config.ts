import commonjs from "@rollup/plugin-commonjs";
import resolve from "@rollup/plugin-node-resolve";
import terser from "@rollup/plugin-terser";
import typescript from "@rollup/plugin-typescript";
import { replaceTscAliasPaths } from "tsc-alias";

/**
 * Creates a Rollup plugin that replaces TypeScript alias paths in compiled output
 * @param {import("tsc-alias").ReplaceTscAliasPathsOptions} options - Configuration options for tsc-alias
 * @returns {Object} Rollup plugin
 */
function tscAlias(options) {
  return {
    name: "tscAlias",
    async writeBundle() {
      return replaceTscAliasPaths(options);
    },
  };
}

const createConfig = ({
  format,
  input,
  outputFile,
  browser = false,
  minify = false,
}) => {
  const plugins = [
    // Resolve node modules
    resolve({
      browser,
      preferBuiltins: !browser,
    }),
    // Convert CommonJS modules to ES modules
    commonjs(),
    // Process TypeScript files
    typescript({
      tsconfig: "./tsconfig.json",
      sourceMap: true,
      declaration: true,
      declarationDir: "dist/types",
      rootDir: "./src",
    }),
    tscAlias({
      outDir: "dist/types",
      declarationDir: "dist/types",
    }),
  ];

  // Add minification for production builds
  if (minify) {
    plugins.push(terser({
      mangle: {
        // We do this so that we can keep track of module scopes
        keep_classnames: true,
      },
    }));
  }

  return {
    input,
    output: {
      file: outputFile,
      format,
      sourcemap: true,
      // Preserve imported WASM module names for better debugging
      interop: "auto",
      // Export individual named exports in ES format
      exports: "named",
      // This is important because we need to match on class names and parameter names for the
      // capture session.
      minifyInternalExports: false,
    },
    // Make sure we externalize the @piston-ml/piston-web dependency, and the CodeMirror/Lezer
    // packages.
    external: [
      "@piston-ml/piston-web-wasm",
      /@codemirror\/[^/]+/,
      // /@lezer\/[^/]+/,
      "codemirror"
    ],
    plugins,
    // Ensure WASM imports are properly handled
    onwarn(warning, warn) {
      // Ignore certain warnings
      if (warning.code === "CIRCULAR_DEPENDENCY") return;
      // Ignore dynamic import warnings
      if (
        warning.code === "UNRESOLVED_IMPORT" &&
        warning.source?.includes("@piston-ml/piston-web-wasm")
      )
        return;
      warn(warning);
    },
  };
};

export default [
  // ESM build for Node.js
  createConfig({
    format: "es",
    input: "src/index.ts",
    outputFile: "dist/piston.module.mjs",
  }),

  // Browser ESM build
  createConfig({
    format: "es",
    input: "src/browser.ts",
    outputFile: "dist/browser.module.mjs",
    browser: true,
    minify: true,
  }),
];
