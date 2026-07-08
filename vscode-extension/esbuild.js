const esbuild = require("esbuild");

const shared = {
  bundle: true,
  external: ["vscode"],
  format: "cjs",
  platform: "node",
  sourcemap: true,
};

Promise.all([
  esbuild.build({
    ...shared,
    entryPoints: ["src/extension.ts"],
    outfile: "dist/extension.js",
  }),
  esbuild.build({
    ...shared,
    entryPoints: ["src/test/runTest.ts"],
    outfile: "dist/test/runTest.js",
  }),
  esbuild.build({
    ...shared,
    entryPoints: ["src/test/suite/index.ts"],
    outfile: "dist/test/suite/index.js",
  }),
  esbuild.build({
    ...shared,
    entryPoints: ["src/test/suite/apiClient.test.ts"],
    outfile: "dist/test/suite/apiClient.test.js",
  }),
]).catch(() => process.exit(1));
