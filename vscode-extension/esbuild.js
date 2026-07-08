const esbuild = require("esbuild");

const watch = process.argv.includes("--watch");

esbuild.build({
  entryPoints: ["src/extension.ts"],
  bundle: true,
  outfile: "dist/extension.js",
  external: ["vscode"],
  format: "cjs",
  platform: "node",
  sourcemap: true,
  watch: watch ? { onRebuild(err) { if (err) console.error(err); else console.log("rebuilt"); } } : false,
}).catch(() => process.exit(1));
