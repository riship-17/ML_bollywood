#!/usr/bin/env node
/**
 * Copies the model weights and WebAssembly binaries into public/ so the app
 * serves them from its own origin.
 *
 * Both sets already arrive as part of installed packages, so nothing is
 * downloaded and nothing large has to live in git:
 *
 *   @vladmandic/face-api/model/*            -> public/models/
 *   @tensorflow/tfjs-backend-wasm/dist/*.wasm -> public/tfjs/
 *
 * Runs automatically before `next build`. Missing source packages are a hard
 * error during a build and a warning otherwise, so a bare `npm install` on a
 * machine that skipped optional packages still succeeds.
 */

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

// Only the nets the app actually loads. tiny_face_detector, age/gender and
// expression models are deliberately left out — nothing references them and
// they would just be dead weight on the deployment.
const FACE_API_FILES = [
  "ssd_mobilenetv1_model-weights_manifest.json",
  "ssd_mobilenetv1_model.bin",
  "face_landmark_68_model-weights_manifest.json",
  "face_landmark_68_model.bin",
  "face_recognition_model-weights_manifest.json",
  "face_recognition_model.bin",
];

const WASM_FILES = [
  "tfjs-backend-wasm.wasm",
  "tfjs-backend-wasm-simd.wasm",
  "tfjs-backend-wasm-threaded-simd.wasm",
];

function packageDir(pkg) {
  return path.dirname(require.resolve(`${pkg}/package.json`));
}

async function copyGroup(label, pkg, fromSubdir, files, destDir) {
  let sourceDir;
  try {
    sourceDir = path.join(packageDir(pkg), fromSubdir);
  } catch {
    return { label, pkg, copied: 0, missing: true };
  }

  await fs.mkdir(destDir, { recursive: true });

  let copied = 0;
  let bytes = 0;
  for (const name of files) {
    const from = path.join(sourceDir, name);
    const to = path.join(destDir, name);
    try {
      await fs.copyFile(from, to);
      bytes += (await fs.stat(to)).size;
      copied++;
    } catch (err) {
      throw new Error(`Could not copy ${name} from ${pkg}: ${err.message}`);
    }
  }

  return { label, pkg, copied, bytes, missing: false };
}

async function main() {
  const root = process.cwd();
  const results = [
    await copyGroup(
      "face-api weights",
      "@vladmandic/face-api",
      "model",
      FACE_API_FILES,
      path.join(root, "public", "models"),
    ),
    await copyGroup(
      "tfjs wasm backend",
      "@tensorflow/tfjs-backend-wasm",
      "dist",
      WASM_FILES,
      path.join(root, "public", "tfjs"),
    ),
  ];

  const missing = results.filter((r) => r.missing);
  const building = process.env.npm_lifecycle_event === "prebuild";

  for (const r of results) {
    if (r.missing) continue;
    console.log(`  ${r.label}: ${r.copied} files, ${(r.bytes / 1024 / 1024).toFixed(1)} MB`);
  }

  if (missing.length === 0) return;

  // The face detector is required; the wasm fallback is not.
  const fatal = missing.find((m) => m.pkg === "@vladmandic/face-api") || building;

  const note = missing.map((m) => `${m.label} (${m.pkg} not installed)`).join(", ");

  if (fatal) {
    console.error(
      `\nMissing required assets: ${note}\n` +
        "Run `npm install` so the packages the weights come from are present.\n",
    );
    process.exit(1);
  }

  console.warn(
    `\n  Skipped: ${note}.\n` +
      "  The app still runs on WebGL; only the WebAssembly fallback is unavailable.\n",
  );
}

main().catch((err) => {
  console.error(err.message ?? err);
  process.exit(1);
});
