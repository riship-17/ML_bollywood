#!/usr/bin/env node
/**
 * Turns a folder of celebrity images into public/celebrities.json.
 *
 * Expected layout (this is how the Kaggle "Bollywood Celebrity Faces" dataset
 * is already organised):
 *
 *   <dataset>/
 *     Shah_Rukh_Khan/
 *       0001.jpg
 *       0002.jpg
 *     Alia_Bhatt/
 *       ...
 *
 * Usage:
 *   npm run embeddings:setup
 *   npm run embeddings -- --input ./datasets/bollywood_celeb_faces
 *
 * Options:
 *   --input <dir>      Dataset root. Required.
 *   --out <file>       Output JSON. Default: public/celebrities.json
 *   --thumbs <dir>     Also write one representative thumbnail per person.
 *                      Default: skipped. See README on image rights.
 *   --max-per-person N Cap images per person (speed). Default: 40
 *   --min-faces N      Drop people with fewer than N usable faces. Default: 2
 *   --backend <name>   Force "wasm" instead of trying native tfjs-node first.
 *
 * Only the resulting 128 numbers per person are needed by the app. No source
 * image is copied anywhere unless you explicitly pass --thumbs.
 */

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const IMAGE_EXT = new Set([".jpg", ".jpeg", ".png", ".webp", ".bmp"]);

function parseArgs(argv) {
  const args = {
    input: null,
    out: "public/celebrities.json",
    thumbs: null,
    maxPerPerson: 40,
    minFaces: 2,
    backend: null,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => {
      const v = argv[++i];
      if (v === undefined) throw new Error(`Missing value after ${a}`);
      return v;
    };
    if (a === "--input" || a === "-i") args.input = next();
    else if (a === "--out" || a === "-o") args.out = next();
    else if (a === "--thumbs") args.thumbs = next();
    else if (a === "--max-per-person") args.maxPerPerson = Number(next());
    else if (a === "--min-faces") args.minFaces = Number(next());
    else if (a === "--backend") args.backend = next();
    else if (a === "--help" || a === "-h") args.help = true;
    else throw new Error(`Unknown argument: ${a}`);
  }
  return args;
}

function slugify(name) {
  return name.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

function prettifyName(raw) {
  return raw
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((w) => (w.length === 0 ? w : w[0].toUpperCase() + w.slice(1).toLowerCase()))
    .join(" ");
}

function meanDescriptor(descriptors) {
  if (descriptors.length === 0) return null;
  const dims = descriptors[0].length;
  const out = new Array(dims).fill(0);
  for (const d of descriptors) {
    for (let i = 0; i < dims; i++) out[i] += d[i];
  }
  for (let i = 0; i < dims; i++) out[i] /= descriptors.length;
  // Round to 6 dp — keeps the JSON a third of the size with no measurable
  // effect on cosine similarity.
  return out.map((v) => Math.round(v * 1e6) / 1e6);
}

/**
 * Two ways to run the models under Node:
 *
 *  native — @tensorflow/tfjs-node, several times faster, but its install
 *           compiles/downloads a libtensorflow binary that fails on locked-down
 *           networks and some platforms.
 *  wasm   — pure JavaScript and WebAssembly, installs anywhere, slower.
 *
 * Try native, fall back to wasm, and only give up if neither is present.
 */
async function loadDeps(preferred) {
  let sharp;
  try {
    sharp = (await import("sharp")).default;
  } catch (err) {
    console.error(
      "\nsharp is required to read images.\n  npm run embeddings:setup\n\n" +
        `Underlying error: ${err.message}\n`,
    );
    process.exit(1);
  }

  const attempts = preferred === "wasm" ? ["wasm"] : ["native", "wasm"];
  const failures = [];

  for (const backend of attempts) {
    try {
      if (backend === "native") {
        const tf = await import("@tensorflow/tfjs-node");
        const mod = await import("@vladmandic/face-api/dist/face-api.node.js");
        return { tf, faceapi: mod.default ?? mod, sharp, backend };
      }
      const tf = await import("@tensorflow/tfjs");
      const wasm = await import("@tensorflow/tfjs-backend-wasm");
      const mod = await import("@vladmandic/face-api/dist/face-api.node-wasm.js");
      const faceapi = mod.default ?? mod;

      // Point the runtime at the .wasm binaries inside the installed package.
      const { createRequire } = await import("node:module");
      const require = createRequire(import.meta.url);
      const wasmDir = path.join(
        path.dirname(require.resolve("@tensorflow/tfjs-backend-wasm/package.json")),
        "dist",
      );
      wasm.setWasmPaths(`${wasmDir}${path.sep}`);
      await tf.setBackend("wasm");
      await tf.ready();
      return { tf, faceapi, sharp, backend };
    } catch (err) {
      failures.push(`  ${backend}: ${err.message}`);
    }
  }

  console.error(
    "\nCould not start a TensorFlow backend.\n\n" +
      "Install one:\n" +
      "  npm run embeddings:setup                 (portable, works anywhere)\n" +
      "  npm install --no-save @tensorflow/tfjs-node   (native, several times faster)\n\n" +
      `Attempts:\n${failures.join("\n")}\n`,
  );
  process.exit(1);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help || !args.input) {
    console.log(
      "Usage: node scripts/build-embeddings.mjs --input <dataset-dir> [--out public/celebrities.json]\n" +
        "       [--thumbs public/celebs] [--max-per-person 40] [--min-faces 2] [--backend wasm]",
    );
    process.exit(args.help ? 0 : 1);
  }

  const inputDir = path.resolve(args.input);
  const stat = await fs.stat(inputDir).catch(() => null);
  if (!stat?.isDirectory()) {
    console.error(`Dataset directory not found: ${inputDir}`);
    process.exit(1);
  }

  const { tf, faceapi, sharp, backend } = await loadDeps(args.backend);
  console.log(`TensorFlow backend: ${backend}`);

  const modelDir = path.resolve("public/models");
  console.log(`Loading models from ${modelDir} ...`);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelDir);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelDir);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelDir);

  const entries = await fs.readdir(inputDir, { withFileTypes: true });
  const personDirs = entries.filter((e) => e.isDirectory()).sort((a, b) => a.name.localeCompare(b.name));

  if (personDirs.length === 0) {
    console.error(
      `No sub-directories in ${inputDir}.\n` +
        "Expected one folder per celebrity, each containing that person's images.",
    );
    process.exit(1);
  }

  console.log(`Found ${personDirs.length} people.\n`);

  if (args.thumbs) await fs.mkdir(path.resolve(args.thumbs), { recursive: true });

  const celebrities = [];
  let totalImages = 0;
  let totalFaces = 0;

  for (let p = 0; p < personDirs.length; p++) {
    const person = personDirs[p];
    const personPath = path.join(inputDir, person.name);
    const files = (await fs.readdir(personPath))
      .filter((f) => IMAGE_EXT.has(path.extname(f).toLowerCase()))
      .sort()
      .slice(0, args.maxPerPerson);

    const descriptors = [];
    let bestFile = null;
    let bestScore = -1;

    for (const file of files) {
      const filePath = path.join(personPath, file);
      totalImages++;
      let tensor = null;
      try {
        const { data, info } = await sharp(filePath)
          .rotate()
          .removeAlpha()
          .resize({ width: 640, height: 640, fit: "inside", withoutEnlargement: true })
          .raw()
          .toBuffer({ resolveWithObject: true });

        tensor = tf.tensor3d(new Uint8Array(data), [info.height, info.width, 3]);

        const detection = await faceapi
          .detectSingleFace(tensor, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.4 }))
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detection) {
          descriptors.push(Array.from(detection.descriptor));
          totalFaces++;
          if (detection.detection.score > bestScore) {
            bestScore = detection.detection.score;
            bestFile = filePath;
          }
        }
      } catch (err) {
        process.stderr.write(`  ! ${path.relative(inputDir, filePath)}: ${err.message}\n`);
      } finally {
        tensor?.dispose();
      }
    }

    const label = prettifyName(person.name);
    const progress = `[${String(p + 1).padStart(3)}/${personDirs.length}]`;

    if (descriptors.length < args.minFaces) {
      console.log(`${progress} ${label} — skipped (${descriptors.length} usable faces)`);
      continue;
    }

    const slug = slugify(person.name);
    celebrities.push({
      name: label,
      slug,
      descriptor: meanDescriptor(descriptors),
      samples: descriptors.length,
    });

    if (args.thumbs && bestFile) {
      await sharp(bestFile)
        .rotate()
        .resize(256, 256, { fit: "cover", position: "attention" })
        .jpeg({ quality: 82 })
        .toFile(path.join(path.resolve(args.thumbs), `${slug}.jpg`));
    }

    console.log(`${progress} ${label} — ${descriptors.length}/${files.length} faces`);
  }

  if (celebrities.length === 0) {
    console.error("\nNo celebrities had enough usable faces. Nothing written.");
    process.exit(1);
  }

  const dataset = {
    version: 1,
    dimensions: celebrities[0].descriptor.length,
    generatedAt: new Date().toISOString(),
    celebrities,
  };

  const outPath = path.resolve(args.out);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(dataset), "utf8");

  const sizeKb = ((await fs.stat(outPath)).size / 1024).toFixed(0);
  console.log(
    `\nDone.\n` +
      `  People written : ${celebrities.length}\n` +
      `  Images scanned : ${totalImages}\n` +
      `  Faces found    : ${totalFaces}\n` +
      `  Output         : ${path.relative(process.cwd(), outPath)} (${sizeKb} KB)\n` +
      (args.thumbs ? `  Thumbnails     : ${args.thumbs}\n` : ""),
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
