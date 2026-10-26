"use client";

/**
 * Browser-side wrapper around face-api.
 *
 * The library is pulled in with a dynamic import rather than a top-level one
 * for two reasons: it touches `window`/WebGL as it initialises, so it must not
 * run during server rendering, and it is a 1.3 MB bundle that nobody needs
 * until they actually pick a photo.
 *
 * We import the pre-bundled ESM build explicitly — the package's `main` field
 * points at the Node build, which expects @tensorflow/tfjs-node to exist.
 */

type FaceApi = typeof import("@vladmandic/face-api/dist/face-api.esm.js");

/**
 * Where the model weights and WebAssembly binaries are fetched from.
 *
 * By default both are served by this app out of /public, so there is no
 * third-party runtime dependency. Setting NEXT_PUBLIC_ASSET_CDN to a package
 * CDN root (for example https://cdn.jsdelivr.net/npm) instead pulls them from
 * there, which keeps the deployed bundle ~14 MB smaller at the cost of relying
 * on someone else's uptime.
 */
const ASSET_CDN = process.env.NEXT_PUBLIC_ASSET_CDN?.replace(/\/+$/, "");

const MODEL_URL = ASSET_CDN
  ? `${ASSET_CDN}/@vladmandic/face-api@1.7.15/model`
  : "/models";

const WASM_URL = ASSET_CDN
  ? `${ASSET_CDN}/@tensorflow/tfjs-backend-wasm@4.22.0/dist/`
  : "/tfjs/";

let faceapi: FaceApi | null = null;
let loadPromise: Promise<FaceApi> | null = null;
let activeBackend = "unknown";

/** Runtime shape of the tfjs namespace face-api re-exports under a trimmed type. */
interface TfRuntime {
  ready(): Promise<void>;
  setBackend(name: string): Promise<boolean>;
  getBackend(): string;
  setWasmPaths?(prefix: string): void;
}

/**
 * Pick a compute backend explicitly.
 *
 * WebGL is far quicker, but the bundle also carries the WebAssembly backend,
 * and tfjs will reach for it if WebGL is unavailable. Its .wasm binaries have
 * to be pointed at explicitly — otherwise the runtime asks the bundler for a
 * file that was never emitted and the request 404s.
 */
async function startBackend(api: FaceApi): Promise<void> {
  const tf = api.tf as unknown as TfRuntime;

  tf.setWasmPaths?.(WASM_URL);

  for (const backend of ["webgl", "wasm", "cpu"]) {
    try {
      if (await tf.setBackend(backend)) {
        await tf.ready();
        if (tf.getBackend() === backend) {
          activeBackend = backend;
          return;
        }
      }
    } catch {
      // Try the next one.
    }
  }

  await tf.ready();
  activeBackend = tf.getBackend() ?? "unknown";
}

/** Which tfjs backend ended up running, for display. */
export function getBackend(): string {
  return activeBackend;
}

/** Loads detector + landmarks + recognition nets once, shared across callers. */
export function loadModels(onProgress?: (label: string) => void): Promise<FaceApi> {
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    onProgress?.("Starting up the ML backend");
    const api = (await import(
      "@vladmandic/face-api/dist/face-api.esm.js"
    )) as unknown as FaceApi;
    await startBackend(api);

    onProgress?.("Loading face detector");
    await api.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);

    onProgress?.("Loading landmark model");
    await api.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

    onProgress?.("Loading recognition model");
    await api.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

    onProgress?.("Ready");
    faceapi = api;
    return api;
  })().catch((err) => {
    // Let the next attempt retry from scratch rather than caching the failure.
    loadPromise = null;
    throw err;
  });

  return loadPromise;
}

export interface FaceResult {
  descriptor: Float32Array;
  box: { x: number; y: number; width: number; height: number };
  score: number;
}

export class NoFaceError extends Error {
  constructor() {
    super("No face detected");
    this.name = "NoFaceError";
  }
}

function requireApi(): FaceApi {
  if (!faceapi) throw new Error("Models are not loaded yet");
  return faceapi;
}

export type ImageInput = HTMLImageElement | HTMLCanvasElement | HTMLVideoElement;

/**
 * Detects the most prominent face in an image and returns its 128-d descriptor.
 * Throws NoFaceError when the image has no detectable face.
 */
export async function describeFace(input: ImageInput): Promise<FaceResult> {
  const api = requireApi();
  const detection = await api
    .detectSingleFace(input, new api.SsdMobilenetv1Options({ minConfidence: 0.3 }))
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!detection) throw new NoFaceError();

  const { x, y, width, height } = detection.detection.box;
  return {
    descriptor: detection.descriptor,
    box: { x, y, width, height },
    score: detection.detection.score,
  };
}

/** Counts every face in the image — used to warn about group photos. */
export async function countFaces(input: ImageInput): Promise<number> {
  const api = requireApi();
  const detections = await api.detectAllFaces(
    input,
    new api.SsdMobilenetv1Options({ minConfidence: 0.3 }),
  );
  return detections.length;
}

/** Loads a File/Blob into an HTMLImageElement, downscaling very large photos. */
export function fileToImage(file: File | Blob, maxDimension = 1024): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();

    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;

      if (Math.max(w, h) <= maxDimension) {
        // Keep the object URL alive — the element still points at it.
        resolve(img);
        return;
      }

      const scale = maxDimension / Math.max(w, h);
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(w * scale);
      canvas.height = Math.round(h * scale);
      const ctx = canvas.getContext("2d");

      if (!ctx) {
        resolve(img);
        return;
      }

      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const resized = new Image();
      resized.onload = () => {
        URL.revokeObjectURL(url);
        resolve(resized);
      };
      resized.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error("Could not process that image"));
      };
      resized.src = canvas.toDataURL("image/jpeg", 0.92);
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("That file could not be read as an image"));
    };

    img.src = url;
  });
}
