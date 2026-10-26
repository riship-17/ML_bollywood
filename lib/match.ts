/**
 * Pure similarity / ranking helpers.
 *
 * A note on the metric, because this is the one place the project knowingly
 * departs from the Python original it is modelled on.
 *
 * That version ranks VGGFace embeddings by cosine similarity. These are not
 * VGGFace embeddings — they are 128-d descriptors from face-api's ResNet
 * recognition net (a port of dlib's), and they behave differently: they carry
 * a large common mean component, so cosine similarity between two completely
 * different faces still lands around 0.85-0.92.
 *
 * Measured over six unrelated faces:
 *
 *   cosine     0.835 - 0.916   (spread 0.08)
 *   euclidean  0.598 - 0.854   (spread 0.26)
 *
 * Ranking on cosine would report every celebrity as a ~90% match and the
 * ordering would rest on the third decimal place. So ranking is done on
 * euclidean distance, which is also the metric these descriptors were trained
 * for and the one dlib's well-known 0.6 same-person threshold refers to.
 * Cosine is still computed and shown, as a secondary readout.
 */

export interface Celebrity {
  /** Display name, e.g. "Shah Rukh Khan" */
  name: string;
  /** Folder-safe identifier, used to look up an optional thumbnail */
  slug: string;
  /** Mean 128-d face descriptor across all usable images of this person */
  descriptor: number[];
  /** How many images contributed to the mean — a rough confidence signal */
  samples: number;
}

export interface CelebrityDataset {
  /** Schema version so the app can fail loudly on a stale file */
  version: number;
  /** Descriptor length, always 128 for face-api's recognition net */
  dimensions: number;
  generatedAt: string;
  celebrities: Celebrity[];
}

export interface Match {
  celebrity: Celebrity;
  /** Cosine similarity, reported but not ranked on — see the note above */
  similarity: number;
  /** Euclidean distance; this is what the ranking uses */
  distance: number;
  /** distance mapped to a friendly 0-100 figure */
  percent: number;
  /** "near-identical" | "strong" | "loose" | "weak" */
  closeness: string;
}

export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) {
    throw new Error(`Descriptor length mismatch: ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function euclideanDistance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) {
    throw new Error(`Descriptor length mismatch: ${a.length} vs ${b.length}`);
  }
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/** Mean of a set of descriptors. Returns null for an empty set. */
export function meanDescriptor(descriptors: ArrayLike<number>[]): number[] | null {
  if (descriptors.length === 0) return null;
  const dims = descriptors[0].length;
  const out = new Array<number>(dims).fill(0);
  for (const d of descriptors) {
    if (d.length !== dims) {
      throw new Error(`Descriptor length mismatch: ${d.length} vs ${dims}`);
    }
    for (let i = 0; i < dims; i++) out[i] += d[i];
  }
  for (let i = 0; i < dims; i++) out[i] /= descriptors.length;
  return out;
}

/** Below this, two descriptors are effectively the same face. */
export const DISTANCE_IDENTICAL = 0.35;
/** dlib / face-api's conventional "same person" boundary. */
export const DISTANCE_SAME_PERSON = 0.6;
/** Beyond this, the faces have nothing meaningful in common. */
export const DISTANCE_UNRELATED = 1.0;

/**
 * Turns a euclidean distance into a friendly 0-100 figure, anchored on the
 * thresholds above rather than on an arbitrary rescaling. 0.35 reads as 100,
 * the 0.6 same-person line lands around 62, and 1.0 bottoms out at 0.
 */
export function distanceToPercent(distance: number): number {
  const span = DISTANCE_UNRELATED - DISTANCE_IDENTICAL;
  const scaled = 1 - (distance - DISTANCE_IDENTICAL) / span;
  return Math.round(Math.min(1, Math.max(0, scaled)) * 100);
}

/** A plain-language read on how close the match actually is. */
export function describeCloseness(distance: number): string {
  if (distance <= DISTANCE_IDENTICAL) return "near-identical";
  if (distance <= DISTANCE_SAME_PERSON) return "strong";
  if (distance <= 0.75) return "loose";
  return "weak";
}

export function rankMatches(
  descriptor: ArrayLike<number>,
  celebrities: Celebrity[],
  topN = 5,
): Match[] {
  return celebrities
    .map((celebrity) => {
      const distance = euclideanDistance(descriptor, celebrity.descriptor);
      return {
        celebrity,
        similarity: cosineSimilarity(descriptor, celebrity.descriptor),
        distance,
        percent: distanceToPercent(distance),
        closeness: describeCloseness(distance),
      };
    })
    // Nearest first. Distance, not cosine — see the note at the top.
    .sort((a, b) => a.distance - b.distance)
    .slice(0, topN);
}

export function slugify(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

/** "shah_rukh_khan" / "Shah Rukh Khan" -> "Shah Rukh Khan" */
export function prettifyName(raw: string): string {
  return raw
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((w) => (w.length === 0 ? w : w[0].toUpperCase() + w.slice(1).toLowerCase()))
    .join(" ");
}
