import { DimensionMismatch } from './errors'

/** Distance metrics, with Vectorize's names. */
export type Metric = 'cosine' | 'euclidean' | 'dot-product'

export const METRICS: readonly Metric[] = ['cosine', 'euclidean', 'dot-product']

function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i] * b[i]
  return s
}

function norm(a: ArrayLike<number>): number {
  return Math.sqrt(dot(a, a))
}

/**
 * The score of `b` for the query `a`, as Vectorize reports it: cosine similarity, euclidean distance (lower is
 * closer) or the dot product.
 */
export function score(metric: Metric, a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) {
    throw new DimensionMismatch(`Cannot compare vectors of ${a.length} and ${b.length} dimensions.`)
  }
  if (metric === 'dot-product') return dot(a, b)
  if (metric === 'euclidean') {
    let s = 0
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2
    return Math.sqrt(s)
  }
  const n = norm(a) * norm(b)
  return n === 0 ? 0 : dot(a, b) / n
}

/** Is a higher score better for `metric`? (false for euclidean distance) */
export function higherIsBetter(metric: Metric): boolean {
  return metric !== 'euclidean'
}

/** Sort comparator putting the best score first. */
export function byScore(metric: Metric): (a: { score: number }, b: { score: number }) => number {
  return higherIsBetter(metric) ? (a, b) => b.score - a.score : (a, b) => a.score - b.score
}

/** Encode a vector as base64 little-endian float32 (4 bytes per dimension). */
export function encodeVector(values: ArrayLike<number>): string {
  const f = new Float32Array(values.length)
  for (let i = 0; i < values.length; i++) f[i] = values[i]
  const bytes = new Uint8Array(f.buffer)
  let bin = ''
  for (let i = 0; i < bytes.length; i += 0x8000) {
    bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }
  return btoa(bin)
}

/** Decode `encodeVector()`'s output. */
export function decodeVector(text: string): number[] {
  const bin = atob(text)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return Array.from(new Float32Array(bytes.buffer))
}

/** Throw `DimensionMismatch` unless every vector has `dimensions` values. */
export function checkDimensions(vectors: number[][], dimensions: number | undefined, what: string): void {
  if (!dimensions) return
  for (const v of vectors) {
    if (v.length !== dimensions) {
      throw new DimensionMismatch(`${what} has ${dimensions} dimensions, got a vector of ${v.length}.`)
    }
  }
}
