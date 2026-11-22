// @ghuri/vectors/testing — test helpers: a deterministic embedder, an in-memory Vectorize binding, store overrides.
import type { Embedder } from './embeddings'
import { VectorsError } from './errors'
import { type Condition, type FilterOp, type MetadataValue, matches } from './filters'
import { byScore, type Metric, score } from './math'
import { DEFAULT_VECTOR_STORE_ALIAS, type VectorStoreConfig, vectorStores } from './registry'
import type { VectorMatch, VectorRecord, VectorStore } from './stores/base'
import type { VectorizeIndexLike } from './stores/vectorize'

export { MemoryVectorStore, memoryStore } from './stores/memory'

/** A fake embedder that also counts what it embedded. */
export interface FakeEmbedder extends Embedder {
  /** Every text embedded, in order. */
  readonly texts: string[]
  /** Number of `embed()` calls. */
  calls: number
  reset(): void
}

function fnv1a(text: string, seed: number): number {
  let h = 0x811c9dc5 ^ seed
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return h >>> 0
}

/**
 * A deterministic hashed bag-of-words embedder: each word adds ±1 to a hashed dimension and vectors are normalized,
 * so texts sharing words are close. No network, no model; for tests.
 */
export function fakeEmbedding(opts: { dimensions?: number; modelId?: string } = {}): FakeEmbedder {
  const dimensions = opts.dimensions ?? 64
  const texts: string[] = []
  const embedder: FakeEmbedder = {
    modelId: opts.modelId ?? `fake:bow-${dimensions}`,
    dimensions,
    maxBatch: 100,
    texts,
    calls: 0,
    reset() {
      texts.length = 0
      embedder.calls = 0
    },
    async embed(input) {
      embedder.calls++
      texts.push(...input)
      return input.map((text) => {
        const v = new Array<number>(dimensions).fill(0)
        for (const word of text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
          const h = fnv1a(word, 0)
          v[h % dimensions] += fnv1a(word, 1) & 1 ? 1 : -1
        }
        const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1
        return v.map((x) => x / n)
      })
    },
  }
  return embedder
}

/**
 * Use `config` for the `VECTORS` alias (default `'default'`) until `restore()`, whatever the settings say:
 *
 * ```ts
 * const { store, restore } = useVectorStore(memoryStore({ embedding: fakeEmbedding() }))
 * ```
 */
export function useVectorStore<S extends VectorStore>(
  config: VectorStoreConfig<S>,
  alias: string = DEFAULT_VECTOR_STORE_ALIAS,
): { store: S; restore(): void } {
  const restore = vectorStores.override(alias, config)
  return { store: config.create(alias), restore }
}

const encoder = new TextEncoder()
const OPS: FilterOp[] = ['$eq', '$ne', '$in', '$nin', '$lt', '$lte', '$gt', '$gte']

function fromVectorizeFilter(filter: Record<string, unknown> | undefined): Condition[] {
  const out: Condition[] = []
  for (const [key, spec] of Object.entries(filter ?? {})) {
    if (spec === null || typeof spec !== 'object' || Array.isArray(spec)) {
      out.push({ key, op: '$eq', value: spec as MetadataValue })
      continue
    }
    for (const [op, value] of Object.entries(spec as Record<string, unknown>)) {
      if (!OPS.includes(op as FilterOp))
        throw new VectorsError(`fakeVectorize: unsupported filter operator '${op}'.`)
      out.push({ key, op: op as FilterOp, value: value as MetadataValue })
    }
  }
  return out
}

/** An in-memory Vectorize binding (`env.VECTORIZE` in tests): namespaces, metadata filters and Vectorize's limits. */
export interface FakeVectorize extends VectorizeIndexLike {
  readonly vectors: Map<string, VectorRecord>
  /** Every `query()` call's options (to assert on namespaces and filters). */
  readonly queries: Array<Record<string, unknown>>
  describe(): Promise<{ dimensions: number | undefined; vectorCount: number; metric: Metric }>
}

export function fakeVectorize(opts: { dimensions?: number; metric?: Metric } = {}): FakeVectorize {
  const metric = opts.metric ?? 'cosine'
  const vectors = new Map<string, VectorRecord>()
  const queries: Array<Record<string, unknown>> = []
  const check = (r: VectorRecord) => {
    if (encoder.encode(r.id).length > 64)
      throw new VectorsError(`VECTOR_INVALID_ID: '${r.id}' is longer than 64 bytes`)
    if (r.namespace && encoder.encode(r.namespace).length > 64) {
      throw new VectorsError(`VECTOR_INVALID_NAMESPACE: '${r.namespace}' is longer than 64 bytes`)
    }
    if (opts.dimensions && r.values.length !== opts.dimensions) {
      throw new VectorsError(`VECTOR_DIMENSION_MISMATCH: expected ${opts.dimensions}, got ${r.values.length}`)
    }
    if (encoder.encode(JSON.stringify(r.metadata ?? {})).length > 10_240) {
      throw new VectorsError(`VECTOR_METADATA_TOO_LARGE: '${r.id}'`)
    }
  }
  return {
    vectors,
    queries,
    async upsert(records) {
      for (const r of records) check(r)
      for (const r of records)
        vectors.set(r.id, { ...r, values: [...r.values], metadata: { ...(r.metadata ?? {}) } })
      return { mutationId: crypto.randomUUID() }
    },
    async query(vector, options) {
      queries.push({ ...options })
      const topK = options.topK ?? 5
      const full = options.returnValues || options.returnMetadata === 'all' || options.returnMetadata === true
      if (topK > (full ? 50 : 100)) throw new VectorsError(`VECTOR_QUERY_ERROR: topK ${topK} is too large`)
      const conditions = fromVectorizeFilter(options.filter)
      const found: VectorMatch[] = []
      for (const r of vectors.values()) {
        if (options.namespace !== undefined && r.namespace !== options.namespace) continue
        if (!matches(conditions, r.metadata)) continue
        found.push({
          id: r.id,
          score: score(metric, vector, r.values),
          ...(r.namespace ? { namespace: r.namespace } : {}),
          ...(options.returnMetadata && options.returnMetadata !== 'none'
            ? { metadata: { ...r.metadata } }
            : {}),
          ...(options.returnValues ? { values: [...r.values] } : {}),
        })
      }
      const sorted = found.sort(byScore(metric)).slice(0, topK)
      return { matches: sorted, count: sorted.length }
    },
    async getByIds(ids) {
      return ids.flatMap((id) => (vectors.has(id) ? [{ ...vectors.get(id)! }] : []))
    },
    async deleteByIds(ids) {
      for (const id of ids) vectors.delete(id)
      return { mutationId: crypto.randomUUID() }
    },
    async describe() {
      return { dimensions: opts.dimensions, vectorCount: vectors.size, metric }
    },
  }
}
