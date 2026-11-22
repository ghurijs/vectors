import { type CheckMessage, checks } from '@ghuri/core'
import { type Embedder, type EmbeddingSpec, lazyEmbedder } from '../embeddings'
import { InvalidVectorStore } from '../errors'
import type { Condition, Metadata } from '../filters'
import { METRICS, type Metric } from '../math'

/** A vector with its id, namespace and metadata (Vectorize's `VectorizeVector`). */
export interface VectorRecord {
  /** At most 64 bytes. */
  id: string
  values: number[]
  /** Partition (tenant); at most 64 bytes. */
  namespace?: string
  metadata?: Metadata
}

/** A query result (Vectorize's `VectorizeMatch`). */
export interface VectorMatch {
  id: string
  /** Cosine similarity, euclidean distance (lower is closer) or dot product. */
  score: number
  namespace?: string
  metadata?: Metadata
  values?: number[]
}

export interface StoreQueryOptions {
  topK: number
  /** Only this namespace (all namespaces when undefined, as Vectorize). */
  namespace?: string
  /** Normalized metadata conditions (see `parseFilter`). */
  filter?: Condition[]
  returnValues?: boolean
}

/** Options every store factory takes. */
export interface StoreOptions {
  /** `'workers-ai:@cf/baai/bge-base-en-v1.5'`, `'openai:text-embedding-3-small'`, an AI SDK model or an `Embedder`. */
  embedding?: EmbeddingSpec
  /** Vector length (default: known for built-in models, else that of the first embedding). */
  dimensions?: number
  /** Default `'cosine'`. */
  metric?: Metric
  /** Options for the embedding provider (default `AGENTS.PROVIDERS[provider]`). */
  providerOptions?: Record<string, unknown>
}

/** One vector index on one backend. The API is Vectorize's; every backend implements it. */
export interface VectorStore {
  readonly alias: string
  /** `'vectorize'`, `'pgvector'`, `'database'` or `'memory'`. */
  readonly backend: string
  readonly metric: Metric
  readonly dimensions: number | undefined
  /** The store's embedding model; throws `InvalidVectorStore` when none is configured. */
  readonly embedding: Embedder
  readonly hasEmbedding: boolean
  /** Can the store delete a document's vectors without the manifest? */
  readonly nativeDeletes: boolean
  upsert(records: VectorRecord[]): Promise<void>
  query(vector: number[], opts: StoreQueryOptions): Promise<VectorMatch[]>
  getByIds(ids: string[]): Promise<VectorRecord[]>
  deleteByIds(ids: string[]): Promise<void>
  /** Delete every vector of these documents (`metadata._key`) in `namespace`. Stores with `nativeDeletes`. */
  deleteDocuments?(namespace: string, keys: string[]): Promise<void>
  /** Delete every vector of `namespace`. Stores with `nativeDeletes`. */
  clearNamespace?(namespace: string): Promise<void>
  /** Create tables / extensions (`ghuri createvectortable`). */
  setup?(): Promise<void>
  check(): CheckMessage[]
}

/** Base class of the built-in stores. */
export abstract class BaseVectorStore implements VectorStore {
  abstract readonly backend: string
  readonly nativeDeletes: boolean = true
  readonly metric: Metric
  private readonly embedder: Embedder | undefined
  private measured: number | undefined

  constructor(
    readonly alias: string,
    protected readonly options: StoreOptions = {},
  ) {
    this.metric = options.metric ?? 'cosine'
    if (!METRICS.includes(this.metric)) {
      throw new InvalidVectorStore(
        `VECTORS['${alias}']: unknown metric '${this.metric}' (use ${METRICS.join(', ')}).`,
      )
    }
    this.embedder = options.embedding
      ? lazyEmbedder(options.embedding, {
          ...(options.providerOptions ? { providerOptions: options.providerOptions } : {}),
          ...(options.dimensions ? { dimensions: options.dimensions } : {}),
        })
      : undefined
  }

  get dimensions(): number | undefined {
    return this.options.dimensions ?? this.embedder?.dimensions ?? this.measured
  }

  get hasEmbedding(): boolean {
    return !!this.embedder
  }

  get embedding(): Embedder {
    if (!this.embedder) {
      throw new InvalidVectorStore(
        `VECTORS['${this.alias}'] has no embedding model: pass { embedding: 'workers-ai:@cf/baai/bge-base-en-v1.5' } (or another model) to the store.`,
      )
    }
    return this.embedder
  }

  /** @internal Remember the length of the vectors seen, for stores whose dimensions are not configured. */
  protected noteDimensions(records: Array<{ values: number[] }>): void {
    if (!this.dimensions && records.length) this.measured = records[0].values.length
  }

  abstract upsert(records: VectorRecord[]): Promise<void>
  abstract query(vector: number[], opts: StoreQueryOptions): Promise<VectorMatch[]>
  abstract getByIds(ids: string[]): Promise<VectorRecord[]>
  abstract deleteByIds(ids: string[]): Promise<void>

  check(): CheckMessage[] {
    if (this.embedder) return []
    return [
      checks.error(`The vector store '${this.alias}' has no embedding model.`, {
        hint: "Pass { embedding: 'workers-ai:@cf/baai/bge-base-en-v1.5' } (or another model) to the store factory.",
        id: 'vectors.E004',
        obj: `VECTORS['${this.alias}']`,
      }),
    ]
  }
}

/** Split `items` into lists of at most `size`. */
export function batches<T>(items: T[], size: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size))
  return out
}
