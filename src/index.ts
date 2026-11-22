// @ghuri/vectors — embeddings, vector stores (Cloudflare Vectorize, pgvector, the database, memory) and retrieval
// for Ghuri apps and agents. Workers-safe: nothing Node-only is imported from this entry.
export { VectorsConfig } from './apps'
export {
  enqueueVectorJob,
  runVectorJob,
  tasksInterop,
  VECTOR_TASK_MAX_ATTEMPTS,
  VECTOR_TASK_NAME,
  type VectorJob,
  type VectorTaskResult,
  vectorTask,
} from './background'
export { checkVectors } from './checks'
export {
  type ChunkOptions,
  chunkText,
  DEFAULT_CHUNK_OVERLAP,
  DEFAULT_CHUNK_SIZE,
  DEFAULT_SEPARATORS,
} from './chunking'
export {
  ClearIndexCommand,
  CreateVectorTableCommand,
  RebuildIndexCommand,
  UpdateIndexCommand,
} from './commands'
export {
  type AISDKEmbeddingModel,
  type Embedder,
  type EmbeddingProvider,
  type EmbeddingProviderContext,
  type EmbeddingSpec,
  getEmbeddingProviders,
  KNOWN_DIMENSIONS,
  knownDimensions,
  lazyEmbedder,
  openAICompatibleEmbedder,
  type ResolveEmbeddingOptions,
  registerEmbeddingProvider,
  resolveEmbedding,
  unregisterEmbeddingProvider,
} from './embeddings'
export {
  DimensionMismatch,
  EmbeddingError,
  FilterError,
  InvalidVectorStore,
  TenantRequired,
  VectorsError,
} from './errors'
export {
  type Condition,
  type Filter,
  type FilterOp,
  type Metadata,
  type MetadataValue,
  matches,
  parseFilter,
  toVectorizeFilter,
} from './filters'
export { decodeVector, encodeVector, type Metric, score } from './math'
export { VectorDocument, VectorEmbedding } from './models'
export { DEFAULT_VECTOR_STORE_ALIAS, type VectorStoreConfig, vectorStores } from './registry'
export { resolveNamespace, TENANT_CARRIER, type TenantScope, vectorId } from './scope'
export {
  BaseVectorStore,
  type StoreOptions,
  type StoreQueryOptions,
  type VectorMatch,
  type VectorRecord,
  type VectorStore,
} from './stores/base'
export { type DatabaseStoreOptions, DatabaseVectorStore, databaseStore } from './stores/database'
export { MemoryVectorStore, memoryStore } from './stores/memory'
export { filterSql, PGVECTOR_OPS, type PgvectorOptions, PgvectorStore, pgvector } from './stores/pgvector'
export { type VectorizeIndexLike, type VectorizeOptions, VectorizeStore, vectorize } from './stores/vectorize'
export {
  type Document,
  getVectorIndex,
  type IndexOptions,
  type IndexResult,
  type Key,
  ModelIndex,
  type ModelIndexOptions,
  type ModelSearchHit,
  manifestAvailable,
  type SearchHit,
  type SearchOptions,
  VectorIndex,
  type VectorIndexOptions,
  vectorIndexes,
} from './vector-index'

import type { Embedder } from './embeddings'
import { DEFAULT_VECTOR_STORE_ALIAS, vectorStores } from './registry'

/** Embed one text with a store's embedding model (default store `'default'`). */
export async function embed(
  text: string,
  opts: { store?: string; kind?: 'query' | 'document' } = {},
): Promise<number[]> {
  const [v] = await embedMany([text], opts)
  return v
}

/** Embed texts with a store's embedding model (default store `'default'`). */
export async function embedMany(
  texts: string[],
  opts: { store?: string; kind?: 'query' | 'document' } = {},
): Promise<number[][]> {
  const embedder: Embedder = vectorStores.get(opts.store ?? DEFAULT_VECTOR_STORE_ALIAS).embedding
  return embedder.embed(texts, { kind: opts.kind ?? 'document' })
}
