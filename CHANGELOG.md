# Changelog

## Unreleased (0.1.0-alpha.0)

- First release: embeddings, vector stores and retrieval (RAG) for Ghuri.
- Stores behind the `VECTORS` setting: `vectorize()` (Cloudflare Vectorize), `pgvector()` (Postgres with pgvector,
  `hnsw` / `ivfflat` indexes), `databaseStore()` (exact search on D1, SQLite or Postgres) and `memoryStore()`.
  `vectorStores` registry, `VectorStore` / `BaseVectorStore` for your own backends.
- Embeddings: `'workers-ai:'`, `'openai:'` (and compatible APIs) and `'ghuri-ai:'` model ids with no extra package,
  AI SDK embedding models through `doEmbed`, `registerEmbeddingProvider()`, `embed()` / `embedMany()`. Provider
  keys come from the store's `providerOptions` or, without them, `AGENTS.PROVIDERS`, with environment variables
  (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `GHURI_AI_URL`, `GHURI_AI_API_KEY`) filling missing keys.
- `VectorIndex` (`index`, `delete`, `clear`, `keys`, `search`) with chunking (`chunkText`), Django-lookup metadata
  filters (`exact`, `in`, `gt`, `gte`, `lt`, `lte`, `exclude`) and `minScore`.
- `ModelIndex` (django-haystack's `SearchIndex`): text and metadata of model rows, `indexQueryset`, re-indexing on
  `postSave` / `postDelete` (`autoIndex`), hits with their `object`.
- The `vectors` app: the `VectorDocument` manifest (unchanged documents are not embedded again; Vectorize deletes),
  the `VectorEmbedding` table of `databaseStore()`, commands `update_index`, `clear_index`, `rebuild_index` and
  `createvectortable`, system checks `vectors.*`.
- Tenant scoping that fails closed (`TenantRequired`): the carried tenant goes into the store namespace.
- Background indexing as the `vectors.index` task through `@ghuri/tasks`' interop object (no import of
  `@ghuri/tasks`).
- `@ghuri/vectors/agents`: `searchTool()`, `formatHits()`, `retrieve()`.
- `@ghuri/vectors/testing`: `fakeEmbedding()`, `fakeVectorize()`, `useVectorStore()`.
- Docs: the README matches the source (provider-option precedence, optional `embedding`, command options, where
  `TenantRequired` is raised).
