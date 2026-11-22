# @ghuri/vectors

Embeddings, vector stores and retrieval (RAG) for [Ghuri](https://github.com/ghurijs/ghuri) apps and agents. The
same code runs on **Cloudflare Workers + D1 + Vectorize** and on **Node + Postgres + pgvector**: stores are chosen in
the `VECTORS` setting, like caches and task backends.

> Status: alpha (`0.1.0-alpha.0`). APIs may change. Needs `@ghuri/core` 0.1.0-alpha.5 and `@ghuri/orm`
> 0.1.0-alpha.5.

```sh
npm install @ghuri/vectors@next
```

## Contents

1. [Quick start](#1-quick-start)
2. [Stores](#2-stores)
3. [Embedding models](#3-embedding-models)
4. [Indexes and search](#4-indexes-and-search)
5. [Model indexes](#5-model-indexes)
6. [Tenants](#6-tenants)
7. [Background indexing](#7-background-indexing)
8. [Agents](#8-agents)
9. [Commands and checks](#9-commands-and-checks)
10. [Testing](#10-testing)
11. [Differences from Django](#11-differences-from-django)

## 1. Quick start

```ts
// src/settings.ts
import { pgvector, vectorize, VectorsConfig } from '@ghuri/vectors'

export default defineSettings((env) => ({
  INSTALLED_APPS: [/* … */, VectorsConfig],                // then: npx ghuri migrate
  VECTORS: {
    default: env.VECTORIZE
      ? vectorize('VECTORIZE', { embedding: 'workers-ai:@cf/baai/bge-base-en-v1.5' })
      : pgvector({ database: 'default', embedding: 'openai:text-embedding-3-small' }),   // npx ghuri createvectortable
  },
}))
```

```ts
// src/blog/vectors.ts — load it from the app: `override modules = { vectors }`
import { ModelIndex, VectorIndex } from '@ghuri/vectors'
import { Post } from './models'

export const posts = new ModelIndex(Post, {
  text: (p) => `${p.title}\n\n${p.body}`,
  metadata: (p) => ({ status: p.status, year: p.year }),
  indexQueryset: (qs) => qs.filter({ status: 'published' }),
})
export const handbook = new VectorIndex('blog.handbook')

// anywhere
await handbook.index({ key: 'refunds', text: refundPolicy, metadata: { section: 'billing' } })
const hits = await posts.search('how do I cancel?', { k: 5, filter: { year__gte: 2025 } })
hits[0].object // the Post
hits[0].text   // the matching chunk
```

## 2. Stores

| Factory | Runs on | Search |
|---|---|---|
| `vectorize(binding, opts)` | Workers (`"vectorize"` binding in wrangler.jsonc) | Vectorize (ANN), namespaces per tenant |
| `pgvector({ database, table, index, m, efConstruction, lists })` | Postgres + pgvector (postgres.js, Hyperdrive, PGlite) | `hnsw` (default) / `ivfflat` / exact |
| `databaseStore({ database })` | D1, SQLite, Postgres (`vectors_vectorembedding`) | exact, one namespace at a time — small apps |
| `memoryStore()` | the process | exact — tests and scripts |

Common options: `embedding` (required), `dimensions` (known for built-in models), `metric` (`'cosine'`,
`'euclidean'`, `'dot-product'`), `providerOptions`. The store API is Vectorize's (`upsert`, `query`, `getByIds`,
`deleteByIds`); extend `BaseVectorStore` for another backend.

Create Vectorize indexes with the model's dimensions: `wrangler vectorize create mysite-default --dimensions=768
--metric=cosine`, and a metadata index per filtered key (`wrangler vectorize create-metadata-index`).

## 3. Embedding models

`'workers-ai:@cf/baai/bge-base-en-v1.5'` (the AI binding), `'openai:text-embedding-3-small'` (`OPENAI_API_KEY`,
`baseURL` for compatible APIs) and `'ghuri-ai:openai/text-embedding-3-small'` (`GHURI_AI_URL`) need no extra
package. Any AI SDK embedding model works too. Keys come from the store's `providerOptions`, then
`AGENTS.PROVIDERS[provider]`, then the environment. `registerEmbeddingProvider(name, factory)` adds providers;
`embed(text)` / `embedMany(texts)` embed with a store's model.

## 4. Indexes and search

- `new VectorIndex(name, { store, chunk, tenant, storeText, prefixes })`.
- `index(docs, { namespace, force, background })` chunks (recursive character splitter, `{ size: 1000, overlap:
  200 }`), embeds and upserts. With the `vectors` app, unchanged documents are skipped and stale chunks removed.
- `delete(keys)`, `clear()`, `keys()`.
- `search(query, { k, filter, exclude, namespace, minScore, includeValues })` → `[{ id, key, chunk, score, text,
  metadata }]`, best first.
- Filters are Django lookups on metadata: `{ status: 'published', year__gte: 2025, lang__in: ['en'] }`, and
  `exclude: { status: 'draft' }`.

## 5. Model indexes

`new ModelIndex(Model, { name?, text, metadata?, indexQueryset?, autoIndex = true, background = 'auto' })`:
re-indexes rows on `postSave` (after commit, `raw` saves skipped), removes them on `postDelete` or when they leave
`indexQueryset`, and returns hits with `object`. `update()` indexes every row (`ghuri update_index`).

## 6. Tenants

With `@ghuri/tenants`, indexes are scoped to the current tenant (`tenant: 'auto'`): the tenant from core's context
carriers goes into the store namespace, and every operation without one raises `TenantRequired` (also inside
`withoutTenant()`). `tenant: false` makes an index shared by every tenant.

## 7. Background indexing

With `@ghuri/tasks` loaded, `index(docs, { background: true })` and model auto-indexing (`background: 'auto'`) run as
the `vectors.index` task (3 attempts), with the tenant carried. `@ghuri/vectors` does not import `@ghuri/tasks`; it
uses `globalThis[Symbol.for('ghuri.tasks.interop')]`.

## 8. Agents

```ts
import { searchTool } from '@ghuri/vectors/agents'

class SupportAgent extends Agent {
  tools = [searchTool(handbook, { description: 'Search the support handbook.', k: 5 })]
}
```

`filter` may be a function of the tool context for per-user scoping. `formatHits(hits)` / `retrieve(index, query)`
build a `[n] (key) text` context block for prompts.

## 9. Commands and checks

`ghuri update_index`, `clear_index`, `rebuild_index` (`[index…] [--tenant pk]`, django-haystack's names) and
`ghuri createvectortable [--store alias]` (pgvector tables). Checks `vectors.E001`–`E006`, `W001`–`W003`.

## 10. Testing

`@ghuri/vectors/testing`: `fakeEmbedding({ dimensions })` (deterministic, counts calls), `fakeVectorize()` (an
in-memory Vectorize binding), `useVectorStore(memoryStore({ embedding }))`.

This package's tests run the stores on SQLite, PGlite (with `@electric-sql/pglite-pgvector`), D1 (Miniflare) and a
fake Vectorize binding, and run indexing and search inside workerd.

## 11. Differences from Django

Django has no vector store; the API borrows django-haystack's names (`update_index`, `index_queryset`) and
pgvector-python's (`hnsw`, `ivfflat`, `vector_cosine_ops`). There is no `VectorField` yet, and `autoIndex` is on by
default (haystack's default signal processor does nothing).

## License

MIT
