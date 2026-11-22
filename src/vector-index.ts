import { apps, ImproperlyConfigured } from '@ghuri/core'
import { type ModelClass, onCommit, postDelete, postSave, type QuerySet } from '@ghuri/orm'
import { enqueueVectorJob, setJobRunner, useBackground, type VectorJob } from './background'
import { type ChunkOptions, chunkText, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE } from './chunking'
import { FilterError, VectorsError } from './errors'
import { checkMetadata, type Filter, type Metadata, type MetadataValue, parseFilter } from './filters'
import { globalSingleton } from './global'
import { higherIsBetter } from './math'
import { VectorDocument } from './models'
import { DEFAULT_VECTOR_STORE_ALIAS, vectorStores } from './registry'
import { chunkIds, resolveNamespace, sha256Hex, stableStringify, type TenantScope, vectorId } from './scope'
import { batches, type VectorRecord, type VectorStore } from './stores/base'

/** A document key: model primary keys and your own ids. */
export type Key = string | number

/** A document to index. */
export interface Document {
  key: Key
  text: string
  /** Strings, numbers or booleans; `null` / `undefined` values are left out. Keys must not start with `_`. */
  metadata?: Record<string, MetadataValue | null | undefined>
}

export interface VectorIndexOptions {
  /** `VECTORS` alias (default `'default'`). */
  store?: string
  /** Chunking (default `{ size: 1000, overlap: 200 }`), `false` for one chunk per document, or your own splitter. */
  chunk?: ChunkOptions | false | ((text: string) => string[])
  /** Scope to the current tenant: `'auto'` (default: when `@ghuri/tenants` is loaded), `true` or `false`. */
  tenant?: TenantScope
  /** Keep chunk text in metadata (`_text`) so search returns it (default true). */
  storeText?: boolean
  /** Text put before queries and documents, for models that want it (`'query: '` for e5). */
  prefixes?: { query?: string; document?: string }
}

export interface IndexOptions {
  /** A partition inside the index (and inside the tenant). */
  namespace?: string
  /** Embed even when the document is unchanged. */
  force?: boolean
  /** Run as the `vectors.index` task (`'auto'`: when `@ghuri/tasks` is loaded). Default false. */
  background?: boolean | 'auto'
}

/** What `index()` did. */
export interface IndexResult {
  /** Documents embedded and written. */
  indexed: number
  /** Unchanged documents (same digest in the manifest). */
  skipped: number
  /** Documents removed (no text, or no longer in a model index's queryset). */
  deleted: number
  /** Chunks written. */
  chunks: number
  /** Set when the work was enqueued instead (`background`). */
  taskResultId?: string
}

export interface SearchOptions {
  /** Number of chunks (default 4). */
  k?: number
  /** Django lookups on metadata: `{ status: 'published', year__gte: 2025 }`. */
  filter?: Filter
  /** Leave out matches: `{ status: 'draft' }`, `{ lang__in: ['de'] }` (one key each). */
  exclude?: Filter
  namespace?: string
  /** Drop matches below this score (cosine and dot-product only). */
  minScore?: number
  /** Return the stored vectors. */
  includeValues?: boolean
}

/** A matching chunk. */
export interface SearchHit {
  /** The vector id. */
  id: string
  /** The document key. */
  key: string
  chunk: number
  /** Cosine similarity, euclidean distance (lower is closer) or dot product, as the store reports it. */
  score: number
  /** The chunk text (null with `storeText: false`). */
  text: string | null
  /** The document's metadata (without reserved `_` keys). */
  metadata: Metadata
  values?: number[]
}

const MAX_KEY_LENGTH = 255

const registry: Map<string, VectorIndex> = globalSingleton('ghuri.vectors.indexes', () => new Map())

/** The index registered as `name`. */
export function getVectorIndex(name: string): VectorIndex | undefined {
  return registry.get(name)
}

/** Every index created in this process. */
export function vectorIndexes(): VectorIndex[] {
  return [...registry.values()]
}

/** Is the `vectors` app (the manifest) installed? */
export function manifestAvailable(): boolean {
  return apps.ready && apps.getAppConfigs().some((a) => a.label === 'vectors')
}

function toKey(key: Key): string {
  const k = String(key)
  if (!k || k.length > MAX_KEY_LENGTH) {
    throw new VectorsError(`Document keys must be 1-${MAX_KEY_LENGTH} characters, got '${k.slice(0, 40)}'.`)
  }
  return k
}

function list<T>(value: T | T[]): T[] {
  return Array.isArray(value) ? value : [value]
}

interface Prepared {
  key: string
  chunks: string[]
  metadata: Metadata
  digest: string
}

/**
 * A named vector index of documents on a store: chunking, embedding, metadata filters, tenant namespaces and a
 * manifest of what is indexed.
 *
 * ```ts
 * export const handbook = new VectorIndex('support.handbook', { chunk: { size: 800 } })
 * await handbook.index({ key: 'refunds', text, metadata: { section: 'billing' } })
 * const hits = await handbook.search('how do refunds work?', { k: 5, filter: { section: 'billing' } })
 * ```
 */
export class VectorIndex {
  readonly name: string
  readonly options: VectorIndexOptions

  constructor(name: string, opts: VectorIndexOptions = {}) {
    if (!/^[A-Za-z0-9_.:-]{1,48}$/.test(name)) {
      throw new ImproperlyConfigured(
        `Invalid vector index name '${name}': use 1-48 letters, digits, '_', '.', ':' or '-'.`,
      )
    }
    this.name = name
    this.options = opts
    registry.set(name, this)
  }

  /** The `VECTORS` alias of the store. */
  get storeAlias(): string {
    return this.options.store ?? DEFAULT_VECTOR_STORE_ALIAS
  }

  /** The store of the current context. */
  get store(): VectorStore {
    return vectorStores.get(this.storeAlias)
  }

  get tenantScope(): TenantScope {
    return this.options.tenant ?? 'auto'
  }

  /** The store namespace for the current tenant (and `namespace`). Throws `TenantRequired` when scoped without one. */
  namespace(namespace?: string | null, what?: string): Promise<string> {
    return resolveNamespace(this.name, this.tenantScope, namespace, what)
  }

  /** Split a document's text into chunks. */
  chunks(text: string): string[] {
    const chunk = this.options.chunk
    if (typeof chunk === 'function') return chunk(text).filter((c) => c.trim())
    if (chunk === false) return text.trim() ? [text.trim()] : []
    return chunkText(text, chunk ?? {})
  }

  private chunkDescriptor(): unknown {
    const chunk = this.options.chunk
    if (typeof chunk === 'function') return `fn:${chunk.name || 'anonymous'}`
    if (chunk === false) return false
    return {
      size: chunk?.size ?? DEFAULT_CHUNK_SIZE,
      overlap: chunk?.overlap ?? DEFAULT_CHUNK_OVERLAP,
      separators: chunk?.separators ?? null,
    }
  }

  private async prepare(docs: Document[], store: VectorStore): Promise<Prepared[]> {
    const byKey = new Map<string, Prepared>()
    for (const doc of docs) {
      if (!doc || typeof doc.text !== 'string')
        throw new VectorsError('Documents need a `key` and a `text` string.')
      const key = toKey(doc.key)
      const metadata = checkMetadata(doc.metadata, `Document '${key}' of '${this.name}'`)
      const chunks = this.chunks(doc.text)
      const digest = await sha256Hex(
        stableStringify({
          model: store.embedding.modelId,
          dimensions: store.dimensions ?? null,
          chunk: this.chunkDescriptor(),
          prefix: this.options.prefixes?.document ?? '',
          storeText: this.options.storeText !== false,
          text: doc.text,
          metadata,
        }),
      )
      byKey.set(key, { key, chunks, metadata, digest })
    }
    return [...byKey.values()]
  }

  /**
   * Chunk, embed and store documents. With the `vectors` app, unchanged documents are skipped and chunks that a
   * shorter version no longer has are deleted. A document whose text has no chunks is deleted.
   */
  async index(docs: Document | Document[], opts: IndexOptions = {}): Promise<IndexResult> {
    const all = list(docs)
    if (useBackground(opts.background)) {
      const documents = all.map((d) => ({
        key: toKey(d.key),
        text: d.text,
        metadata: checkMetadata(d.metadata, `Document '${d.key}' of '${this.name}'`),
      }))
      const result = await enqueueVectorJob({
        index: this.name,
        op: 'index',
        documents,
        ...(opts.namespace ? { namespace: opts.namespace } : {}),
        ...(opts.force ? { force: true } : {}),
      })
      return { indexed: 0, skipped: 0, deleted: 0, chunks: 0, taskResultId: result.id }
    }
    return this.indexDocuments(all, opts)
  }

  /** @internal `index()` without the background option. */
  async indexDocuments(
    docs: Document[],
    opts: { namespace?: string; force?: boolean } = {},
  ): Promise<IndexResult> {
    const result: IndexResult = { indexed: 0, skipped: 0, deleted: 0, chunks: 0 }
    if (!docs.length) return result
    const store = this.store
    const namespace = await this.namespace(opts.namespace, 'Indexing')
    const manifest = manifestAvailable()
    if (!store.nativeDeletes && !manifest) {
      throw new ImproperlyConfigured(
        `The ${store.backend} store '${store.alias}' needs the 'vectors' app to index documents: add VectorsConfig to INSTALLED_APPS.`,
      )
    }
    const prepared = await this.prepare(docs, store)
    const existing = manifest
      ? await this.manifestRows(
          namespace,
          prepared.map((p) => p.key),
        )
      : new Map<string, VectorDocument>()

    const empty = prepared.filter((p) => !p.chunks.length)
    const changed = prepared.filter(
      (p) => p.chunks.length && (opts.force || existing.get(p.key)?.digest !== p.digest),
    )
    result.skipped = prepared.length - empty.length - changed.length

    // 1. Embed everything first: a failed call changes nothing.
    const prefix = this.options.prefixes?.document ?? ''
    const texts = changed.flatMap((p) => p.chunks.map((c) => prefix + c))
    const vectors = texts.length ? await store.embedding.embed(texts, { kind: 'document' }) : []
    if (vectors.length !== texts.length) {
      throw new VectorsError(
        `The embedding model returned ${vectors.length} vectors for ${texts.length} chunks.`,
      )
    }

    // 2. Write the vectors.
    const storeText = this.options.storeText !== false
    const records: VectorRecord[] = []
    let n = 0
    for (const p of changed) {
      for (let i = 0; i < p.chunks.length; i++) {
        records.push({
          id: await vectorId(namespace, p.key, i),
          values: vectors[n++],
          namespace,
          metadata: { ...p.metadata, _key: p.key, _chunk: i, ...(storeText ? { _text: p.chunks[i] } : {}) },
        })
      }
    }
    if (!manifest && store.deleteDocuments && changed.length) {
      // Without a manifest the old chunk count is unknown: remove the documents' vectors, then write the new ones.
      await store.deleteDocuments(
        namespace,
        changed.map((p) => p.key),
      )
    }
    for (const batch of batches(records, 500)) await store.upsert(batch)

    // 3. Remove chunks the new versions no longer have.
    if (manifest) {
      const stale: string[] = []
      for (const p of changed) {
        const old = existing.get(p.key)?.chunks ?? 0
        stale.push(...(await chunkIds(namespace, p.key, p.chunks.length, old)))
      }
      if (stale.length) await store.deleteByIds(stale)
    }

    // 4. Record what is indexed.
    if (manifest) {
      for (const p of changed) {
        const row = existing.get(p.key)
        if (row) {
          row.chunks = p.chunks.length
          row.digest = p.digest
          await row.save()
        } else {
          await VectorDocument.objects.create({
            index: this.name,
            namespace,
            key: p.key,
            chunks: p.chunks.length,
            digest: p.digest,
          })
        }
      }
    }
    result.indexed = changed.length
    result.chunks = records.length
    if (empty.length) {
      await this.deleteKeys(
        empty.map((p) => p.key),
        namespace,
      )
      result.deleted = empty.length
    }
    return result
  }

  private async manifestRows(namespace: string, keys: string[]): Promise<Map<string, VectorDocument>> {
    const out = new Map<string, VectorDocument>()
    for (const batch of batches(keys, 90)) {
      const rows = await VectorDocument.objects.filter({ index: this.name, namespace, key__in: batch })
      for (const row of rows) out.set(row.key, row)
    }
    return out
  }

  /** Delete documents' vectors (and manifest rows). */
  async delete(
    keys: Key | Key[],
    opts: { namespace?: string; background?: boolean | 'auto' } = {},
  ): Promise<void> {
    const all = list(keys).map(toKey)
    if (!all.length) return
    if (useBackground(opts.background)) {
      await enqueueVectorJob({
        index: this.name,
        op: 'delete',
        keys: all,
        ...(opts.namespace ? { namespace: opts.namespace } : {}),
      })
      return
    }
    await this.deleteKeys(all, await this.namespace(opts.namespace, 'Deleting'))
  }

  private async deleteKeys(keys: string[], namespace: string): Promise<void> {
    const store = this.store
    const manifest = manifestAvailable()
    if (!store.nativeDeletes && !manifest) {
      throw new ImproperlyConfigured(
        `The ${store.backend} store '${store.alias}' needs the 'vectors' app to delete documents: add VectorsConfig to INSTALLED_APPS.`,
      )
    }
    if (manifest) {
      const rows = await this.manifestRows(namespace, keys)
      if (!store.deleteDocuments) {
        const ids: string[] = []
        for (const row of rows.values()) ids.push(...(await chunkIds(namespace, row.key, 0, row.chunks)))
        if (ids.length) await store.deleteByIds(ids)
      }
      for (const batch of batches(keys, 90)) {
        await VectorDocument.objects.filter({ index: this.name, namespace, key__in: batch }).delete()
      }
    }
    if (store.deleteDocuments) await store.deleteDocuments(namespace, keys)
  }

  /** Delete every document of the current namespace (the current tenant's, when scoped). */
  async clear(opts: { namespace?: string } = {}): Promise<void> {
    const namespace = await this.namespace(opts.namespace, 'Clearing')
    const store = this.store
    const manifest = manifestAvailable()
    if (store.clearNamespace) await store.clearNamespace(namespace)
    else if (manifest) {
      const rows = await VectorDocument.objects.filter({ index: this.name, namespace })
      const ids: string[] = []
      for (const row of rows) ids.push(...(await chunkIds(namespace, row.key, 0, row.chunks)))
      if (ids.length) await store.deleteByIds(ids)
    } else {
      throw new ImproperlyConfigured(
        `The ${store.backend} store '${store.alias}' needs the 'vectors' app to clear an index: add VectorsConfig to INSTALLED_APPS.`,
      )
    }
    if (manifest) await VectorDocument.objects.filter({ index: this.name, namespace }).delete()
  }

  /** Keys of the indexed documents in the current namespace (from the manifest). */
  async keys(opts: { namespace?: string } = {}): Promise<string[]> {
    if (!manifestAvailable()) {
      throw new ImproperlyConfigured("Listing indexed documents needs the 'vectors' app (the manifest).")
    }
    const namespace = await this.namespace(opts.namespace, 'Listing')
    return (await VectorDocument.objects
      .filter({ index: this.name, namespace })
      .orderBy('key')
      .valuesList('key', { flat: true })) as string[]
  }

  /** Embed `query` (or use a vector) and return the closest chunks, best first. */
  async search(query: string | number[], opts: SearchOptions = {}): Promise<SearchHit[]> {
    const store = this.store
    const namespace = await this.namespace(opts.namespace, 'Searching')
    const conditions = parseFilter(opts.filter, opts.exclude)
    if (opts.minScore !== undefined && !higherIsBetter(store.metric)) {
      throw new FilterError(
        `minScore needs a cosine or dot-product store; '${store.alias}' uses ${store.metric}.`,
      )
    }
    const k = Math.max(1, Math.trunc(opts.k ?? 4))
    let vector: number[]
    if (Array.isArray(query)) vector = query
    else {
      if (!query.trim()) return []
      ;[vector] = await store.embedding.embed([(this.options.prefixes?.query ?? '') + query], {
        kind: 'query',
      })
    }
    const matches = await store.query(vector, {
      topK: k,
      namespace,
      filter: conditions,
      returnValues: !!opts.includeValues,
    })
    const hits: SearchHit[] = []
    for (const m of matches) {
      // Belt and braces: never return another namespace's vectors.
      if (m.namespace !== undefined && m.namespace !== namespace) continue
      if (opts.minScore !== undefined && m.score < opts.minScore) continue
      const md = m.metadata ?? {}
      const metadata: Metadata = {}
      for (const [key, value] of Object.entries(md)) if (!key.startsWith('_')) metadata[key] = value
      hits.push({
        id: m.id,
        key: String(md._key ?? m.id),
        chunk: typeof md._chunk === 'number' ? md._chunk : Number(md._chunk ?? 0),
        score: m.score,
        text: typeof md._text === 'string' ? md._text : null,
        metadata,
        ...(m.values ? { values: m.values } : {}),
      })
    }
    return hits
  }

  /** @internal Run a background job (`vectors.index` task). */
  async runJob(job: VectorJob): Promise<IndexResult | null> {
    if (job.op === 'delete') {
      await this.delete(job.keys ?? [], { ...(job.namespace ? { namespace: job.namespace } : {}) })
      return null
    }
    return this.indexDocuments(job.documents ?? [], {
      ...(job.namespace ? { namespace: job.namespace } : {}),
      ...(job.force ? { force: true } : {}),
    })
  }
}

// ---- model indexes -----------------------------------------------------------------------------------------------

type Instance<M> = M extends abstract new (...args: any) => infer I ? I : any

export interface ModelIndexOptions<M extends ModelClass> extends VectorIndexOptions {
  /** Index name (default `'<app label>.<model name>'`, e.g. `'blog.post'`). */
  name?: string
  /** The text to embed: a function, or field names joined with blank lines. */
  text: ((obj: Instance<M>) => string) | string[]
  /** Metadata to store and filter on. */
  metadata?: (obj: Instance<M>) => Record<string, MetadataValue | null | undefined>
  /** Rows to index (haystack's `index_queryset`); rows outside it are removed from the index. */
  indexQueryset?: (qs: QuerySet<Instance<M>>) => QuerySet<Instance<M>>
  /** Re-index on `postSave` and remove on `postDelete` (default true). */
  autoIndex?: boolean
  /** Run automatic re-indexing as a task (`'auto'`, the default: when `@ghuri/tasks` is loaded). */
  background?: boolean | 'auto'
}

/** A search hit with the model instance it came from. */
export interface ModelSearchHit<T> extends SearchHit {
  object: T
}

/**
 * An index of model rows (django-haystack's `SearchIndex`): the text and metadata of each row, kept up to date on
 * `postSave` / `postDelete`.
 *
 * ```ts
 * export const posts = new ModelIndex(Post, {
 *   text: (p) => `${p.title}\n\n${p.body}`,
 *   metadata: (p) => ({ status: p.status }),
 *   indexQueryset: (qs) => qs.filter({ status: 'published' }),
 * })
 * const hits = await posts.search('refund policy', { k: 3 })   // hits[0].object is a Post
 * ```
 */
export class ModelIndex<M extends ModelClass> extends VectorIndex {
  declare readonly options: ModelIndexOptions<M>

  constructor(
    readonly model: M,
    opts: ModelIndexOptions<M>,
  ) {
    super(opts.name ?? modelLabel(model), opts)
    if (opts.autoIndex !== false) this.connectSignals()
  }

  private queryset(): QuerySet<Instance<M>> {
    const qs = (this.model as any).objects.all() as QuerySet<Instance<M>>
    return this.options.indexQueryset ? this.options.indexQueryset(qs) : qs
  }

  /** The document for a row. */
  document(obj: Instance<M>): Document {
    const o = obj as any
    const text =
      typeof this.options.text === 'function'
        ? this.options.text(obj)
        : this.options.text
            .map((f) => o[f])
            .filter((v) => v !== null && v !== undefined && v !== '')
            .map(String)
            .join('\n\n')
    return { key: String(o.pk), text, metadata: this.options.metadata ? this.options.metadata(obj) : {} }
  }

  /** Index rows (those outside `indexQueryset` are removed instead). */
  override async index(
    objs: Instance<M> | Instance<M>[] | Document | Document[],
    opts: IndexOptions = {},
  ): Promise<IndexResult> {
    const all = list(objs as any[])
    const pks = all.map((o) => (isDocument(o) ? o.key : (o as any).pk))
    if (pks.some((pk) => pk === null || pk === undefined))
      throw new VectorsError('Only saved rows can be indexed.')
    if (useBackground(opts.background)) {
      const result = await enqueueVectorJob({
        index: this.name,
        op: 'index',
        keys: pks.map(String),
        ...(opts.namespace ? { namespace: opts.namespace } : {}),
        ...(opts.force ? { force: true } : {}),
      })
      return { indexed: 0, skipped: 0, deleted: 0, chunks: 0, taskResultId: result.id }
    }
    return this.indexPks(pks, opts)
  }

  /** Load rows by primary key and index them; keys with no row in `indexQueryset` are removed. */
  async indexPks(pks: unknown[], opts: { namespace?: string; force?: boolean } = {}): Promise<IndexResult> {
    const keys = pks.map(String)
    const found = new Map<string, Instance<M>>()
    for (const batch of batches(pks, 90)) {
      for (const row of await this.queryset().filter({ pk__in: batch }))
        found.set(String((row as any).pk), row)
    }
    const result = await this.indexDocuments(
      [...found.values()].map((o) => this.document(o)),
      opts,
    )
    const gone = keys.filter((k) => !found.has(k))
    if (gone.length) {
      await this.delete(gone, opts.namespace ? { namespace: opts.namespace } : {})
      result.deleted += gone.length
    }
    return result
  }

  /** Remove rows (instances or primary keys) from the index. */
  override async delete(
    objsOrPks: Key | Key[] | Instance<M> | Instance<M>[],
    opts: { namespace?: string; background?: boolean | 'auto' } = {},
  ): Promise<void> {
    const keys = list(objsOrPks as any[]).map((o) => (o && typeof o === 'object' ? (o as any).pk : o))
    return super.delete(keys as Key[], opts)
  }

  /**
   * Index every row of `indexQueryset` (`ghuri update_index`), skipping unchanged ones, and remove documents whose
   * rows are gone. Runs in the current tenant's namespace when the index is tenant-scoped.
   */
  async update(opts: { batchSize?: number; force?: boolean; namespace?: string } = {}): Promise<IndexResult> {
    await this.namespace(opts.namespace, 'Updating')
    const pks = (await this.queryset().orderBy('pk').valuesList('pk', { flat: true })) as unknown[]
    const total: IndexResult = { indexed: 0, skipped: 0, deleted: 0, chunks: 0 }
    for (const batch of batches(pks, Math.max(1, opts.batchSize ?? 100))) {
      const r = await this.indexPks(batch, opts)
      total.indexed += r.indexed
      total.skipped += r.skipped
      total.deleted += r.deleted
      total.chunks += r.chunks
    }
    if (manifestAvailable()) {
      const current = new Set(pks.map(String))
      const indexed = await this.keys(opts.namespace ? { namespace: opts.namespace } : {})
      const gone = indexed.filter((k) => !current.has(k))
      if (gone.length) {
        await this.delete(gone, opts.namespace ? { namespace: opts.namespace } : {})
        total.deleted += gone.length
      }
    }
    return total
  }

  /** Search, with each hit's row loaded (one query); hits whose row is gone are dropped. */
  override async search(
    query: string | number[],
    opts: SearchOptions = {},
  ): Promise<ModelSearchHit<Instance<M>>[]> {
    const hits = await super.search(query, opts)
    if (!hits.length) return []
    const rows = await (this.model as any).objects.filter({ pk__in: [...new Set(hits.map((h) => h.key))] })
    const byKey = new Map<string, Instance<M>>()
    for (const row of rows) byKey.set(String(row.pk), row)
    return hits.flatMap((h) => (byKey.has(h.key) ? [{ ...h, object: byKey.get(h.key)! }] : []))
  }

  override async runJob(job: VectorJob): Promise<IndexResult | null> {
    if (job.op === 'delete') return super.runJob(job)
    return this.indexPks(job.keys ?? [], {
      ...(job.namespace ? { namespace: job.namespace } : {}),
      ...(job.force ? { force: true } : {}),
    })
  }

  private connectSignals(): void {
    const background = this.options.background ?? 'auto'
    // A new index with the same name replaces the old one's receivers (tests, reloaded modules).
    this.disconnectSignals()
    postSave.connect(
      async ({ instance, raw, using }) => {
        if (raw) return
        await onCommit(() => this.index(instance as Instance<M>, { background }), { using })
      },
      { sender: this.model, uid: `ghuri.vectors.save:${this.name}` },
    )
    postDelete.connect(
      async ({ instance, using }) => {
        const pk = (instance as any).pk
        if (pk === null || pk === undefined) return
        await onCommit(() => this.delete(pk, { background }), { using })
      },
      { sender: this.model, uid: `ghuri.vectors.delete:${this.name}` },
    )
  }

  /** Stop re-indexing on save / delete. */
  disconnectSignals(): void {
    postSave.disconnect(`ghuri.vectors.save:${this.name}`, this.model)
    postDelete.disconnect(`ghuri.vectors.delete:${this.name}`, this.model)
  }
}

function isDocument(value: unknown): value is Document {
  return (
    !!value && typeof value === 'object' && typeof (value as Document).text === 'string' && 'key' in value
  )
}

function modelLabel(model: ModelClass): string {
  const meta = (model as any)._meta
  if (!meta?.hasAppLabel) {
    throw new ImproperlyConfigured(
      `ModelIndex(${(model as any).name}) needs a \`name\`: the model has no app label yet (set Meta.appLabel or pass { name }).`,
    )
  }
  return meta.labelLower
}

setJobRunner(async (job) => {
  const index = getVectorIndex(job.index)
  if (!index) {
    throw new ImproperlyConfigured(
      `No vector index named '${job.index}': define it in a module the worker loads (an app's vectors.ts imported by app.ts).`,
    )
  }
  return index.runJob(job)
})
