import { apps, type CheckMessage, checks, setting } from '@ghuri/core'
import { checkDimensions, decodeVector, encodeVector } from '../math'
import { VectorEmbedding } from '../models'
import { storeConfig, type VectorStoreConfig } from '../registry'
import {
  BaseVectorStore,
  batches,
  type StoreOptions,
  type StoreQueryOptions,
  type VectorMatch,
  type VectorRecord,
} from './base'
import { bruteForce } from './memory'

export interface DatabaseStoreOptions extends StoreOptions {
  /** `DATABASES` alias (default `'default'`). */
  database?: string
}

// D1 binds at most 100 parameters per statement.
const ID_BATCH = 90

/**
 * Vectors in the `vectors_vectorembedding` table on any `DATABASES` alias (D1, SQLite, Postgres). Search is exact
 * and reads every row of one namespace, so it suits small apps (a few thousand chunks per namespace) and tests.
 * Needs the `vectors` app.
 */
export class DatabaseVectorStore extends BaseVectorStore {
  readonly backend = 'database'
  readonly database: string

  constructor(alias: string, opts: DatabaseStoreOptions = {}) {
    super(alias, opts)
    this.database = opts.database ?? 'default'
  }

  private get objects() {
    return VectorEmbedding.objects.using(this.database)
  }

  async upsert(records: VectorRecord[]): Promise<void> {
    if (!records.length) return
    this.noteDimensions(records)
    checkDimensions(
      records.map((r) => r.values),
      this.dimensions,
      `Vector store '${this.alias}'`,
    )
    // The last record wins when an id repeats, as in Vectorize.
    const byId = new Map(records.map((r) => [r.id, r]))
    await this.deleteByIds([...byId.keys()])
    const rows = [...byId.values()].map(
      (r) =>
        new VectorEmbedding({
          store: this.alias,
          vectorId: r.id,
          namespace: r.namespace ?? '',
          document: r.metadata?._key === undefined ? '' : String(r.metadata._key),
          embedding: encodeVector(r.values),
          metadata: { ...(r.metadata ?? {}) },
        }),
    )
    for (const batch of batches(rows, 10)) await this.objects.bulkCreate(batch)
  }

  async query(vector: number[], opts: StoreQueryOptions): Promise<VectorMatch[]> {
    checkDimensions([vector], this.dimensions, `Vector store '${this.alias}'`)
    let qs = this.objects.filter({ store: this.alias })
    if (opts.namespace !== undefined) qs = qs.filter({ namespace: opts.namespace })
    const rows = (await qs.values('vectorId', 'namespace', 'embedding', 'metadata')) as Array<{
      vectorId: string
      namespace: string
      embedding: string
      metadata: Record<string, any>
    }>
    return bruteForce(
      rows.map((r) => ({
        id: r.vectorId,
        values: decodeVector(r.embedding),
        ...(r.namespace ? { namespace: r.namespace } : {}),
        metadata: r.metadata ?? {},
      })),
      vector,
      opts,
      this.metric,
    )
  }

  async getByIds(ids: string[]): Promise<VectorRecord[]> {
    const found = new Map<string, VectorRecord>()
    for (const batch of batches(ids, ID_BATCH)) {
      const rows = await this.objects.filter({ store: this.alias, vectorId__in: batch })
      for (const r of rows) {
        found.set(r.vectorId, {
          id: r.vectorId,
          values: decodeVector(r.embedding),
          ...(r.namespace ? { namespace: r.namespace } : {}),
          metadata: (r.metadata as any) ?? {},
        })
      }
    }
    return ids.flatMap((id) => (found.has(id) ? [found.get(id)!] : []))
  }

  async deleteByIds(ids: string[]): Promise<void> {
    for (const batch of batches(ids, ID_BATCH)) {
      await this.objects.filter({ store: this.alias, vectorId__in: batch }).delete()
    }
  }

  async deleteDocuments(namespace: string, keys: string[]): Promise<void> {
    for (const batch of batches(keys, ID_BATCH)) {
      await this.objects.filter({ store: this.alias, namespace, document__in: batch }).delete()
    }
  }

  async clearNamespace(namespace: string): Promise<void> {
    await this.objects.filter({ store: this.alias, namespace }).delete()
  }

  override check(): CheckMessage[] {
    const out = super.check()
    if (!apps.ready || !apps.getAppConfigs().some((a) => a.label === 'vectors')) {
      out.push(
        checks.error(`The vector store '${this.alias}' (databaseStore) needs the 'vectors' app.`, {
          hint: "Add VectorsConfig from '@ghuri/vectors' to INSTALLED_APPS and run ghuri migrate.",
          id: 'vectors.E003',
          obj: `VECTORS['${this.alias}']`,
        }),
      )
    }
    if (!setting('DEBUG', false)) {
      out.push(
        checks.warning(
          `The vector store '${this.alias}' searches by reading every vector of a namespace; it suits small indexes only.`,
          {
            hint: 'Use vectorize() on Workers or pgvector() on Postgres for larger indexes.',
            id: 'vectors.W002',
            obj: `VECTORS['${this.alias}']`,
          },
        ),
      )
    }
    return out
  }
}

/** `VECTORS: { default: databaseStore({ database: 'default', embedding }) }` — exact search on any database. */
export function databaseStore(opts: DatabaseStoreOptions = {}): VectorStoreConfig<DatabaseVectorStore> {
  return storeConfig('database', (alias) => new DatabaseVectorStore(alias, opts))
}
