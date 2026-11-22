import { apps, type CheckMessage, checks, ImproperlyConfigured, tryGetContext } from '@ghuri/core'
import { toVectorizeFilter } from '../filters'
import { checkDimensions } from '../math'
import { storeConfig, type VectorStoreConfig } from '../registry'
import {
  BaseVectorStore,
  batches,
  type StoreOptions,
  type StoreQueryOptions,
  type VectorMatch,
  type VectorRecord,
} from './base'

/** The parts of a Workers `VectorizeIndex` binding (Vectorize V2) Ghuri uses. */
export interface VectorizeIndexLike {
  upsert(vectors: VectorRecord[]): Promise<unknown>
  query(
    vector: number[],
    options: {
      topK?: number
      namespace?: string
      returnValues?: boolean
      returnMetadata?: boolean | 'none' | 'indexed' | 'all'
      filter?: Record<string, unknown>
    },
  ): Promise<{ matches: VectorMatch[]; count?: number }>
  getByIds(ids: string[]): Promise<VectorRecord[]>
  deleteByIds(ids: string[]): Promise<unknown>
}

export interface VectorizeOptions extends StoreOptions {}

// The binding takes up to 1000 vectors per upsert; stay well below to keep requests small.
const UPSERT_BATCH = 500
const ID_BATCH = 500

/**
 * A Cloudflare Vectorize index, through its Workers binding. Namespaces isolate tenants natively. Vectorize cannot
 * delete by metadata, so deleting documents needs the `vectors` app's manifest.
 */
export class VectorizeStore extends BaseVectorStore {
  readonly backend = 'vectorize'
  override readonly nativeDeletes = false

  constructor(
    alias: string,
    readonly binding: string,
    opts: VectorizeOptions = {},
  ) {
    super(alias, opts)
  }

  /** The binding of the current request. */
  index(): VectorizeIndexLike {
    const b = tryGetContext()?.env?.[this.binding] as VectorizeIndexLike | undefined
    if (!b || typeof b.query !== 'function') {
      throw new ImproperlyConfigured(
        `Vectorize binding '${this.binding}' (VECTORS['${this.alias}']) not found in env. Add "vectorize": [{ "binding": "${this.binding}", "index_name": "…" }] to wrangler.jsonc.`,
      )
    }
    return b
  }

  async upsert(records: VectorRecord[]): Promise<void> {
    if (!records.length) return
    this.noteDimensions(records)
    checkDimensions(
      records.map((r) => r.values),
      this.dimensions,
      `Vector store '${this.alias}'`,
    )
    const index = this.index()
    for (const batch of batches(records, UPSERT_BATCH)) {
      await index.upsert(
        batch.map((r) => ({
          id: r.id,
          values: r.values,
          ...(r.namespace ? { namespace: r.namespace } : {}),
          ...(r.metadata ? { metadata: r.metadata } : {}),
        })),
      )
    }
  }

  async query(vector: number[], opts: StoreQueryOptions): Promise<VectorMatch[]> {
    checkDimensions([vector], this.dimensions, `Vector store '${this.alias}'`)
    const filter = opts.filter?.length ? toVectorizeFilter(opts.filter) : undefined
    const res = await this.index().query(vector, {
      topK: opts.topK,
      ...(opts.namespace !== undefined ? { namespace: opts.namespace } : {}),
      returnMetadata: 'all',
      returnValues: !!opts.returnValues,
      ...(filter ? { filter } : {}),
    })
    return (res?.matches ?? []).map((m) => ({
      id: m.id,
      score: m.score,
      ...(m.namespace ? { namespace: m.namespace } : {}),
      metadata: m.metadata ?? {},
      ...(opts.returnValues && m.values ? { values: Array.from(m.values) } : {}),
    }))
  }

  async getByIds(ids: string[]): Promise<VectorRecord[]> {
    const out: VectorRecord[] = []
    const index = this.index()
    for (const batch of batches(ids, ID_BATCH)) {
      for (const r of await index.getByIds(batch)) out.push({ ...r, values: Array.from(r.values) })
    }
    return out
  }

  async deleteByIds(ids: string[]): Promise<void> {
    const index = this.index()
    for (const batch of batches(ids, ID_BATCH)) await index.deleteByIds(batch)
  }

  override check(): CheckMessage[] {
    const out = super.check()
    if (!apps.ready || !apps.getAppConfigs().some((a) => a.label === 'vectors')) {
      out.push(
        checks.error(
          `The Vectorize store '${this.alias}' needs the 'vectors' app to delete and re-index documents.`,
          {
            hint: "Add VectorsConfig from '@ghuri/vectors' to INSTALLED_APPS and run ghuri migrate.",
            id: 'vectors.E003',
            obj: `VECTORS['${this.alias}']`,
          },
        ),
      )
    }
    return out
  }
}

/**
 * `VECTORS: { default: vectorize('VECTORIZE', { embedding: 'workers-ai:@cf/baai/bge-base-en-v1.5' }) }` — a
 * Cloudflare Vectorize index. Create it with the embedding model's dimensions and metric:
 * `wrangler vectorize create mysite-default --dimensions=768 --metric=cosine`.
 */
export function vectorize(binding: string, opts: VectorizeOptions = {}): VectorStoreConfig<VectorizeStore> {
  return storeConfig('vectorize', (alias) => new VectorizeStore(alias, binding, opts))
}
