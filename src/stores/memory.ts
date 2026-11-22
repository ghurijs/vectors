import { type CheckMessage, checks, setting } from '@ghuri/core'
import { matches } from '../filters'
import { byScore, checkDimensions, type Metric, score } from '../math'
import { storeConfig, type VectorStoreConfig } from '../registry'
import {
  BaseVectorStore,
  type StoreOptions,
  type StoreQueryOptions,
  type VectorMatch,
  type VectorRecord,
} from './base'

/** Exact search over records in a list (the memory and database stores). */
export function bruteForce(
  records: Iterable<VectorRecord>,
  vector: number[],
  opts: StoreQueryOptions,
  metric: Metric,
): VectorMatch[] {
  const out: VectorMatch[] = []
  for (const r of records) {
    if (opts.namespace !== undefined && (r.namespace ?? '') !== opts.namespace) continue
    if (opts.filter?.length && !matches(opts.filter, r.metadata)) continue
    out.push({
      id: r.id,
      score: score(metric, vector, r.values),
      ...(r.namespace ? { namespace: r.namespace } : {}),
      metadata: { ...(r.metadata ?? {}) },
      ...(opts.returnValues ? { values: [...r.values] } : {}),
    })
  }
  return out.sort(byScore(metric)).slice(0, opts.topK)
}

/** Vectors in the process. For tests and scripts; everything is lost on restart. */
export class MemoryVectorStore extends BaseVectorStore {
  readonly backend = 'memory'
  readonly records = new Map<string, VectorRecord>()

  async upsert(records: VectorRecord[]): Promise<void> {
    this.noteDimensions(records)
    checkDimensions(
      records.map((r) => r.values),
      this.dimensions,
      `Vector store '${this.alias}'`,
    )
    for (const r of records) {
      this.records.set(r.id, {
        id: r.id,
        values: [...r.values],
        ...(r.namespace ? { namespace: r.namespace } : {}),
        metadata: { ...(r.metadata ?? {}) },
      })
    }
  }

  async query(vector: number[], opts: StoreQueryOptions): Promise<VectorMatch[]> {
    checkDimensions([vector], this.dimensions, `Vector store '${this.alias}'`)
    return bruteForce(this.records.values(), vector, opts, this.metric)
  }

  async getByIds(ids: string[]): Promise<VectorRecord[]> {
    return ids.flatMap((id) => {
      const r = this.records.get(id)
      return r ? [{ ...r, values: [...r.values], metadata: { ...(r.metadata ?? {}) } }] : []
    })
  }

  async deleteByIds(ids: string[]): Promise<void> {
    for (const id of ids) this.records.delete(id)
  }

  async deleteDocuments(namespace: string, keys: string[]): Promise<void> {
    const wanted = new Set(keys)
    for (const [id, r] of this.records) {
      if ((r.namespace ?? '') === namespace && wanted.has(String(r.metadata?._key))) this.records.delete(id)
    }
  }

  async clearNamespace(namespace: string): Promise<void> {
    for (const [id, r] of this.records) if ((r.namespace ?? '') === namespace) this.records.delete(id)
  }

  /** Delete every vector (tests). */
  clear(): void {
    this.records.clear()
  }

  override check(): CheckMessage[] {
    const out = super.check()
    if (!setting('DEBUG', false)) {
      out.push(
        checks.warning(
          `The vector store '${this.alias}' keeps vectors in memory and loses them on restart.`,
          {
            hint: 'Use vectorize() on Workers or pgvector() on Postgres in production.',
            id: 'vectors.W001',
            obj: `VECTORS['${this.alias}']`,
          },
        ),
      )
    }
    return out
  }
}

/** `VECTORS: { default: memoryStore({ embedding }) }` — vectors in the process (tests, scripts). */
export function memoryStore(opts: StoreOptions = {}): VectorStoreConfig<MemoryVectorStore> {
  return storeConfig('memory', (alias) => new MemoryVectorStore(alias, opts))
}
