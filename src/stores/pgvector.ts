import { type CheckMessage, checks, ImproperlyConfigured } from '@ghuri/core'
import { type Connection, connections } from '@ghuri/orm'
import { InvalidVectorStore, VectorsError } from '../errors'
import type { Condition, Metadata, MetadataValue } from '../filters'
import { checkDimensions, type Metric } from '../math'
import { storeConfig, type VectorStoreConfig } from '../registry'
import {
  BaseVectorStore,
  batches,
  type StoreOptions,
  type StoreQueryOptions,
  type VectorMatch,
  type VectorRecord,
} from './base'

export interface PgvectorOptions extends StoreOptions {
  /** `DATABASES` alias of a Postgres database with the `vector` extension (default `'default'`). */
  database?: string
  /** Table name (default `'vectors_pgembedding'`, `'vectors_pgembedding_<alias>'` for other aliases). */
  table?: string
  /** Approximate index: `'hnsw'` (default), `'ivfflat'` or `null` for exact search. */
  index?: 'hnsw' | 'ivfflat' | null
  /** HNSW `m` (default 16). */
  m?: number
  /** HNSW `ef_construction` (default 64). */
  efConstruction?: number
  /** IVFFlat `lists` (default 100). */
  lists?: number
}

/** pgvector's distance operator and opclass for each metric. */
export const PGVECTOR_OPS: Record<Metric, { operator: string; opclass: string }> = {
  cosine: { operator: '<=>', opclass: 'vector_cosine_ops' },
  euclidean: { operator: '<->', opclass: 'vector_l2_ops' },
  'dot-product': { operator: '<#>', opclass: 'vector_ip_ops' },
}

function quote(name: string): string {
  return `"${name}"`
}

/** `[1,2,3]`, pgvector's text form. */
export function vectorLiteral(values: number[]): string {
  for (const v of values) {
    if (!Number.isFinite(v)) throw new VectorsError('Vectors must contain finite numbers only.')
  }
  return `[${values.join(',')}]`
}

/** Translate metadata conditions into SQL on the `metadata` jsonb column; `params` receives the values. */
export function filterSql(conditions: Condition[], params: unknown[]): string[] {
  const p = (value: unknown) => {
    params.push(value)
    return `$${params.length}`
  }
  const out: string[] = []
  for (const c of conditions) {
    const field = `"metadata" -> ${p(c.key)}::text`
    switch (c.op) {
      case '$eq':
        out.push(`${field} = ${p(JSON.stringify(c.value))}::jsonb`)
        break
      case '$ne':
        out.push(`(${field} IS NULL OR ${field} <> ${p(JSON.stringify(c.value))}::jsonb)`)
        break
      case '$in':
      case '$nin': {
        const list = (c.value as MetadataValue[]).map((v) => `${p(JSON.stringify(v))}::jsonb`).join(', ')
        out.push(c.op === '$in' ? `${field} IN (${list})` : `(${field} IS NULL OR ${field} NOT IN (${list}))`)
        break
      }
      default: {
        const op = { $lt: '<', $lte: '<=', $gt: '>', $gte: '>=' }[c.op]
        if (typeof c.value === 'number') {
          out.push(
            `(CASE WHEN jsonb_typeof(${field}) = 'number' THEN ("metadata" ->> ${p(c.key)}::text)::float8 END) ${op} ${p(c.value)}::float8`,
          )
        } else {
          out.push(
            `(CASE WHEN jsonb_typeof(${field}) = 'string' THEN ("metadata" ->> ${p(c.key)}::text) END) COLLATE "C" ${op} ${p(String(c.value))}::text`,
          )
        }
      }
    }
  }
  return out
}

function parseMetadata(value: unknown): Metadata {
  if (typeof value === 'string') return JSON.parse(value)
  return (value as Metadata) ?? {}
}

/**
 * Vectors in a Postgres table with pgvector (`vector(n)` column, `hnsw` or `ivfflat` index). Works wherever
 * `@ghuri/orm` reaches Postgres: postgres.js, Hyperdrive on Workers, PGlite with `@electric-sql/pglite-pgvector`.
 * Create the table with `ghuri createvectortable`.
 */
export class PgvectorStore extends BaseVectorStore {
  readonly backend = 'pgvector'
  readonly database: string
  readonly table: string

  constructor(
    alias: string,
    private readonly pgOptions: PgvectorOptions = {},
  ) {
    super(alias, pgOptions)
    this.database = pgOptions.database ?? 'default'
    this.table =
      pgOptions.table ?? (alias === 'default' ? 'vectors_pgembedding' : `vectors_pgembedding_${alias}`)
    if (!/^[A-Za-z_][A-Za-z0-9_]{0,62}$/.test(this.table)) {
      throw new InvalidVectorStore(`VECTORS['${alias}']: invalid pgvector table name '${this.table}'.`)
    }
  }

  private async connection(): Promise<Connection> {
    const conn = await connections.get(this.database)
    if (conn.dialect !== 'postgresql') {
      throw new InvalidVectorStore(
        `VECTORS['${this.alias}'] (pgvector) needs a Postgres database, but DATABASES['${this.database}'] is ${conn.vendor}.`,
      )
    }
    return conn
  }

  private async run(sql: string, params: unknown[] = []) {
    const conn = await this.connection()
    try {
      return await conn.execute(sql, params)
    } catch (err) {
      const message = String((err as Error)?.message ?? err)
      if (/relation .* does not exist|type "vector" does not exist/i.test(message)) {
        throw new VectorsError(
          `The pgvector table '${this.table}' does not exist: run ghuri createvectortable (store '${this.alias}').`,
          { cause: err },
        )
      }
      throw err
    }
  }

  /** The statements `createvectortable` runs. */
  setupSql(): string[] {
    const dims = this.dimensions
    if (!dims) {
      throw new ImproperlyConfigured(
        `VECTORS['${this.alias}'] (pgvector) needs \`dimensions\` (the embedding model's are not known).`,
      )
    }
    const t = quote(this.table)
    const sql = [
      'CREATE EXTENSION IF NOT EXISTS vector',
      `CREATE TABLE IF NOT EXISTS ${t} ("id" text PRIMARY KEY, "namespace" text NOT NULL DEFAULT '', "document" text NOT NULL DEFAULT '', "embedding" vector(${dims}) NOT NULL, "metadata" jsonb NOT NULL DEFAULT '{}'::jsonb)`,
      `CREATE INDEX IF NOT EXISTS ${quote(`${this.table}_doc_idx`)} ON ${t} ("namespace", "document")`,
    ]
    const index = this.pgOptions.index === undefined ? 'hnsw' : this.pgOptions.index
    const { opclass } = PGVECTOR_OPS[this.metric]
    if (index === 'hnsw') {
      sql.push(
        `CREATE INDEX IF NOT EXISTS ${quote(`${this.table}_embedding_idx`)} ON ${t} USING hnsw ("embedding" ${opclass}) WITH (m = ${Math.trunc(this.pgOptions.m ?? 16)}, ef_construction = ${Math.trunc(this.pgOptions.efConstruction ?? 64)})`,
      )
    } else if (index === 'ivfflat') {
      sql.push(
        `CREATE INDEX IF NOT EXISTS ${quote(`${this.table}_embedding_idx`)} ON ${t} USING ivfflat ("embedding" ${opclass}) WITH (lists = ${Math.trunc(this.pgOptions.lists ?? 100)})`,
      )
    }
    return sql
  }

  async setup(): Promise<void> {
    const conn = await this.connection()
    for (const statement of this.setupSql()) await conn.execute(statement)
  }

  /** Drop the table (tests). */
  async drop(): Promise<void> {
    await (await this.connection()).execute(`DROP TABLE IF EXISTS ${quote(this.table)}`)
  }

  async upsert(records: VectorRecord[]): Promise<void> {
    if (!records.length) return
    this.noteDimensions(records)
    checkDimensions(
      records.map((r) => r.values),
      this.dimensions,
      `Vector store '${this.alias}'`,
    )
    const unique = [...new Map(records.map((r) => [r.id, r])).values()]
    for (const batch of batches(unique, 100)) {
      const params: unknown[] = []
      const rows = batch.map((r) => {
        params.push(
          r.id,
          r.namespace ?? '',
          r.metadata?._key === undefined ? '' : String(r.metadata._key),
          vectorLiteral(r.values),
          JSON.stringify(r.metadata ?? {}),
        )
        const n = params.length
        return `($${n - 4}, $${n - 3}, $${n - 2}, $${n - 1}::vector, $${n}::jsonb)`
      })
      await this.run(
        `INSERT INTO ${quote(this.table)} ("id", "namespace", "document", "embedding", "metadata") VALUES ${rows.join(', ')} ON CONFLICT ("id") DO UPDATE SET "namespace" = EXCLUDED."namespace", "document" = EXCLUDED."document", "embedding" = EXCLUDED."embedding", "metadata" = EXCLUDED."metadata"`,
        params,
      )
    }
  }

  async query(vector: number[], opts: StoreQueryOptions): Promise<VectorMatch[]> {
    checkDimensions([vector], this.dimensions, `Vector store '${this.alias}'`)
    const { operator } = PGVECTOR_OPS[this.metric]
    const params: unknown[] = [vectorLiteral(vector)]
    const where: string[] = []
    if (opts.namespace !== undefined) {
      params.push(opts.namespace)
      where.push(`"namespace" = $${params.length}`)
    }
    where.push(...filterSql(opts.filter ?? [], params))
    params.push(Math.max(1, Math.trunc(opts.topK)))
    const res = await this.run(
      `SELECT "id", "namespace", "metadata", "embedding" ${operator} $1::vector AS "distance"${opts.returnValues ? ', "embedding"::text AS "values"' : ''} FROM ${quote(this.table)}${where.length ? ` WHERE ${where.join(' AND ')}` : ''} ORDER BY "embedding" ${operator} $1::vector LIMIT $${params.length}`,
      params,
    )
    return res.rows.map((row) => {
      const d = Number(row.distance)
      const s = this.metric === 'cosine' ? 1 - d : this.metric === 'dot-product' ? -d : d
      return {
        id: String(row.id),
        score: s,
        ...(row.namespace ? { namespace: String(row.namespace) } : {}),
        metadata: parseMetadata(row.metadata),
        ...(opts.returnValues ? { values: JSON.parse(String(row.values)) as number[] } : {}),
      }
    })
  }

  async getByIds(ids: string[]): Promise<VectorRecord[]> {
    const found = new Map<string, VectorRecord>()
    for (const batch of batches(ids, 500)) {
      const res = await this.run(
        `SELECT "id", "namespace", "metadata", "embedding"::text AS "values" FROM ${quote(this.table)} WHERE "id" IN (${batch.map((_, i) => `$${i + 1}`).join(', ')})`,
        batch,
      )
      for (const row of res.rows) {
        found.set(String(row.id), {
          id: String(row.id),
          values: JSON.parse(String(row.values)),
          ...(row.namespace ? { namespace: String(row.namespace) } : {}),
          metadata: parseMetadata(row.metadata),
        })
      }
    }
    return ids.flatMap((id) => (found.has(id) ? [found.get(id)!] : []))
  }

  async deleteByIds(ids: string[]): Promise<void> {
    for (const batch of batches(ids, 500)) {
      await this.run(
        `DELETE FROM ${quote(this.table)} WHERE "id" IN (${batch.map((_, i) => `$${i + 1}`).join(', ')})`,
        batch,
      )
    }
  }

  async deleteDocuments(namespace: string, keys: string[]): Promise<void> {
    for (const batch of batches(keys, 500)) {
      await this.run(
        `DELETE FROM ${quote(this.table)} WHERE "namespace" = $1 AND "document" IN (${batch.map((_, i) => `$${i + 2}`).join(', ')})`,
        [namespace, ...batch],
      )
    }
  }

  async clearNamespace(namespace: string): Promise<void> {
    await this.run(`DELETE FROM ${quote(this.table)} WHERE "namespace" = $1`, [namespace])
  }

  override check(): CheckMessage[] {
    const out = super.check()
    if (!this.dimensions) {
      out.push(
        checks.error(`The pgvector store '${this.alias}' does not know its dimensions.`, {
          hint: 'Pass { dimensions } to pgvector(): the embedding model is not one with known dimensions.',
          id: 'vectors.E006',
          obj: `VECTORS['${this.alias}']`,
        }),
      )
    }
    return out
  }
}

/** `VECTORS: { default: pgvector({ database: 'default', embedding }) }` — pgvector on Postgres. */
export function pgvector(opts: PgvectorOptions = {}): VectorStoreConfig<PgvectorStore> {
  return storeConfig('pgvector', (alias) => new PgvectorStore(alias, opts))
}
