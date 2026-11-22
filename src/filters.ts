import { FilterError } from './errors'

/** A metadata value: what Vectorize can store and filter on. */
export type MetadataValue = string | number | boolean

/** Metadata of a vector or a document. */
export type Metadata = Record<string, MetadataValue>

/**
 * Django lookups on metadata: `{ status: 'published', year__gte: 2025, lang__in: ['en', 'bn'] }`. Supported lookups:
 * `exact` (the default), `in`, `gt`, `gte`, `lt`, `lte`. Every key must match (AND).
 */
export type Filter = Record<string, MetadataValue | MetadataValue[]>

/** A normalized comparison (Vectorize's operators). */
export type FilterOp = '$eq' | '$ne' | '$in' | '$nin' | '$lt' | '$lte' | '$gt' | '$gte'

export interface Condition {
  key: string
  op: FilterOp
  value: MetadataValue | MetadataValue[]
}

const LOOKUPS: Record<string, FilterOp> = {
  exact: '$eq',
  in: '$in',
  gt: '$gt',
  gte: '$gte',
  lt: '$lt',
  lte: '$lte',
}
const NEGATED: Partial<Record<FilterOp, FilterOp>> = { $eq: '$ne', $in: '$nin' }

function isScalar(v: unknown): v is MetadataValue {
  return typeof v === 'string' || typeof v === 'boolean' || (typeof v === 'number' && Number.isFinite(v))
}

/** Check a metadata key (letters, digits, `_` and `-`; no `$` or `.`, which Vectorize reserves). */
export function checkMetadataKey(key: string): void {
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(key)) {
    throw new FilterError(`Invalid metadata key '${key}': use 1-64 letters, digits, '_' or '-'.`)
  }
}

function parseOne(raw: string, value: unknown, negate: boolean): Condition {
  const i = raw.lastIndexOf('__')
  let key = raw
  let op: FilterOp = '$eq'
  if (i > 0 && raw.slice(i + 2) in LOOKUPS) {
    key = raw.slice(0, i)
    op = LOOKUPS[raw.slice(i + 2)]
  } else if (i > 0 && /^[a-z]+$/.test(raw.slice(i + 2))) {
    throw new FilterError(
      `Unsupported lookup '${raw.slice(i + 2)}' in '${raw}' (use exact, in, gt, gte, lt or lte).`,
    )
  }
  checkMetadataKey(key)
  if (op === '$in') {
    if (!Array.isArray(value) || !value.length || !value.every(isScalar)) {
      throw new FilterError(`'${raw}' needs a non-empty list of strings, numbers or booleans.`)
    }
  } else if (!isScalar(value)) {
    throw new FilterError(`'${raw}' needs a string, number or boolean, got ${JSON.stringify(value)}.`)
  } else if (op !== '$eq' && typeof value === 'boolean') {
    throw new FilterError(`'${raw}': booleans can only be compared for equality.`)
  }
  if (negate) {
    const neg = NEGATED[op]
    if (!neg) throw new FilterError(`exclude supports exact and in lookups only, not '${raw}'.`)
    op = neg
  }
  return { key, op, value: value as MetadataValue | MetadataValue[] }
}

/** Normalize `filter` (and `exclude`, one negated condition per key) into conditions. */
export function parseFilter(filter?: Filter | null, exclude?: Filter | null): Condition[] {
  const out: Condition[] = []
  for (const [k, v] of Object.entries(filter ?? {})) out.push(parseOne(k, v, false))
  for (const [k, v] of Object.entries(exclude ?? {})) out.push(parseOne(k, v, true))
  return out
}

function compare(a: MetadataValue, b: MetadataValue): number | null {
  if (typeof a !== typeof b || typeof a === 'boolean') return null
  return a < b ? -1 : a > b ? 1 : 0
}

/** Does `metadata` satisfy every condition? (the JS matcher of the memory and database stores) */
export function matches(conditions: Condition[], metadata: Metadata | undefined): boolean {
  const md = metadata ?? {}
  for (const c of conditions) {
    const has = Object.hasOwn(md, c.key)
    const v = md[c.key]
    switch (c.op) {
      case '$eq':
        if (!has || v !== c.value) return false
        break
      case '$ne':
        if (has && v === c.value) return false
        break
      case '$in':
        if (!has || !(c.value as MetadataValue[]).includes(v)) return false
        break
      case '$nin':
        if (has && (c.value as MetadataValue[]).includes(v)) return false
        break
      default: {
        if (!has) return false
        const r = compare(v, c.value as MetadataValue)
        if (r === null) return false
        if (c.op === '$lt' && !(r < 0)) return false
        if (c.op === '$lte' && !(r <= 0)) return false
        if (c.op === '$gt' && !(r > 0)) return false
        if (c.op === '$gte' && !(r >= 0)) return false
      }
    }
  }
  return true
}

/** Conditions as a Vectorize metadata filter (`{ year: { $gte: 2025 }, status: { $eq: 'published' } }`). */
export function toVectorizeFilter(conditions: Condition[]): Record<string, Record<string, unknown>> {
  const out: Record<string, Record<string, unknown>> = {}
  for (const c of conditions) {
    if (!out[c.key]) out[c.key] = {}
    const entry = out[c.key]
    if (c.op in entry) {
      throw new FilterError(`Two '${c.op}' conditions on '${c.key}' cannot be combined.`)
    }
    entry[c.op] = c.value
  }
  return out
}

/** Check that metadata values are strings, numbers or booleans and keys are valid. */
export function checkMetadata(metadata: Record<string, unknown> | undefined, what: string): Metadata {
  const out: Metadata = {}
  for (const [k, v] of Object.entries(metadata ?? {})) {
    if (v === undefined || v === null) continue
    if (k.startsWith('_'))
      throw new FilterError(`${what}: metadata key '${k}' is reserved (keys starting with '_').`)
    checkMetadataKey(k)
    if (!isScalar(v)) {
      throw new FilterError(`${what}: metadata '${k}' must be a string, number or boolean.`)
    }
    out[k] = v
  }
  return out
}
