// Tenant scoping: a scoped index takes the tenant from core's context carriers (`captureContext().tenant`, which
// @ghuri/tenants registers) and puts it into the store namespace. No tenant means `TenantRequired`: fail closed.
import { captureContext, getContextCarriers } from '@ghuri/core'
import { TenantRequired, VectorsError } from './errors'

/** Name of the tenant carrier of `@ghuri/tenants`. */
export const TENANT_CARRIER = 'tenant'

/** How an index is scoped to tenants: `'auto'` (when a tenant carrier is registered), always or never. */
export type TenantScope = 'auto' | boolean

/** Is an index with this option scoped to tenants right now? */
export function isTenantScoped(scope: TenantScope): boolean {
  return scope === 'auto' ? getContextCarriers().includes(TENANT_CARRIER) : scope
}

/** The carried tenant primary key, or undefined. */
export function carriedTenant(): string | undefined {
  return captureContext()[TENANT_CARRIER]
}

const encoder = new TextEncoder()

/** Hex SHA-256 of `text`. */
export async function sha256Hex(text: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', encoder.encode(text))
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

/** JSON with object keys sorted, for digests. */
export function stableStringify(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(',')}]`
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .filter(([, v]) => v !== undefined)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableStringify(v)}`).join(',')}}`
  }
  return JSON.stringify(value ?? null)
}

const MAX_NAMESPACE_BYTES = 64

/**
 * The store namespace of an index: `<index>`, plus `|t=<tenant>` when scoped and `|<namespace>` when given. Longer
 * than 64 bytes (Vectorize's limit) it becomes `h:<sha-256>`.
 */
export async function resolveNamespace(
  index: string,
  scope: TenantScope,
  namespace?: string | null,
  what = 'This operation',
): Promise<string> {
  const parts = [index]
  if (isTenantScoped(scope)) {
    const tenant = carriedTenant()
    if (!tenant) {
      throw new TenantRequired(
        `${what} on the tenant-scoped vector index '${index}' needs a current tenant: run it inside a request handled by TenantMiddleware or runAsTenant(tenant, fn). Use { tenant: false } for indexes shared by every tenant.`,
      )
    }
    parts.push(`t=${tenant}`)
  }
  if (namespace !== undefined && namespace !== null && namespace !== '') {
    if (namespace.includes('|')) throw new VectorsError(`Invalid namespace '${namespace}': '|' is reserved.`)
    parts.push(namespace)
  }
  const ns = parts.join('|')
  if (encoder.encode(ns).length <= MAX_NAMESPACE_BYTES) return ns
  return `h:${(await sha256Hex(ns)).slice(0, 62)}`
}

/** A chunk's vector id: 40 hex characters of SHA-256 over namespace and key, then `-<chunk>` (at most 64 bytes). */
export async function vectorId(namespace: string, key: string, chunk: number): Promise<string> {
  return `${(await sha256Hex(`${namespace}\u0000${key}`)).slice(0, 40)}-${chunk}`
}

/** Ids of chunks `from` (inclusive) to `to` (exclusive) of a document. */
export async function chunkIds(namespace: string, key: string, from: number, to: number): Promise<string[]> {
  if (to <= from) return []
  const prefix = (await sha256Hex(`${namespace}\u0000${key}`)).slice(0, 40)
  const out: string[] = []
  for (let i = from; i < to; i++) out.push(`${prefix}-${i}`)
  return out
}
