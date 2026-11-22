import { ImproperlyConfigured, tryGetContext } from '@ghuri/core'
import { InvalidVectorStore } from './errors'
import { globalSingleton } from './global'
import type { VectorStore } from './stores/base'

export const DEFAULT_VECTOR_STORE_ALIAS = 'default'

/**
 * An entry of the `VECTORS` setting, made by a factory (`vectorize()`, `pgvector()`, `databaseStore()`,
 * `memoryStore()`), like `CACHES` and `TASKS` entries. One store instance is created per alias and kept.
 */
export interface VectorStoreConfig<S extends VectorStore = VectorStore> {
  /** `'vectorize'`, `'pgvector'`, `'database'`, `'memory'`. */
  readonly backend: string
  create(alias: string): S
}

declare module '@ghuri/core' {
  interface GhuriSettings {
    /** `{ default: vectorize('VECTORIZE', { embedding: 'workers-ai:@cf/baai/bge-base-en-v1.5' }) }`, … */
    VECTORS?: Record<string, VectorStoreConfig>
  }
}

/** @internal Build a config whose store instances are created once per alias. */
export function storeConfig<S extends VectorStore>(
  backend: string,
  make: (alias: string) => S,
): VectorStoreConfig<S> {
  const instances = new Map<string, S>()
  return {
    backend,
    create(alias: string): S {
      let s = instances.get(alias)
      if (!s) {
        s = make(alias)
        instances.set(alias, s)
      }
      return s
    },
  }
}

const overrides: Map<string, VectorStoreConfig[]> = globalSingleton(
  'ghuri.vectors.storeOverrides',
  () => new Map(),
)

export function isStoreConfig(value: unknown): value is VectorStoreConfig {
  return typeof value === 'object' && value !== null && typeof (value as any).create === 'function'
}

/** The `VECTORS` setting of the current context (undefined when not set). */
export function vectorsSetting(): Record<string, unknown> | undefined {
  return tryGetContext()?.settings?.VECTORS as Record<string, unknown> | undefined
}

/** The vector stores of the `VECTORS` setting (like `caches`). */
export const vectorStores = {
  /** The `VECTORS` entry for `alias` (with test overrides applied), or undefined. */
  config(alias: string = DEFAULT_VECTOR_STORE_ALIAS): VectorStoreConfig | undefined {
    const stack = overrides.get(alias)
    if (stack?.length) return stack[stack.length - 1]
    const entry = vectorsSetting()?.[alias]
    if (entry === undefined) return undefined
    if (!isStoreConfig(entry)) {
      throw new InvalidVectorStore(
        `VECTORS['${alias}'] is not a vector store: use vectorize(), pgvector(), databaseStore() or memoryStore().`,
      )
    }
    return entry
  },

  /** The store for `alias`. Raises `ImproperlyConfigured` when it is not configured. */
  get(alias: string = DEFAULT_VECTOR_STORE_ALIAS): VectorStore {
    const config = this.config(alias)
    if (!config) {
      throw new ImproperlyConfigured(
        vectorsSetting()
          ? `The vector store '${alias}' is not configured (settings.VECTORS).`
          : `settings.VECTORS is not set: add { ${alias}: vectorize(...) | pgvector(...) | databaseStore(...) } to use vector indexes.`,
      )
    }
    return config.create(alias)
  },

  /** Configured aliases. */
  aliases(): string[] {
    const names = new Set(Object.keys(vectorsSetting() ?? {}))
    for (const [alias, stack] of overrides) if (stack.length) names.add(alias)
    return [...names]
  },

  /** Every configured store. */
  all(): VectorStore[] {
    return this.aliases().map((a) => this.get(a))
  },

  /** Use `config` for `alias` until the returned function is called (tests; see `useVectorStore`). */
  override(alias: string, config: VectorStoreConfig): () => void {
    const stack = overrides.get(alias) ?? []
    stack.push(config)
    overrides.set(alias, stack)
    return () => {
      const i = stack.lastIndexOf(config)
      if (i >= 0) stack.splice(i, 1)
    }
  },
}
