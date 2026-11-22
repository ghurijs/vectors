// The store backends every contract test runs against.
import type { VectorStoreConfig } from '../src/index'
import { databaseStore, memoryStore, pgvector, vectorize } from '../src/index'
import { type FakeEmbedder, fakeEmbedding, fakeVectorize } from '../src/testing'
import { dbBackends, type HarnessOptions } from './helpers'

export interface StoreCase {
  name: string
  native: boolean
  embedder: FakeEmbedder
  harness(): HarnessOptions
  /** Called after the harness is set up (create pgvector tables…). */
  prepare?(): Promise<void>
}

export function storeCases(extraSettings: Record<string, unknown> = {}): StoreCase[] {
  const cases: StoreCase[] = []
  const make = (
    name: string,
    native: boolean,
    using: HarnessOptions['using'],
    store: (e: FakeEmbedder) => VectorStoreConfig,
    env?: () => Record<string, unknown>,
  ) => {
    const embedder = fakeEmbedding({ dimensions: 64 })
    const config = store(embedder)
    cases.push({
      name,
      native,
      embedder,
      harness: () => ({
        using,
        env: env?.() ?? {},
        settings: { VECTORS: { default: config }, ...extraSettings },
      }),
      ...(name === 'pgvector'
        ? {
            async prepare() {
              const { vectorStores } = await import('../src/index')
              await vectorStores.get().setup!()
            },
          }
        : {}),
    })
  }
  make('memory', true, 'sqlite', (embedding) => memoryStore({ embedding }))
  for (const using of dbBackends)
    make(`database (${using})`, true, using, (embedding) => databaseStore({ embedding }))
  make('pgvector', true, 'pgvector', (embedding) => pgvector({ embedding }))
  make(
    'vectorize',
    false,
    'sqlite',
    (embedding) => vectorize('VECTORIZE', { embedding }),
    () => ({
      VECTORIZE: fakeVectorize({ dimensions: 64 }),
    }),
  )
  return cases
}
