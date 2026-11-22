import { type CheckContext, ImproperlyConfigured } from '@ghuri/core'
import { connections } from '@ghuri/orm'
import { MigrationLoader, makeMigrations } from '@ghuri/orm/migrations'
import { describe, expect, it } from 'vitest'
import { checkWrangler } from '../src/checks'
import {
  checkVectors,
  databaseStore,
  memoryStore,
  pgvector,
  VectorIndex,
  vectorize,
  vectorStores,
} from '../src/index'
import { useVectorStore } from '../src/testing'
import { setupHarness, useHarness } from './helpers'

function ids(messages: Array<{ id?: string }>): string[] {
  return messages.map((m) => m.id ?? '').sort()
}

describe('the vectors app', () => {
  const h = useHarness(() => ({
    settings: { VECTORS: { default: databaseStore({ embedding: 'openai:x' }) } },
  }))

  it('ships 0001_initial, applied by the test database, with no missing changes', async () => {
    const loader = await MigrationLoader.load({ connection: null })
    expect(loader.getMigration('vectors', '0001_initial').operations.map((o) => o.describe())).toEqual([
      'Create model VectorDocument',
      'Create model VectorEmbedding',
    ])
    const res = await makeMigrations({ appLabels: ['vectors'], check: true, write: false })
    expect(res.migrations).toEqual([])
    expect(await connections.get()).toBe(h.db.connection)
  })

  it('checks stores and indexes', async () => {
    new VectorIndex('test.orphan', { store: 'missing' })
    const ctx = { settings: h.ctx.settings, appConfigs: [], deploy: false } as unknown as CheckContext
    expect(ids(await checkVectors(ctx))).toEqual(['vectors.E002'])
    const restore = useVectorStore(pgvector({ embedding: 'openai:unknown-model' }), 'pg').restore
    const r2 = useVectorStore(memoryStore(), 'mem').restore
    try {
      expect(ids(await checkVectors(ctx))).toEqual(['vectors.E002', 'vectors.E004', 'vectors.E006'])
    } finally {
      restore()
      r2()
    }
  })

  it('reports a missing VECTORS setting clearly', async () => {
    const saved = h.ctx.settings.VECTORS
    delete (h.ctx.settings as any).VECTORS
    try {
      expect(() => vectorStores.get()).toThrow(ImproperlyConfigured)
      expect(() => vectorStores.get()).toThrow(/settings.VECTORS is not set/)
    } finally {
      h.ctx.settings.VECTORS = saved
    }
  })
})

describe('checks without the vectors app', () => {
  it('vectorize() and databaseStore() need the app; memoryStore() warns when DEBUG is off', async () => {
    const h = await setupHarness({
      vectorsApp: false,
      settings: {
        DEBUG: false,
        VECTORS: {
          default: vectorize('VECTORIZE', { embedding: 'workers-ai:@cf/baai/bge-base-en-v1.5' }),
          db: databaseStore({ embedding: 'workers-ai:@cf/baai/bge-base-en-v1.5' }),
          mem: memoryStore({ embedding: 'workers-ai:@cf/baai/bge-base-en-v1.5' }),
          bad: { nope: true } as any,
        },
      },
    })
    try {
      const ctx = { settings: h.ctx.settings, appConfigs: [], deploy: false } as unknown as CheckContext
      // (vectors.E002 is test.orphan, from the describe above.)
      expect(ids(await checkVectors(ctx)).filter((id) => id !== 'vectors.E002')).toEqual([
        'vectors.E001',
        'vectors.E003',
        'vectors.E003',
        'vectors.W001',
        'vectors.W002',
      ])
      const wrangler = checkWrangler('wrangler.jsonc', { vectorize: [] }, new Set(['default', 'mem']), ctx)
      expect(ids(wrangler)).toEqual(['vectors.E005', 'vectors.W003', 'vectors.W003'])
      expect(
        ids(
          checkWrangler(
            'wrangler.jsonc',
            { vectorize: [{ binding: 'VECTORIZE' }], ai: { binding: 'AI' } },
            new Set(['default']),
            ctx,
          ),
        ),
      ).toEqual([])
      await expect(new VectorIndex('test.noapp').index({ key: 'k', text: 'hello' })).rejects.toThrow(
        /needs the 'vectors' app/,
      )
    } finally {
      await h.db.teardown()
    }
  })
})
