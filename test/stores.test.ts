import { describe, expect, it } from 'vitest'
import {
  DimensionMismatch,
  MemoryVectorStore,
  parseFilter,
  type VectorStore,
  vectorStores,
} from '../src/index'
import { storeCases } from './backends'
import { useHarness } from './helpers'

function unit(dims: number, i: number, j?: number): number[] {
  const v = new Array(dims).fill(0)
  v[i] = 1
  if (j !== undefined) v[j] = 0.5
  return v
}

for (const c of storeCases()) {
  describe(`store contract: ${c.name}`, () => {
    let store: VectorStore
    const h = useHarness(c.harness, async () => {
      store = vectorStores.get()
      if (store instanceof MemoryVectorStore) store.clear()
      const env = h.env as any
      env.VECTORIZE?.vectors.clear()
    })
    it('setup', async () => {
      await c.prepare?.()
      expect(h.db).toBeTruthy()
    })

    it('upserts, queries best first within a namespace, and filters metadata', async () => {
      await store.upsert([
        { id: 'a1', values: unit(64, 0), namespace: 'a', metadata: { _key: 'd1', year: 2024, kind: 'faq' } },
        {
          id: 'a2',
          values: unit(64, 0, 1),
          namespace: 'a',
          metadata: { _key: 'd1', year: 2025, kind: 'doc' },
        },
        { id: 'a3', values: unit(64, 2), namespace: 'a', metadata: { _key: 'd2', year: 2026, kind: 'doc' } },
        { id: 'b1', values: unit(64, 0), namespace: 'b', metadata: { _key: 'd1', year: 2025, kind: 'doc' } },
      ])
      const top = await store.query(unit(64, 0), { topK: 2, namespace: 'a' })
      expect(top.map((m) => m.id)).toEqual(['a1', 'a2'])
      expect(top[0].score).toBeCloseTo(1, 5)
      expect(top[0].metadata).toMatchObject({ _key: 'd1', year: 2024, kind: 'faq' })

      const all = await store.query(unit(64, 0), { topK: 10, namespace: 'a' })
      expect(all.map((m) => m.id).sort()).toEqual(['a1', 'a2', 'a3'])

      const filtered = await store.query(unit(64, 0), {
        topK: 10,
        namespace: 'a',
        filter: parseFilter({ kind: 'doc', year__gte: 2025 }),
      })
      expect(filtered.map((m) => m.id)).toEqual(['a2', 'a3'])
      const notFaq = await store.query(unit(64, 0), {
        topK: 10,
        namespace: 'a',
        filter: parseFilter({ year__in: [2024, 2026] }, { kind: 'faq' }),
      })
      expect(notFaq.map((m) => m.id)).toEqual(['a3'])

      const withValues = await store.query(unit(64, 2), { topK: 1, namespace: 'a', returnValues: true })
      expect(withValues[0].values?.[2]).toBeCloseTo(1)
    })

    it('replaces vectors with the same id and reads them back by id', async () => {
      await store.upsert([{ id: 'x', values: unit(64, 0), namespace: 'n', metadata: { v: 1 } }])
      await store.upsert([{ id: 'x', values: unit(64, 3), namespace: 'n', metadata: { v: 2 } }])
      const [r] = await store.getByIds(['x', 'missing'])
      expect(r.id).toBe('x')
      expect(r.metadata).toEqual({ v: 2 })
      expect(r.values[3]).toBeCloseTo(1)
      await store.deleteByIds(['x'])
      expect(await store.getByIds(['x'])).toEqual([])
    })

    it.runIf(c.native)('deletes documents and namespaces natively', async () => {
      await store.upsert([
        { id: '1', values: unit(64, 0), namespace: 'n', metadata: { _key: 'k1' } },
        { id: '2', values: unit(64, 1), namespace: 'n', metadata: { _key: 'k2' } },
        { id: '3', values: unit(64, 1), namespace: 'm', metadata: { _key: 'k1' } },
      ])
      await store.deleteDocuments!('n', ['k1'])
      expect((await store.getByIds(['1', '2', '3'])).map((r) => r.id)).toEqual(['2', '3'])
      await store.clearNamespace!('n')
      expect((await store.getByIds(['2', '3'])).map((r) => r.id)).toEqual(['3'])
    })

    it('rejects vectors of the wrong length', async () => {
      await expect(store.upsert([{ id: 'bad', values: [1, 2, 3] }])).rejects.toThrow(DimensionMismatch)
      await expect(store.query([1, 2], { topK: 1 })).rejects.toThrow(DimensionMismatch)
    })
  })
}
