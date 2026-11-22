import { bufferIO, callCommand } from '@ghuri/core'
import { describe, expect, it } from 'vitest'
import {
  FilterError,
  getVectorIndex,
  MemoryVectorStore,
  ModelIndex,
  VectorDocument,
  VectorIndex,
  vectorStores,
} from '../src/index'
import { storeCases } from './backends'
import { Post } from './demo'
import { useHarness } from './helpers'

const handbook = new VectorIndex('test.handbook', { chunk: { size: 60, overlap: 0 } })
const posts = new ModelIndex(Post, {
  text: ['title', 'body'],
  metadata: (p) => ({ status: p.status, year: p.year }),
  indexQueryset: (qs) => qs.filter({ status: 'published' }),
  chunk: false,
  background: false,
})

const REFUNDS = 'Refunds are paid within five days. Ask support for a refund of your order.'
const SHIPPING = 'Shipping takes two weeks. Tracking numbers arrive by email.'

async function vectorCount(): Promise<number> {
  const store = vectorStores.get() as any
  if (store instanceof MemoryVectorStore) return store.records.size
  if (store.backend === 'vectorize')
    return (await import('@ghuri/core')).tryGetContext()!.env.VECTORIZE.vectors.size
  if (store.backend === 'database') {
    const { VectorEmbedding } = await import('../src/index')
    return VectorEmbedding.objects.count()
  }
  const { connections } = await import('@ghuri/orm')
  const res = await (await connections.get()).execute(`SELECT count(*)::int AS n FROM "${store.table}"`)
  return Number(res.rows[0].n)
}

for (const c of storeCases()) {
  describe(`indexes: ${c.name}`, () => {
    const h = useHarness(c.harness, async () => {
      const store = vectorStores.get()
      if (store instanceof MemoryVectorStore) store.clear()
      ;(h.env as any).VECTORIZE?.vectors.clear()
      c.embedder.reset()
    })
    it('setup', async () => {
      await c.prepare?.()
      expect(getVectorIndex('test.handbook')).toBe(handbook)
      expect(getVectorIndex('demo.post')).toBe(posts)
    })

    it('indexes documents in chunks and finds the closest ones', async () => {
      const r = await handbook.index([
        { key: 'refunds', text: REFUNDS, metadata: { section: 'billing' } },
        { key: 'shipping', text: SHIPPING, metadata: { section: 'delivery' } },
      ])
      expect(r.indexed).toBe(2)
      expect(r.chunks).toBeGreaterThan(2)
      const hits = await handbook.search('how do I get a refund for my order', { k: 2 })
      expect(hits[0].key).toBe('refunds')
      expect(hits[0].text).toMatch(/refund/i)
      expect(hits[0].metadata).toEqual({ section: 'billing' })
      expect(hits[0].score).toBeGreaterThan(hits[1].score)
      const delivery = await handbook.search('refund', { k: 5, filter: { section: 'delivery' } })
      expect(delivery.every((hit) => hit.key === 'shipping')).toBe(true)
      expect(await handbook.search('   ')).toEqual([])
    })

    it('skips unchanged documents, re-embeds changed ones and drops stale chunks', async () => {
      const long = `${REFUNDS} ${SHIPPING} ${REFUNDS}`
      const first = await handbook.index({ key: 'faq', text: long })
      const before = await vectorCount()
      expect(before).toBe(first.chunks)
      c.embedder.reset()
      const again = await handbook.index({ key: 'faq', text: long })
      expect([again.indexed, again.skipped, c.embedder.calls]).toEqual([0, 1, 0])
      const forced = await handbook.index({ key: 'faq', text: long }, { force: true })
      expect(forced.indexed).toBe(1)
      const shorter = await handbook.index({ key: 'faq', text: 'Refunds take five days.' })
      expect(shorter.chunks).toBe(1)
      expect(await vectorCount()).toBe(1)
      const row = await VectorDocument.objects.get({ index: 'test.handbook', key: 'faq' })
      expect([row.chunks, row.namespace]).toEqual([1, 'test.handbook'])
      expect(await handbook.keys()).toEqual(['faq'])
    })

    it('deletes documents, empty documents and whole indexes', async () => {
      await handbook.index([
        { key: 'refunds', text: REFUNDS },
        { key: 'shipping', text: SHIPPING },
      ])
      await handbook.delete('refunds')
      expect(await handbook.keys()).toEqual(['shipping'])
      expect((await handbook.search('refund order', { k: 10 })).some((x) => x.key === 'refunds')).toBe(false)
      const r = await handbook.index({ key: 'shipping', text: '  ' })
      expect(r.deleted).toBe(1)
      expect(await vectorCount()).toBe(0)
      await handbook.index({ key: 'refunds', text: REFUNDS }, { namespace: 'en' })
      await handbook.index({ key: 'refunds', text: REFUNDS })
      await handbook.clear()
      expect(await handbook.keys()).toEqual([])
      expect(await handbook.keys({ namespace: 'en' })).toEqual(['refunds'])
      expect((await handbook.search('refund', { namespace: 'en' }))[0].key).toBe('refunds')
    })

    it('validates search options and metadata', async () => {
      await expect(handbook.search('x', { filter: { a__icontains: 'x' } })).rejects.toThrow(FilterError)
      await expect(handbook.index({ key: 'k', text: 't', metadata: { _key: 'x' } })).rejects.toThrow(
        /reserved/,
      )
      await expect(handbook.index({ key: 'k', text: 't', metadata: { tags: ['a'] as any } })).rejects.toThrow(
        /string, number or boolean/,
      )
      await handbook.index({ key: 'refunds', text: REFUNDS })
      expect(await handbook.search('refund', { minScore: 0.99 })).toEqual([])
    })

    it('ModelIndex re-indexes on save and delete, honouring indexQueryset', async () => {
      const post = await Post.objects.create({ title: 'Refund policy', body: REFUNDS, year: 2024 })
      let hits = await posts.search('refund', { k: 3 })
      expect(hits.map((x) => x.object.pk)).toEqual([post.pk])
      expect(hits[0].object).toBeInstanceOf(Post)
      expect(hits[0].metadata).toEqual({ status: 'published', year: 2024 })

      const other = await Post.objects.create({ title: 'Shipping', body: SHIPPING, year: 2026 })
      hits = await posts.search('refund shipping', { k: 5, filter: { year__gte: 2025 } })
      expect(hits.map((x) => x.object.pk)).toEqual([other.pk])

      post.status = 'draft'
      await post.save()
      hits = await posts.search('refund', { k: 5 })
      expect(hits.map((x) => x.key)).not.toContain(String(post.pk))

      await other.delete()
      expect(await posts.keys()).toEqual([])
    })

    it('update_index indexes every row, skips unchanged ones and removes stale keys', async () => {
      posts.disconnectSignals()
      try {
        const a = await Post.objects.create({ title: 'Alpha refunds', body: REFUNDS })
        await Post.objects.create({ title: 'Beta shipping', body: SHIPPING })
        await Post.objects.create({ title: 'Gamma draft', status: 'draft' })
        expect(await posts.keys()).toEqual([])
        const io = bufferIO()
        await callCommand('update_index', { index: ['demo.post'] }, io)
        expect(io.out.join('\n')).toMatch(/Indexed 2 Post rows into 'demo.post'/)
        expect(await posts.keys()).toEqual([String(a.pk), String(a.pk + 1)].sort())
        await a.delete()
        const r = await posts.update()
        expect([r.indexed, r.skipped, r.deleted]).toEqual([0, 1, 1])
        const io2 = bufferIO()
        await callCommand('rebuild_index', { index: ['demo.post'] }, io2)
        expect(io2.out.join('\n')).toMatch(/Cleared 'demo.post'[\s\S]*Indexed 1 Post rows/)
        await callCommand('clear_index', {}, bufferIO())
        expect(await posts.keys()).toEqual([])
        await expect(callCommand('update_index', { index: 'nope' }, bufferIO())).rejects.toThrow(
          /Unknown vector index/,
        )
      } finally {
        ;(posts as any).connectSignals()
      }
    })
  })
}
