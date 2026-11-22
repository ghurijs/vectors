import { describe, expect, it } from 'vitest'
import {
  chunkText,
  decodeVector,
  encodeVector,
  FilterError,
  filterSql,
  matches,
  parseFilter,
  resolveNamespace,
  score,
  toVectorizeFilter,
  vectorId,
} from '../src/index'
import { stableStringify } from '../src/scope'
import { fakeEmbedding } from '../src/testing'

describe('chunkText', () => {
  it('keeps short texts whole and drops blank ones', () => {
    expect(chunkText('Hello world.')).toEqual(['Hello world.'])
    expect(chunkText('  \n\n ')).toEqual([])
  })

  it('splits on paragraphs, then sentences, within the size, with overlap', () => {
    const para = (n: number) => `Paragraph ${n} has a sentence. And another one here.`
    const text = [1, 2, 3, 4, 5, 6].map(para).join('\n\n')
    const chunks = chunkText(text, { size: 120, overlap: 60 })
    expect(chunks.length).toBeGreaterThan(2)
    for (const c of chunks) expect(c.length).toBeLessThanOrEqual(120)
    // Every paragraph survives somewhere, and consecutive chunks share text.
    for (let n = 1; n <= 6; n++) expect(chunks.some((c) => c.includes(`Paragraph ${n}`))).toBe(true)
    expect(chunks[1].includes(chunks[0].split('\n\n').pop()!.slice(0, 10))).toBe(true)
  })

  it('falls back to characters for long words, and validates options', () => {
    const chunks = chunkText('x'.repeat(250), { size: 100, overlap: 0 })
    expect(chunks.map((c) => c.length)).toEqual([100, 100, 50])
    expect(() => chunkText('abc', { size: 0 })).toThrow(RangeError)
  })
})

describe('filters', () => {
  it('parses Django lookups and exclude', () => {
    expect(
      parseFilter({ status: 'published', year__gte: 2025, lang__in: ['en', 'bn'] }, { kind: 'draft' }),
    ).toEqual([
      { key: 'status', op: '$eq', value: 'published' },
      { key: 'year', op: '$gte', value: 2025 },
      { key: 'lang', op: '$in', value: ['en', 'bn'] },
      { key: 'kind', op: '$ne', value: 'draft' },
    ])
    expect(parseFilter({ title__exact: 'x' })).toEqual([{ key: 'title', op: '$eq', value: 'x' }])
  })

  it('rejects unknown lookups, bad values and range excludes', () => {
    expect(() => parseFilter({ title__icontains: 'x' })).toThrow(FilterError)
    expect(() => parseFilter({ tags__in: [] })).toThrow(FilterError)
    expect(() => parseFilter({ year__gt: true as any })).toThrow(FilterError)
    expect(() => parseFilter({ x: null as any })).toThrow(FilterError)
    expect(() => parseFilter({}, { year__gt: 3 })).toThrow(FilterError)
    expect(() => parseFilter({ 'a.b': 1 })).toThrow(FilterError)
  })

  it('matches metadata in JS with Vectorize semantics', () => {
    const md = { status: 'published', year: 2025, featured: true }
    expect(matches(parseFilter({ status: 'published', year__gte: 2025 }), md)).toBe(true)
    expect(matches(parseFilter({ year__gt: 2025 }), md)).toBe(false)
    expect(matches(parseFilter({ year: '2025' }), md)).toBe(false)
    expect(matches(parseFilter({ featured: true }), md)).toBe(true)
    expect(matches(parseFilter({}, { status: 'draft' }), md)).toBe(true)
    expect(matches(parseFilter({}, { missing: 'x' }), md)).toBe(true)
    expect(matches(parseFilter({ missing: 'x' }), md)).toBe(false)
    expect(matches(parseFilter({ year__lt: 'z' }), md)).toBe(false)
  })

  it('translates to Vectorize filters and pgvector SQL', () => {
    const conds = parseFilter({ year__gte: 2020, year__lt: 2026, status: 'published' })
    expect(toVectorizeFilter(conds)).toEqual({
      year: { $gte: 2020, $lt: 2026 },
      status: { $eq: 'published' },
    })
    const params: unknown[] = []
    const sql = filterSql(parseFilter({ status: 'published', year__gte: 2020, lang__in: ['en'] }), params)
    expect(sql[0]).toBe(`"metadata" -> $1::text = $2::jsonb`)
    expect(sql[1]).toContain('::float8 END) >= $5::float8')
    expect(sql[2]).toBe(`"metadata" -> $6::text IN ($7::jsonb)`)
    expect(params).toEqual(['status', '"published"', 'year', 'year', 2020, 'lang', '"en"'])
  })
})

describe('math and ids', () => {
  it('scores like Vectorize', () => {
    expect(score('cosine', [1, 0], [1, 0])).toBeCloseTo(1)
    expect(score('cosine', [1, 0], [0, 1])).toBeCloseTo(0)
    expect(score('euclidean', [0, 0], [3, 4])).toBeCloseTo(5)
    expect(score('dot-product', [1, 2], [3, 4])).toBe(11)
    expect(() => score('cosine', [1], [1, 2])).toThrow(/dimensions/)
  })

  it('round-trips vectors through base64 float32', () => {
    const v = [0.5, -1.25, 3, 0.1]
    const back = decodeVector(encodeVector(v))
    for (const [i, x] of back.entries()) expect(x).toBeCloseTo(v[i], 6)
  })

  it('builds namespaces and ids within 64 bytes', async () => {
    expect(await resolveNamespace('blog.posts', false)).toBe('blog.posts')
    expect(await resolveNamespace('blog.posts', false, 'drafts')).toBe('blog.posts|drafts')
    const long = await resolveNamespace('blog.posts', false, 'x'.repeat(80))
    expect(long).toMatch(/^h:[0-9a-f]{62}$/)
    await expect(resolveNamespace('a', false, 'a|b')).rejects.toThrow(/reserved/)
    const id = await vectorId('blog.posts', '42', 3)
    expect(id).toMatch(/^[0-9a-f]{40}-3$/)
    expect(await vectorId('blog.posts|t=1', '42', 3)).not.toBe(id)
    expect(stableStringify({ b: 1, a: [2, { d: 1, c: 2 }] })).toBe('{"a":[2,{"c":2,"d":1}],"b":1}')
  })

  it('fakeEmbedding puts texts that share words close together', async () => {
    const e = fakeEmbedding({ dimensions: 64 })
    const [a, b, c] = await e.embed([
      'refund policy for orders',
      'how do refunds of orders work',
      'gardening tips',
    ])
    expect(score('cosine', a, b)).toBeGreaterThan(score('cosine', a, c))
    expect(e.calls).toBe(1)
    expect(e.texts).toHaveLength(3)
  })
})
