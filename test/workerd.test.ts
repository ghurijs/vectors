// Bundles the main entry for Workers (esbuild, workerd conditions) and runs indexing and search inside workerd
// (Miniflare), with a fake Workers AI binding for the `workers-ai:` embedding provider.
import { join } from 'node:path'
import { build } from 'esbuild'
import { Miniflare } from 'miniflare'
import { afterAll, beforeAll, describe, expect, it } from 'vitest'

const WORKER = `
import { createContext, runWithContext } from '@ghuri/core'
import { chunkText, memoryStore, VectorIndex } from './src/index'

const index = new VectorIndex('workers.docs', { chunk: { size: 60, overlap: 0 }, tenant: false })
const settings = {
  SECRET_KEY: 'workers-secret', INSTALLED_APPS: [], ROOT_URLCONF: [],
  VECTORS: { default: memoryStore({ embedding: 'workers-ai:@cf/baai/bge-small-en-v1.5' }) },
}

function embedWords(text) {
  const v = new Array(384).fill(0)
  for (const w of text.toLowerCase().match(/[a-z]+/g) ?? []) {
    let h = 7
    for (let i = 0; i < w.length; i++) h = (h * 31 + w.charCodeAt(i)) >>> 0
    v[h % 384] += 1
  }
  return v
}

export default {
  async fetch(_req, env) {
    const calls = []
    const AI = { async run(model, input) { calls.push(model); return { shape: [input.text.length, 384], data: input.text.map(embedWords) } } }
    return runWithContext(createContext({ env: { ...env, AI }, settings }), async () => {
      const indexed = await index.index([
        { key: 'refunds', text: 'Refunds are paid within five days. Contact support for refunds.', metadata: { section: 'billing' } },
        { key: 'shipping', text: 'Shipping takes two weeks. Tracking arrives by email.', metadata: { section: 'delivery' } },
      ])
      const hits = await index.search('when are refunds paid', { k: 2 })
      const filtered = await index.search('refunds', { k: 5, filter: { section: 'delivery' } })
      return Response.json({
        userAgent: navigator.userAgent,
        chunks: chunkText('a b c', { size: 2, overlap: 0 }),
        indexed,
        top: hits[0].key,
        text: hits[0].text,
        filtered: filtered.map((h) => h.key),
        calls,
      })
    })
  },
}
`

describe('inside workerd (Cloudflare Workers runtime)', () => {
  let mf: Miniflare
  let result: Record<string, any>
  let bundle: string

  beforeAll(async () => {
    const root = join(import.meta.dirname, '..')
    const out = await build({
      stdin: { contents: WORKER, resolveDir: root, loader: 'ts', sourcefile: 'worker.ts' },
      bundle: true,
      format: 'esm',
      platform: 'browser',
      conditions: ['workerd', 'worker', 'browser'],
      // postgres.js is imported lazily by the ORM and never used here
      external: ['node:*', 'cloudflare:*', 'postgres'],
      write: false,
      logLevel: 'silent',
    })
    bundle = out.outputFiles[0].text
    // Modules listed explicitly (as wrangler does): the bundle keeps the computed `import()` of node:fs that
    // `ghuri check` uses on Node, which Miniflare's dependency scan of a plain `script` refuses.
    const options = {
      modules: [{ type: 'ESModule', path: 'worker.js', contents: bundle }],
      compatibilityDate: '2025-06-01',
      compatibilityFlags: ['nodejs_compat'],
    }
    const mod: any = await import('miniflare')
    mf = new Miniflare(
      typeof mod.convertV4MiniflareOptions === 'function'
        ? mod.convertV4MiniflareOptions(options)
        : (options as any),
    )
    const res = await mf.dispatchFetch('http://localhost/')
    const body = await res.text()
    try {
      result = JSON.parse(body)
    } catch {
      throw new Error(`Worker failed: ${body.slice(0, 500)}`)
    }
  }, 120_000)

  afterAll(async () => {
    await mf?.dispose()
  })

  it('bundles without @ghuri/tasks, @ghuri/agents or Node-only database drivers', () => {
    // esbuild marks each bundled module with a `// <path>` comment; messages may name the packages, imports would not.
    const modules = bundle.match(/^\/\/ \S+$/gm) ?? []
    expect(modules.some((m) => m.includes('src/vector-index.ts'))).toBe(true)
    expect(
      modules.filter((m) =>
        /better-sqlite3|@electric-sql|\.\.\/(tasks|agents|tenants)\/|@ghuri\/(tasks|agents)/.test(m),
      ),
    ).toEqual([])
  })

  it('indexes and searches on the Workers runtime with the workers-ai provider', () => {
    expect(result.userAgent).toBe('Cloudflare-Workers')
    expect(result.chunks).toEqual(['a', 'b', 'c'])
    expect(result.indexed).toMatchObject({ indexed: 2, skipped: 0 })
    expect(result.top).toBe('refunds')
    expect(result.text).toMatch(/Refunds/)
    expect(result.filtered).toEqual(['shipping'])
    expect(new Set(result.calls)).toEqual(new Set(['@cf/baai/bge-small-en-v1.5']))
  })
})
