import { AsyncLocalStorage } from 'node:async_hooks'
import { createContext, registerContextCarrier, runWithContext, unregisterContextCarrier } from '@ghuri/core'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  EmbeddingError,
  getEmbeddingProviders,
  knownDimensions,
  lazyEmbedder,
  registerEmbeddingProvider,
  resolveEmbedding,
  unregisterEmbeddingProvider,
} from '../src/index'

function context(settings: Record<string, unknown> = {}, env: Record<string, unknown> = {}) {
  return createContext({
    env,
    settings: { SECRET_KEY: 'x', DEBUG: true, INSTALLED_APPS: [], ROOT_URLCONF: [], ...settings } as any,
  })
}

function fetchStub(dimensions = 3) {
  const calls: Array<{ url: string; init: RequestInit; body: any }> = []
  const stub = vi.fn(async (url: string, init: RequestInit) => {
    const body = JSON.parse(String(init.body))
    calls.push({ url, init, body })
    const data = (body.input as string[]).map((_, i) => ({
      index: i,
      embedding: new Array(dimensions).fill(i + 1),
    }))
    // Out of order on purpose: the client sorts by index.
    return new Response(JSON.stringify({ data: data.reverse() }), {
      headers: { 'content-type': 'application/json' },
    })
  })
  vi.stubGlobal('fetch', stub)
  return calls
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('embedding providers', () => {
  it('has the built-in providers and known dimensions', () => {
    expect(getEmbeddingProviders()).toEqual(expect.arrayContaining(['workers-ai', 'openai', 'ghuri-ai']))
    expect(knownDimensions('workers-ai:@cf/baai/bge-base-en-v1.5')).toBe(768)
    expect(knownDimensions('openai:text-embedding-3-small')).toBe(1536)
    expect(knownDimensions('ghuri-ai:openai/text-embedding-3-large')).toBe(3072)
    expect(knownDimensions('openai:unknown')).toBeUndefined()
  })

  it('workers-ai: calls the AI binding (AGENTS.AI_BINDING) in batches of 100', async () => {
    const run = vi.fn(async (_model: string, input: { text: string[] }) => ({
      shape: [input.text.length, 2],
      data: input.text.map((t) => [t.length, 1]),
    }))
    const ctx = context({ AGENTS: { AI_BINDING: 'MYAI' } }, { MYAI: { run } })
    const vectors = await runWithContext(ctx, async () => {
      const e = await resolveEmbedding('workers-ai:@cf/baai/bge-small-en-v1.5')
      expect(e.modelId).toBe('workers-ai:@cf/baai/bge-small-en-v1.5')
      expect(e.dimensions).toBe(384)
      return e.embed(Array.from({ length: 150 }, (_, i) => 'x'.repeat(i)))
    })
    expect(vectors).toHaveLength(150)
    expect(vectors[149]).toEqual([149, 1])
    expect(run).toHaveBeenCalledTimes(2)
    expect(run.mock.calls[0][0]).toBe('@cf/baai/bge-small-en-v1.5')
    await expect(
      runWithContext(context(), async () => (await resolveEmbedding('workers-ai:@cf/x')).embed(['a'])),
    ).rejects.toThrow(/Workers AI binding 'AI' not found/)
  })

  it('openai: posts to /embeddings with the key from AGENTS.PROVIDERS, sorted by index', async () => {
    const calls = fetchStub(3)
    const ctx = context({
      AGENTS: { PROVIDERS: { openai: { apiKey: 'sk-test', baseURL: 'https://llm.test/v1/' } } },
    })
    const vectors = await runWithContext(ctx, async () =>
      (await resolveEmbedding('openai:text-embedding-3-small', { dimensions: 256 })).embed(['a', 'b']),
    )
    expect(vectors).toEqual([
      [1, 1, 1],
      [2, 2, 2],
    ])
    expect(calls[0].url).toBe('https://llm.test/v1/embeddings')
    expect((calls[0].init.headers as Record<string, string>).authorization).toBe('Bearer sk-test')
    expect(calls[0].body).toEqual({
      model: 'text-embedding-3-small',
      input: ['a', 'b'],
      encoding_format: 'float',
      dimensions: 256,
    })
  })

  it('openai: store providerOptions win over AGENTS.PROVIDERS; errors become EmbeddingError', async () => {
    vi.stubGlobal('fetch', async () => new Response('quota exceeded', { status: 429 }))
    const ctx = context({ AGENTS: { PROVIDERS: { openai: { apiKey: 'agents' } } } })
    await expect(
      runWithContext(ctx, async () =>
        (
          await resolveEmbedding('openai:text-embedding-3-small', { providerOptions: { apiKey: 'store' } })
        ).embed(['a']),
      ),
    ).rejects.toThrow(/HTTP 429 quota exceeded/)
  })

  it('ghuri-ai: uses the gateway URL and sends x-ghuri headers for the carried context', async () => {
    const calls = fetchStub(2)
    const tenant = new AsyncLocalStorage<string>()
    registerContextCarrier({
      name: 'tenant',
      capture: () => tenant.getStore(),
      restore: (v, fn) => tenant.run(v, fn),
    })
    try {
      const ctx = context({}, { GHURI_AI_URL: 'https://gw.test/', GHURI_AI_API_KEY: 'gk' })
      await runWithContext(ctx, () =>
        tenant.run('42', async () =>
          (await resolveEmbedding('ghuri-ai:openai/text-embedding-3-small')).embed(['q']),
        ),
      )
      expect(calls[0].url).toBe('https://gw.test/v1/embeddings')
      expect(calls[0].body.model).toBe('openai/text-embedding-3-small')
      const headers = calls[0].init.headers as Record<string, string>
      expect(headers['x-ghuri-tenant']).toBe('42')
      expect(headers['x-ghuri-feature']).toBe('vectors')
      expect(headers.authorization).toBe('Bearer gk')
      await expect(runWithContext(context(), () => resolveEmbedding('ghuri-ai:x'))).rejects.toThrow(
        /needs the gateway URL/,
      )
    } finally {
      unregisterContextCarrier('tenant')
    }
  })

  it('AI SDK embedding models are called through doEmbed, honouring maxEmbeddingsPerCall', async () => {
    const doEmbed = vi.fn(async ({ values }: { values: string[] }) => ({
      embeddings: values.map(() => [0.5, 0.5]),
    }))
    const model = {
      specificationVersion: 'v3',
      provider: 'google',
      modelId: 'text-embedding-004',
      maxEmbeddingsPerCall: Promise.resolve(2),
      doEmbed,
    }
    const e = lazyEmbedder(model)
    expect(e.modelId).toBe('google:text-embedding-004')
    const vectors = await runWithContext(context(), () => e.embed(['a', 'b', 'c']))
    expect(vectors).toHaveLength(3)
    expect(doEmbed).toHaveBeenCalledTimes(2)
  })

  it('registerEmbeddingProvider adds providers; bad specs raise EmbeddingError', async () => {
    registerEmbeddingProvider('const', (modelId) => ({
      modelId: `const:${modelId}`,
      dimensions: 1,
      maxBatch: 10,
      embed: async (texts) => texts.map(() => [Number(modelId)]),
    }))
    try {
      expect(
        await runWithContext(context(), async () => (await resolveEmbedding('const:7')).embed(['x'])),
      ).toEqual([[7]])
    } finally {
      unregisterEmbeddingProvider('const')
    }
    await expect(resolveEmbedding('nope')).rejects.toThrow(EmbeddingError)
    await expect(runWithContext(context(), () => resolveEmbedding('unknown:x'))).rejects.toThrow(
      /Unknown embedding provider/,
    )
  })
})
