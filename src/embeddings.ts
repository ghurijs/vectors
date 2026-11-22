// Embedding models behind adapters: `'<provider>:<model id>'` strings (as agent models are), AI SDK embedding models
// or your own `Embedder`. The built-in providers need no package: Workers AI is called through its binding,
// OpenAI-compatible APIs and the ghuri-ai gateway over fetch.
import { captureContext, type Env, setting, tryGetContext } from '@ghuri/core'
import { EmbeddingError } from './errors'
import { globalSingleton } from './global'

/** Turns texts into vectors. */
export interface Embedder {
  /** `'workers-ai:@cf/baai/bge-base-en-v1.5'`; part of every document digest. */
  readonly modelId: string
  /** Length of the vectors, when known before the first call. */
  readonly dimensions: number | undefined
  /** Texts per call. */
  readonly maxBatch: number
  /** One vector per text, in order. `kind` lets models embed queries and documents differently. */
  embed(texts: string[], opts?: { kind?: 'query' | 'document' }): Promise<number[][]>
}

/** The parts of an AI SDK embedding model (`EmbeddingModelV2`-`V4`) Ghuri calls, so `ai` is not a dependency. */
export interface AISDKEmbeddingModel {
  readonly specificationVersion: string
  readonly provider: string
  readonly modelId: string
  readonly maxEmbeddingsPerCall?: PromiseLike<number | undefined> | number | undefined
  doEmbed(options: { values: string[]; abortSignal?: AbortSignal }): PromiseLike<{ embeddings: number[][] }>
}

/** What a store's `embedding` option accepts. */
export type EmbeddingSpec = string | AISDKEmbeddingModel | Embedder

export interface EmbeddingProviderContext {
  /** Worker bindings / environment variables of the current request. */
  env: Env
  /** Provider options: the store's `providerOptions`, else `AGENTS.PROVIDERS[name]` (empty object when unset). */
  config: Record<string, unknown>
  /** Name of the Workers AI binding (`AGENTS.AI_BINDING`, default `'AI'`). */
  aiBinding: string
  /** Dimensions the store asks for (OpenAI's `text-embedding-3-*` can shorten vectors). */
  dimensions?: number
}

/** Builds an embedder for `'<provider>:<model id>'`. */
export type EmbeddingProvider = (
  modelId: string,
  ctx: EmbeddingProviderContext,
) => Embedder | Promise<Embedder>

const providers: Map<string, EmbeddingProvider> = globalSingleton(
  'ghuri.vectors.embeddingProviders',
  () => new Map(),
)

/** Register (or replace) an embedding provider for `'name:model-id'` strings. */
export function registerEmbeddingProvider(name: string, factory: EmbeddingProvider): void {
  providers.set(name, factory)
}

export function unregisterEmbeddingProvider(name: string): boolean {
  return providers.delete(name)
}

/** Names of the registered embedding providers. */
export function getEmbeddingProviders(): string[] {
  return [...providers.keys()]
}

/** Dimensions of well-known embedding models, so stores know them before the first call. */
export const KNOWN_DIMENSIONS: Record<string, number> = {
  '@cf/baai/bge-small-en-v1.5': 384,
  '@cf/baai/bge-base-en-v1.5': 768,
  '@cf/baai/bge-large-en-v1.5': 1024,
  '@cf/baai/bge-m3': 1024,
  '@cf/google/embeddinggemma-300m': 768,
  '@cf/qwen/qwen3-embedding-0.6b': 1024,
  'text-embedding-3-small': 1536,
  'text-embedding-3-large': 3072,
  'text-embedding-ada-002': 1536,
}

/** Known dimensions of a model id (`'workers-ai:@cf/baai/bge-base-en-v1.5'`, `'openai/text-embedding-3-small'`). */
export function knownDimensions(modelId: string): number | undefined {
  let id = modelId.slice(modelId.indexOf(':') + 1)
  if (id in KNOWN_DIMENSIONS) return KNOWN_DIMENSIONS[id]
  id = id.slice(id.lastIndexOf('/') + 1)
  return KNOWN_DIMENSIONS[id]
}

export function isEmbedder(value: unknown): value is Embedder {
  return (
    !!value &&
    typeof value === 'object' &&
    typeof (value as Embedder).embed === 'function' &&
    typeof (value as Embedder).modelId === 'string'
  )
}

export function isAISDKEmbeddingModel(value: unknown): value is AISDKEmbeddingModel {
  return (
    !!value &&
    typeof value === 'object' &&
    typeof (value as AISDKEmbeddingModel).doEmbed === 'function' &&
    typeof (value as AISDKEmbeddingModel).modelId === 'string'
  )
}

/** Split `texts` into batches of `size` and embed them one batch after another. */
export async function embedInBatches(
  texts: string[],
  size: number,
  fn: (batch: string[]) => Promise<number[][]>,
): Promise<number[][]> {
  const out: number[][] = []
  for (let i = 0; i < texts.length; i += size) {
    const batch = texts.slice(i, i + size)
    const vectors = await fn(batch)
    if (!Array.isArray(vectors) || vectors.length !== batch.length) {
      throw new EmbeddingError(
        `The embedding model returned ${vectors?.length ?? 0} vectors for ${batch.length} texts.`,
      )
    }
    out.push(...vectors.map((v) => Array.from(v)))
  }
  return out
}

function fromAISDK(model: AISDKEmbeddingModel): Embedder {
  const modelId = `${model.provider}:${model.modelId}`
  return {
    modelId,
    dimensions: knownDimensions(model.modelId),
    maxBatch: 100,
    async embed(texts) {
      const max = (await model.maxEmbeddingsPerCall) ?? 100
      return embedInBatches(texts, Math.max(1, max), async (values) => {
        try {
          return (await model.doEmbed({ values })).embeddings
        } catch (err) {
          throw new EmbeddingError(`Embedding with '${modelId}' failed: ${(err as Error).message}`, {
            cause: err,
          })
        }
      })
    },
  }
}

function processEnv(name: string): string | undefined {
  const v = (globalThis as any).process?.env?.[name]
  return typeof v === 'string' && v ? v : undefined
}

function configString(ctx: EmbeddingProviderContext, key: string, envName?: string): string | undefined {
  const v = ctx.config[key] ?? (envName ? (ctx.env?.[envName] ?? processEnv(envName)) : undefined)
  return typeof v === 'string' && v ? v : undefined
}

/** Settings shared with `@ghuri/agents` (read only; vectors does not import agents). */
function agentsSetting(): { AI_BINDING?: string; PROVIDERS?: Record<string, Record<string, unknown>> } {
  try {
    return (setting('AGENTS', {}) as any) ?? {}
  } catch {
    return {}
  }
}

/** Options for resolving a spec. */
export interface ResolveEmbeddingOptions {
  env?: Env
  /** Options for the provider (API key, base URL…); default `AGENTS.PROVIDERS[name]`. */
  providerOptions?: Record<string, unknown>
  dimensions?: number
}

/**
 * Resolve an embedding spec for the current request. Strings go through the provider registry (`workers-ai:`,
 * `openai:` and `ghuri-ai:` are built in); AI SDK embedding models are called through `doEmbed`.
 */
export async function resolveEmbedding(
  spec: EmbeddingSpec,
  opts: ResolveEmbeddingOptions = {},
): Promise<Embedder> {
  if (isEmbedder(spec)) return spec
  if (isAISDKEmbeddingModel(spec)) return fromAISDK(spec)
  if (typeof spec !== 'string') throw new EmbeddingError(`Not an embedding model: ${String(spec)}`)
  const i = spec.indexOf(':')
  if (i <= 0) {
    throw new EmbeddingError(
      `Embedding model '${spec}' must be '<provider>:<model-id>' (providers: ${getEmbeddingProviders().join(', ')}).`,
    )
  }
  const name = spec.slice(0, i)
  const factory = providers.get(name)
  if (!factory) {
    throw new EmbeddingError(
      `Unknown embedding provider '${name}' in '${spec}' (registered: ${getEmbeddingProviders().join(', ')}). Use registerEmbeddingProvider().`,
    )
  }
  const agents = agentsSetting()
  const ctx: EmbeddingProviderContext = {
    env: opts.env ?? tryGetContext()?.env ?? {},
    config: { ...(opts.providerOptions ?? agents.PROVIDERS?.[name] ?? {}) },
    aiBinding: agents.AI_BINDING ?? 'AI',
    ...(opts.dimensions ? { dimensions: opts.dimensions } : {}),
  }
  return factory(spec.slice(i + 1), ctx)
}

/**
 * An embedder that resolves `spec` on every call, so Workers bindings and keys come from the current request. Stores
 * keep one of these.
 */
export function lazyEmbedder(spec: EmbeddingSpec, opts: Omit<ResolveEmbeddingOptions, 'env'> = {}): Embedder {
  if (isEmbedder(spec)) return spec
  const modelId = isAISDKEmbeddingModel(spec) ? `${spec.provider}:${spec.modelId}` : String(spec)
  return {
    modelId,
    dimensions: opts.dimensions ?? knownDimensions(isAISDKEmbeddingModel(spec) ? spec.modelId : modelId),
    maxBatch: 100,
    async embed(texts, embedOpts) {
      if (!texts.length) return []
      return (await resolveEmbedding(spec, opts)).embed(texts, embedOpts)
    },
  }
}

// ---- built-in providers ----------------------------------------------------------------------------------------

registerEmbeddingProvider('workers-ai', (modelId, ctx) => {
  const binding = ctx.env?.[ctx.aiBinding] as { run(model: string, input: unknown): Promise<any> } | undefined
  return {
    modelId: `workers-ai:${modelId}`,
    dimensions: ctx.dimensions ?? knownDimensions(modelId),
    maxBatch: 100,
    async embed(texts) {
      if (!binding || typeof binding.run !== 'function') {
        throw new EmbeddingError(
          `Workers AI binding '${ctx.aiBinding}' not found in env. Add "ai": { "binding": "${ctx.aiBinding}" } to wrangler.jsonc or set AGENTS.AI_BINDING.`,
        )
      }
      return embedInBatches(texts, 100, async (text) => {
        let res: any
        try {
          res = await binding.run(modelId, { text })
        } catch (err) {
          throw new EmbeddingError(`Workers AI embedding '${modelId}' failed: ${(err as Error).message}`, {
            cause: err,
          })
        }
        const data = res?.data ?? res?.response
        if (!Array.isArray(data))
          throw new EmbeddingError(`Workers AI model '${modelId}' returned no embeddings.`)
        return data
      })
    },
  }
})

/** An embedder for an OpenAI-compatible `POST <baseURL>/embeddings` API. */
export function openAICompatibleEmbedder(opts: {
  modelId: string
  model: string
  baseURL: string
  apiKey?: string
  headers?: Record<string, string>
  dimensions?: number
  /** Send `dimensions` in the request (OpenAI `text-embedding-3-*`). */
  sendDimensions?: boolean
  maxBatch?: number
}): Embedder {
  const url = `${opts.baseURL.replace(/\/+$/, '')}/embeddings`
  const maxBatch = opts.maxBatch ?? 512
  return {
    modelId: opts.modelId,
    dimensions: opts.dimensions,
    maxBatch,
    async embed(texts) {
      return embedInBatches(texts, maxBatch, async (input) => {
        const body: Record<string, unknown> = { model: opts.model, input, encoding_format: 'float' }
        if (opts.sendDimensions && opts.dimensions) body.dimensions = opts.dimensions
        let res: Response
        try {
          res = await fetch(url, {
            method: 'POST',
            headers: {
              'content-type': 'application/json',
              ...(opts.apiKey ? { authorization: `Bearer ${opts.apiKey}` } : {}),
              ...opts.headers,
            },
            body: JSON.stringify(body),
          })
        } catch (err) {
          throw new EmbeddingError(`Embedding request to ${url} failed: ${(err as Error).message}`, {
            cause: err,
          })
        }
        if (!res.ok) {
          const detail = (await res.text().catch(() => '')).slice(0, 300)
          throw new EmbeddingError(`Embedding with '${opts.modelId}' failed: HTTP ${res.status} ${detail}`)
        }
        const json = (await res.json()) as { data?: Array<{ index: number; embedding: number[] }> }
        const data = [...(json.data ?? [])].sort((a, b) => a.index - b.index)
        return data.map((d) => d.embedding)
      })
    },
  }
}

function stringHeaders(value: unknown): Record<string, string> {
  if (!value || typeof value !== 'object') return {}
  const out: Record<string, string> = {}
  for (const [k, v] of Object.entries(value)) if (typeof v === 'string') out[k] = v
  return out
}

registerEmbeddingProvider('openai', (modelId, ctx) => {
  const known = knownDimensions(modelId)
  return openAICompatibleEmbedder({
    modelId: `openai:${modelId}`,
    model: modelId,
    baseURL: configString(ctx, 'baseURL', 'OPENAI_BASE_URL') ?? 'https://api.openai.com/v1',
    apiKey: configString(ctx, 'apiKey', 'OPENAI_API_KEY'),
    headers: stringHeaders(ctx.config.headers),
    dimensions: ctx.dimensions ?? known,
    sendDimensions: !!ctx.dimensions && ctx.dimensions !== known,
  })
})

/** `x-ghuri-<carrier>` headers for the carried context (`x-ghuri-tenant`), as agents send to ghuri-ai. */
export function carriedHeaders(): Record<string, string> {
  const out: Record<string, string> = { 'x-ghuri-feature': 'vectors' }
  for (const [name, value] of Object.entries(captureContext())) {
    if (/^[a-z0-9-]+$/i.test(name)) out[`x-ghuri-${name.toLowerCase()}`] = value
  }
  return out
}

registerEmbeddingProvider('ghuri-ai', (modelId, ctx) => {
  const base = configString(ctx, 'baseURL', 'GHURI_AI_URL')?.replace(/\/+$/, '')
  if (!base) {
    throw new EmbeddingError(
      "The 'ghuri-ai:' embedding provider needs the gateway URL: set AGENTS.PROVIDERS['ghuri-ai'].baseURL or GHURI_AI_URL.",
    )
  }
  const known = knownDimensions(modelId)
  return openAICompatibleEmbedder({
    modelId: `ghuri-ai:${modelId}`,
    model: modelId,
    baseURL: `${base}/v1`,
    apiKey: configString(ctx, 'apiKey', 'GHURI_AI_API_KEY'),
    headers: { ...carriedHeaders(), ...stringHeaders(ctx.config.headers) },
    dimensions: ctx.dimensions ?? known,
    sendDimensions: !!ctx.dimensions && ctx.dimensions !== known,
  })
})
