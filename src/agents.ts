// @ghuri/vectors/agents — retrieval for @ghuri/agents: a search tool and a context formatter. `@ghuri/agents` is an
// optional peer and only its types are used here; tools are plain `AgentTool` objects.
import type { AgentTool, ToolContext } from '@ghuri/agents'
import type { Filter } from './filters'
import type { SearchHit, SearchOptions, VectorIndex } from './vector-index'

export interface SearchToolOptions {
  /** Tool name (default `search_<index name>`, sanitized). */
  name?: string
  /** What the model reads to decide when to search (default names the index). */
  description?: string
  /** Chunks returned (default 5); the model may ask for up to `maxK`. */
  k?: number
  /** Largest `k` the model may ask for (default 10). */
  maxK?: number
  /** Metadata filter for every search, or a function of the tool context (per-user scoping). */
  filter?: Filter | ((ctx: ToolContext) => Filter | undefined | Promise<Filter | undefined>)
  exclude?: Filter
  minScore?: number
  /** Characters of text per hit (default 2000). */
  maxChars?: number
}

function sanitize(name: string): string {
  return name.replace(/[^A-Za-z0-9_-]+/g, '_').slice(0, 64)
}

/** What the tool returns for one hit. */
export interface SearchToolHit {
  key: string
  score: number
  text: string | null
  metadata: Record<string, unknown>
}

function toolHit(h: SearchHit, maxChars: number): SearchToolHit {
  return {
    key: h.key,
    score: Math.round(h.score * 1000) / 1000,
    text: h.text === null ? null : h.text.slice(0, maxChars),
    metadata: h.metadata,
  }
}

/**
 * A retrieval tool for agents: the model sends a query, the tool searches `index` (in the current tenant when the
 * index is scoped) and returns the matching chunks.
 *
 * ```ts
 * class SupportAgent extends Agent {
 *   tools = [searchTool(handbook, { description: 'Search the support handbook.' })]
 * }
 * ```
 */
export function searchTool(index: VectorIndex, opts: SearchToolOptions = {}): AgentTool {
  const k = opts.k ?? 5
  const maxK = Math.max(k, opts.maxK ?? 10)
  const maxChars = opts.maxChars ?? 2000
  return {
    name: sanitize(opts.name ?? `search_${index.name}`),
    description:
      opts.description ??
      `Search '${index.name}' for passages relevant to a question. Returns the best matching text chunks with their keys and metadata.`,
    inputSchema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'What to search for, in natural language.' },
        k: { type: 'integer', minimum: 1, maximum: maxK, description: `Number of passages (default ${k}).` },
      },
      required: ['query'],
      additionalProperties: false,
    },
    annotations: {
      title: `Search ${index.name}`,
      readOnlyHint: true,
      idempotentHint: true,
      openWorldHint: false,
    },
    examples: [{ query: 'How do refunds work?' }],
    parse(input: unknown) {
      const o = (input ?? {}) as Record<string, unknown>
      if (typeof o.query !== 'string' || !o.query.trim())
        throw new TypeError('`query` must be a non-empty string.')
      const n = o.k === undefined ? k : Number(o.k)
      if (!Number.isInteger(n) || n < 1 || n > maxK)
        throw new TypeError(`\`k\` must be an integer from 1 to ${maxK}.`)
      return { query: o.query, k: n }
    },
    async execute(input: { query: string; k?: number }, ctx: ToolContext) {
      const filter = typeof opts.filter === 'function' ? await opts.filter(ctx) : opts.filter
      const searchOpts: SearchOptions = {
        k: input.k ?? k,
        ...(filter ? { filter } : {}),
        ...(opts.exclude ? { exclude: opts.exclude } : {}),
        ...(opts.minScore !== undefined ? { minScore: opts.minScore } : {}),
      }
      const hits = await index.search(input.query, searchOpts)
      return { results: hits.map((h) => toolHit(h, maxChars)) }
    },
  }
}

/**
 * Hits as a context block for a prompt: one `[n] (key) text` section per hit, separated by blank lines.
 *
 * ```ts
 * const context = formatHits(await handbook.search(question))
 * instructions = `Answer from these passages and cite them as [n]:\n\n${context}`
 * ```
 */
export function formatHits(hits: SearchHit[], opts: { maxChars?: number } = {}): string {
  const maxChars = opts.maxChars ?? 2000
  return hits.map((h, i) => `[${i + 1}] (${h.key}) ${(h.text ?? '').slice(0, maxChars)}`.trim()).join('\n\n')
}

/** Search `index` and format the hits for a prompt (`formatHits(await index.search(query, opts))`). */
export async function retrieve(
  index: VectorIndex,
  query: string,
  opts: SearchOptions & { maxChars?: number } = {},
) {
  const { maxChars, ...searchOpts } = opts
  return formatHits(await index.search(query, searchOpts), maxChars === undefined ? {} : { maxChars })
}
