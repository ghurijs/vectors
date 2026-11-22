import { type CheckContext, type CheckMessage, checks } from '@ghuri/core'
import { vectorStores } from './registry'
import { VectorizeStore } from './stores/vectorize'
import { vectorIndexes } from './vector-index'

/** Strip `//` and `/* *\/` comments and trailing commas from JSONC, leaving strings alone. */
export function parseJsonc(text: string): unknown {
  let out = ''
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (ch === '"') {
      let j = i + 1
      while (j < text.length && text[j] !== '"') j += text[j] === '\\' ? 2 : 1
      out += text.slice(i, j + 1)
      i = j
    } else if (ch === '/' && text[i + 1] === '/') {
      while (i < text.length && text[i] !== '\n') i++
      out += '\n'
    } else if (ch === '/' && text[i + 1] === '*') {
      i = text.indexOf('*/', i + 2)
      if (i < 0) break
      i++
    } else {
      out += ch
    }
  }
  return JSON.parse(out.replace(/,(\s*[}\]])/g, '$1'))
}

interface WranglerConfig {
  vectorize?: Array<{ binding: string; index_name?: string }>
  ai?: { binding?: string }
}

/** wrangler.jsonc / wrangler.json in the working directory (Node only), or null. */
async function readWranglerConfig(): Promise<{ file: string; config: WranglerConfig } | null> {
  const proc = (globalThis as any).process
  if (typeof proc?.cwd !== 'function' || !proc.versions?.node) return null
  try {
    const fsSpec = 'node:fs/promises'
    const pathSpec = 'node:path'
    const fs = await import(/* @vite-ignore */ fsSpec)
    const path = await import(/* @vite-ignore */ pathSpec)
    for (const name of ['wrangler.jsonc', 'wrangler.json']) {
      let text: string
      try {
        text = await fs.readFile(path.join(proc.cwd(), name), 'utf8')
      } catch {
        continue
      }
      return { file: name, config: parseJsonc(text) as WranglerConfig }
    }
  } catch {
    return null
  }
  return null
}

/** `vectors.*` system checks. */
export async function checkVectors(ctx: CheckContext): Promise<CheckMessage[]> {
  const messages: CheckMessage[] = []
  const setting = ctx.settings.VECTORS as unknown
  if (setting !== undefined && (typeof setting !== 'object' || setting === null)) {
    return [checks.error('VECTORS must be an object of vector stores.', { id: 'vectors.E001' })]
  }
  const ok = new Set<string>()
  for (const alias of vectorStores.aliases()) {
    try {
      messages.push(...vectorStores.get(alias).check())
      ok.add(alias)
    } catch (err) {
      messages.push(
        checks.error((err as Error).message, {
          hint: 'Use vectorize(), pgvector(), databaseStore() or memoryStore() from @ghuri/vectors.',
          id: 'vectors.E001',
          obj: `VECTORS['${alias}']`,
        }),
      )
    }
  }
  for (const index of vectorIndexes()) {
    if (vectorStores.config(index.storeAlias) === undefined) {
      messages.push(
        checks.error(
          `The vector index '${index.name}' uses the store '${index.storeAlias}', which is not configured.`,
          {
            hint: `Add '${index.storeAlias}' to settings.VECTORS.`,
            id: 'vectors.E002',
            obj: index.name,
          },
        ),
      )
    }
  }
  const wrangler = await readWranglerConfig()
  if (wrangler) messages.push(...checkWrangler(wrangler.file, wrangler.config, ok, ctx))
  return messages
}

export function checkWrangler(
  file: string,
  config: WranglerConfig,
  ok: Set<string>,
  ctx: CheckContext,
): CheckMessage[] {
  const out: CheckMessage[] = []
  const bindings = new Set((config.vectorize ?? []).map((v) => v.binding))
  const aiBinding = (ctx.settings as any).AGENTS?.AI_BINDING ?? 'AI'
  for (const alias of ok) {
    const store = vectorStores.get(alias)
    if (store instanceof VectorizeStore && !bindings.has(store.binding)) {
      out.push(
        checks.error(`The Vectorize binding '${store.binding}' (VECTORS['${alias}']) is not in ${file}.`, {
          hint: `Add { "binding": "${store.binding}", "index_name": "<index>" } to "vectorize" and create the index with wrangler vectorize create.`,
          id: 'vectors.E005',
          obj: `VECTORS['${alias}']`,
        }),
      )
    }
    if (
      store.hasEmbedding &&
      store.embedding.modelId.startsWith('workers-ai:') &&
      config.ai?.binding !== aiBinding
    ) {
      out.push(
        checks.warning(
          `VECTORS['${alias}'] embeds with Workers AI, but ${file} has no "ai" binding '${aiBinding}'.`,
          {
            hint: `Add "ai": { "binding": "${aiBinding}" } to ${file}.`,
            id: 'vectors.W003',
            obj: `VECTORS['${alias}']`,
          },
        ),
      )
    }
  }
  return out
}

checks.register(checkVectors, { tags: ['vectors'] })
