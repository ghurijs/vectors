// Management commands, with django-haystack's names: update_index, clear_index, rebuild_index; plus
// createvectortable for pgvector stores (Django's createcachetable).
import { type ArgumentParser, BaseCommand, CommandError, type CommandIO, restoreContext } from '@ghuri/core'
import { vectorStores } from './registry'
import { TENANT_CARRIER } from './scope'
import { ModelIndex, type VectorIndex, vectorIndexes } from './vector-index'

function names(value: unknown): string[] {
  if (value === undefined || value === null || value === '') return []
  return (Array.isArray(value) ? value : [value]).flatMap((v) =>
    String(v)
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
  )
}

function selectIndexes(args: Record<string, unknown>): VectorIndex[] {
  const wanted = names(args.index)
  const all = vectorIndexes()
  if (!wanted.length) return all
  return wanted.map((name) => {
    const found = all.find((i) => i.name === name)
    if (!found) {
      throw new CommandError(
        `Unknown vector index '${name}' (known: ${all.map((i) => i.name).join(', ') || 'none'}). Import the module that defines it from its app.`,
      )
    }
    return found
  })
}

/** Run `fn` as the tenant given by `--tenant`, through core's context carriers. */
function asTenant<R>(args: Record<string, unknown>, fn: () => Promise<R>): Promise<R> {
  const tenant =
    args.tenant === undefined || args.tenant === null || args.tenant === '' ? '' : String(args.tenant)
  return tenant ? restoreContext({ [TENANT_CARRIER]: tenant }, fn) : fn()
}

function namespaceOpt(args: Record<string, unknown>): { namespace?: string } {
  return args.namespace ? { namespace: String(args.namespace) } : {}
}

function indexArguments(parser: ArgumentParser): ArgumentParser {
  return parser
    .argument('index', { variadic: true, help: 'Index names (default: every index).' })
    .option('--tenant <pk>', { help: 'Run as this tenant (tenant-scoped indexes).' })
    .option('--namespace <name>', { help: 'A namespace inside the index.' })
}

async function update(args: Record<string, unknown>, io: CommandIO, force: boolean): Promise<void> {
  const batchSize = args.batchSize === undefined ? 100 : Number(args.batchSize)
  if (!Number.isInteger(batchSize) || batchSize < 1)
    throw new CommandError('--batch-size must be a positive integer.')
  await asTenant(args, async () => {
    for (const index of selectIndexes(args)) {
      if (!(index instanceof ModelIndex)) {
        io.stdout(`Skipping '${index.name}': not a model index.`)
        continue
      }
      const r = await index.update({ batchSize, force, ...namespaceOpt(args) })
      io.stdout(
        io.style.success(
          `Indexed ${r.indexed} ${index.model.name} rows into '${index.name}' (${r.chunks} chunks, ${r.skipped} unchanged, ${r.deleted} removed).`,
        ),
      )
    }
  })
}

async function clear(args: Record<string, unknown>, io: CommandIO): Promise<void> {
  await asTenant(args, async () => {
    for (const index of selectIndexes(args)) {
      await index.clear(namespaceOpt(args))
      io.stdout(io.style.success(`Cleared '${index.name}'.`))
    }
  })
}

/** `ghuri update_index [index…] [--tenant pk] [--batch-size 100] [--force]` */
export class UpdateIndexCommand extends BaseCommand {
  static override help = 'Index every row of model vector indexes (unchanged rows are skipped).'

  override addArguments(parser: ArgumentParser): void {
    indexArguments(parser)
      .option('--batch-size <n>', { help: 'Rows per batch (default 100).' })
      .option('--force', { help: 'Embed unchanged rows again.' })
  }

  async handle(args: Record<string, unknown>, io: CommandIO): Promise<void> {
    await update(args, io, !!args.force)
  }
}

/** `ghuri clear_index [index…] [--tenant pk]` */
export class ClearIndexCommand extends BaseCommand {
  static override help = 'Delete every vector of vector indexes (in one tenant when scoped).'

  override addArguments(parser: ArgumentParser): void {
    indexArguments(parser)
  }

  async handle(args: Record<string, unknown>, io: CommandIO): Promise<void> {
    await clear(args, io)
  }
}

/** `ghuri rebuild_index [index…] [--tenant pk]` — clear_index, then update_index. */
export class RebuildIndexCommand extends BaseCommand {
  static override help = 'Clear vector indexes, then index every row again.'

  override addArguments(parser: ArgumentParser): void {
    indexArguments(parser).option('--batch-size <n>', { help: 'Rows per batch (default 100).' })
  }

  async handle(args: Record<string, unknown>, io: CommandIO): Promise<void> {
    await clear(args, io)
    await update(args, io, true)
  }
}

/** `ghuri createvectortable [--store alias]` — create the tables of pgvector stores (Django's createcachetable). */
export class CreateVectorTableCommand extends BaseCommand {
  static override help = 'Create the tables (and the vector extension) of pgvector stores.'

  override addArguments(parser: ArgumentParser): void {
    parser.option('--store <alias>', { help: 'VECTORS alias (default: every store with tables).' })
  }

  async handle(args: Record<string, unknown>, io: CommandIO): Promise<void> {
    const aliases = args.store ? [String(args.store)] : vectorStores.aliases()
    let created = 0
    for (const alias of aliases) {
      const store = vectorStores.get(alias)
      if (!store.setup) {
        if (args.store)
          throw new CommandError(`The ${store.backend} store '${alias}' has no tables to create.`)
        continue
      }
      await store.setup()
      created++
      io.stdout(io.style.success(`Created the table of the ${store.backend} store '${alias}'.`))
    }
    if (!created) io.stdout('No pgvector stores in VECTORS.')
  }
}
