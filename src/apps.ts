import { AppConfig } from '@ghuri/core'
import {
  ClearIndexCommand,
  CreateVectorTableCommand,
  RebuildIndexCommand,
  UpdateIndexCommand,
} from './commands'
import * as vectorsMigrations from './migrations/index'
import { VectorDocument, VectorEmbedding } from './models'

/** `path` when running from the TypeScript sources (makemigrations), undefined in dist / on Workers. */
function appPath(): string | undefined {
  const meta = import.meta as ImportMeta & { dirname?: string; filename?: string }
  if (!meta.dirname || !/[\\/]src$/.test(meta.dirname) || !/\.[cm]?ts$/.test(meta.filename ?? ''))
    return undefined
  return meta.dirname
}

/**
 * The `vectors` app: the document manifest (`vectors_vectordocument`), the `databaseStore()` table
 * (`vectors_vectorembedding`) and the `update_index` / `clear_index` / `rebuild_index` / `createvectortable`
 * commands. Add it to `INSTALLED_APPS` and run `ghuri migrate`.
 */
export class VectorsConfig extends AppConfig {
  name = '@ghuri/vectors'
  override label = 'vectors'
  override verboseName = 'Vectors'
  override path = appPath()
  override migrations = vectorsMigrations
  override models = { VectorDocument, VectorEmbedding }
  override commands = {
    update_index: UpdateIndexCommand,
    clear_index: ClearIndexCommand,
    rebuild_index: RebuildIndexCommand,
    createvectortable: CreateVectorTableCommand,
  }
}
