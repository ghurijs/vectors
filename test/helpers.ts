import { PGlite } from '@electric-sql/pglite'
import { vector } from '@electric-sql/pglite-pgvector'
import { createApp, defineSettings, type Env, type GhuriSettings, tryGetContext } from '@ghuri/core'
import { connections, type DatabaseConfig } from '@ghuri/orm'
import { PGliteConnection } from '@ghuri/orm/pglite'
import {
  migrateOrCreateTables,
  setupTestDatabase,
  type TestBackend,
  type TestDatabase,
} from '@ghuri/orm/testing'
import { afterAll, beforeAll, beforeEach } from 'vitest'
import { VectorsConfig } from '../src/index'
import { DemoConfig } from './demo'

export const dbBackends: TestBackend[] = (process.env.GHURI_TEST_BACKENDS?.split(',') as
  | TestBackend[]
  | undefined) ?? ['sqlite', 'pglite', 'd1']

/** `'pgvector'`: PGlite with the pgvector extension as the default database. */
export type HarnessBackend = TestBackend | 'pgvector'

export interface HarnessOptions {
  using?: HarnessBackend
  settings?: Partial<GhuriSettings> & Record<string, unknown>
  env?: Env
  /** Install the vectors app (default true). */
  vectorsApp?: boolean
  apps?: unknown[]
}

/** A PGlite database with pgvector, as `DATABASES.default`, with migrations applied. */
export async function setupPgvectorDatabase(): Promise<TestDatabase> {
  const pg = new PGlite({ extensions: { vector } })
  await pg.waitReady
  await pg.exec('CREATE EXTENSION IF NOT EXISTS vector')
  const conn = new PGliteConnection(pg)
  conn.alias = 'default'
  const config: DatabaseConfig = {
    vendor: 'pglite',
    dialect: 'postgresql',
    connect: () => conn,
    options: { test: true },
  }
  const ctx = tryGetContext()!
  const previous = ctx.settings.DATABASES
  ctx.settings.DATABASES = { ...(previous ?? {}), default: config }
  connections.reset('default')
  const { apps } = await import('@ghuri/core')
  const models = apps.getModels() as any[]
  await migrateOrCreateTables(conn, models)
  return {
    alias: 'default',
    using: 'pglite',
    connection: conn,
    models,
    async flush() {
      const tables = (
        await conn.execute(
          "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename NOT LIKE 'ghuri_%' AND tablename <> 'django_migrations'",
        )
      ).rows.map((r) => `"${r.tablename}"`)
      const keep = new Set(['"ghuri_migrations"'])
      const list = tables.filter((t) => !keep.has(t) && !t.includes('migrations'))
      if (list.length) await conn.execute(`TRUNCATE ${list.join(', ')} RESTART IDENTITY CASCADE`)
    },
    async teardown() {
      connections.reset('default')
      await conn.close()
      ctx.settings.DATABASES = previous
    },
  }
}

export async function setupHarness(opts: HarnessOptions = {}) {
  const env: Env = opts.env ?? {}
  const installed = [...(opts.vectorsApp === false ? [] : [VectorsConfig]), DemoConfig, ...(opts.apps ?? [])]
  const settings = defineSettings(() => ({
    SECRET_KEY: 'test-secret',
    DEBUG: true,
    INSTALLED_APPS: installed as any,
    ROOT_URLCONF: [],
    ...opts.settings,
  }))
  const app = createApp(settings, { checks: false })
  const ctx = await app.setup(env)
  const using = opts.using ?? 'sqlite'
  const db = using === 'pgvector' ? await setupPgvectorDatabase() : await setupTestDatabase({ using })
  return { app, env, ctx, db, using }
}

export type Harness = Awaited<ReturnType<typeof setupHarness>>

/** Per-describe harness; the database is flushed before each test. */
export function useHarness(
  factory: () => HarnessOptions = () => ({}),
  beforeEachFn?: () => unknown,
): Harness {
  const h = {} as Harness
  beforeAll(async () => {
    Object.assign(h, await setupHarness(factory()))
  }, 60_000)
  afterAll(async () => {
    await h.db?.teardown()
  })
  beforeEach(async () => {
    await h.db?.flush()
    await beforeEachFn?.()
  })
  return h
}
