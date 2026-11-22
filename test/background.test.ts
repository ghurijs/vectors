import { ImproperlyConfigured } from '@ghuri/core'
import { afterEach, describe, expect, it } from 'vitest'
import {
  type MemoryVectorStore,
  memoryStore,
  runVectorJob,
  VECTOR_TASK_NAME,
  VectorIndex,
  type VectorJob,
  vectorStores,
  vectorTask,
} from '../src/index'
import { fakeEmbedding } from '../src/testing'
import { useHarness } from './helpers'

const KEY = Symbol.for('ghuri.tasks.interop')
const notes = new VectorIndex('test.bg', { chunk: false })

/** A stand-in for @ghuri/tasks' interop object that stores jobs instead of running them. */
function stubInterop() {
  const jobs: VectorJob[] = []
  const tasks = new Map<string, any>()
  const interop = {
    version: 1 as const,
    task(fn: (job: VectorJob) => Promise<unknown>, options: Record<string, unknown>) {
      const t = {
        modulePath: String(options.name),
        options,
        fn,
        async enqueue(job: VectorJob) {
          jobs.push(JSON.parse(JSON.stringify(job)))
          return { id: `r${jobs.length}`, status: 'READY' as const }
        },
      }
      tasks.set(t.modulePath, t)
      return t
    },
    getTask: (name: string) => tasks.get(name),
  }
  ;(globalThis as any)[KEY] = interop
  return { jobs, tasks }
}

afterEach(() => {
  delete (globalThis as any)[KEY]
})

describe('background indexing', () => {
  useHarness(
    () => ({ settings: { VECTORS: { default: memoryStore({ embedding: fakeEmbedding() }) } } }),
    () => (vectorStores.get() as MemoryVectorStore).clear(),
  )

  it('without @ghuri/tasks, background: true raises and "auto" runs inline', async () => {
    expect(vectorTask()).toBeUndefined()
    await expect(notes.index({ key: 'a', text: 'hello' }, { background: true })).rejects.toThrow(
      ImproperlyConfigured,
    )
    const r = await notes.index({ key: 'a', text: 'hello world' }, { background: 'auto' })
    expect(r.indexed).toBe(1)
  })

  it('enqueues JSON jobs on the vectors.index task and runs them by index name', async () => {
    const { jobs, tasks } = stubInterop()
    const r = await notes.index(
      { key: 'b', text: 'background text', metadata: { lang: 'en' } },
      { background: true },
    )
    expect(r.taskResultId).toBe('r1')
    expect(tasks.get(VECTOR_TASK_NAME).options).toEqual({ name: VECTOR_TASK_NAME, maxAttempts: 3 })
    expect(jobs[0]).toEqual({
      index: 'test.bg',
      op: 'index',
      documents: [{ key: 'b', text: 'background text', metadata: { lang: 'en' } }],
    })
    expect(await notes.search('background')).toEqual([])
    await runVectorJob(jobs[0])
    expect((await notes.search('background'))[0]).toMatchObject({ key: 'b', metadata: { lang: 'en' } })

    await notes.delete('b', { background: true })
    expect(jobs[1]).toEqual({ index: 'test.bg', op: 'delete', keys: ['b'] })
    await tasks.get(VECTOR_TASK_NAME).fn(jobs[1])
    expect(await notes.search('background')).toEqual([])
    await expect(runVectorJob({ index: 'nope', op: 'delete' })).rejects.toThrow(
      /No vector index named 'nope'/,
    )
  })
})
