// Background indexing on @ghuri/tasks, without depending on it: when @ghuri/tasks is loaded it publishes `task()` and
// `getTask()` on `globalThis[Symbol.for('ghuri.tasks.interop')]`, and vectors defines the `vectors.index` task through
// that (as @ghuri/agents defines `agents.run`). A dynamic import of a missing package breaks a Workers bundle, so
// there is no import.
import { apps, ImproperlyConfigured } from '@ghuri/core'
import type { Metadata } from './filters'
import { globalSingleton } from './global'

/** Name of the task that runs background indexing. */
export const VECTOR_TASK_NAME = 'vectors.index'
/** Attempts of a background job before its task result is `FAILED`. */
export const VECTOR_TASK_MAX_ATTEMPTS = 3

/** The argument of the `vectors.index` task (JSON). */
export interface VectorJob {
  /** The index name. */
  index: string
  op: 'index' | 'delete'
  /** Document keys (`delete`) or model primary keys (`index` on a `ModelIndex`). */
  keys?: string[]
  /** Plain documents to index (`VectorIndex`). */
  documents?: Array<{ key: string; text: string; metadata?: Metadata }>
  namespace?: string
  force?: boolean
}

/** The parts of an `@ghuri/tasks` `TaskResult` vectors uses. */
export interface VectorTaskResult {
  id: string
  status: 'READY' | 'RUNNING' | 'FAILED' | 'SUCCESSFUL'
}

interface TaskLike {
  modulePath: string
  enqueue(job: VectorJob): Promise<VectorTaskResult>
}

/** `@ghuri/tasks`' interop object (`{ version: 1, task, getTask }`). */
export interface TasksInterop {
  version: 1
  task(fn: (job: VectorJob) => Promise<unknown>, options: Record<string, unknown>): TaskLike
  getTask(name: string): TaskLike | undefined
}

/** `@ghuri/tasks`' interop object when it is loaded in this process, else undefined. */
export function tasksInterop(): TasksInterop | undefined {
  const interop = (globalThis as Record<symbol, unknown>)[Symbol.for('ghuri.tasks.interop')] as
    | TasksInterop
    | undefined
  return interop?.version === 1 ? interop : undefined
}

type JobRunner = (job: VectorJob) => Promise<unknown>

const runner = globalSingleton<{ run?: JobRunner }>('ghuri.vectors.jobRunner', () => ({}))

/** @internal Set by the index module: runs a job against the registered index. */
export function setJobRunner(fn: JobRunner): void {
  runner.run = fn
}

/** Run a background job now (the body of the `vectors.index` task). */
export async function runVectorJob(job: VectorJob): Promise<unknown> {
  if (!runner.run) throw new ImproperlyConfigured('@ghuri/vectors indexes are not loaded.')
  return runner.run(job)
}

/** The `vectors.index` task, defined on first use; undefined when `@ghuri/tasks` is not loaded. */
export function vectorTask(): TaskLike | undefined {
  const interop = tasksInterop()
  if (!interop) return undefined
  return (
    interop.getTask(VECTOR_TASK_NAME) ??
    interop.task(runVectorJob, { name: VECTOR_TASK_NAME, maxAttempts: VECTOR_TASK_MAX_ATTEMPTS })
  )
}

/** Should work with `background` run as a task? `'auto'`: when `@ghuri/tasks` is loaded. */
export function useBackground(background: boolean | 'auto' | undefined): boolean {
  if (background === true) {
    if (!tasksInterop()) {
      throw new ImproperlyConfigured(
        'background: true needs @ghuri/tasks: install it and import it in your settings.',
      )
    }
    return true
  }
  return background === 'auto' ? !!tasksInterop() : false
}

/** Enqueue a job as the `vectors.index` task. */
export async function enqueueVectorJob(job: VectorJob): Promise<VectorTaskResult> {
  const t = vectorTask()
  if (!t) throw new ImproperlyConfigured('@ghuri/tasks is not loaded.')
  return t.enqueue(job)
}

// Define the task when apps load, so a worker knows `vectors.index` before the first message arrives.
const HOOK_KEY = Symbol.for('ghuri.vectors.taskHook')
if (!(globalThis as Record<symbol, unknown>)[HOOK_KEY]) {
  ;(globalThis as Record<symbol, unknown>)[HOOK_KEY] = true
  apps.onPopulate(() => {
    vectorTask()
  })
}
