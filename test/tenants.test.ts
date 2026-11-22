import { AppConfig, bufferIO, callCommand, getContextCarriers } from '@ghuri/core'
import { CharField, type Manager, Model, TextField } from '@ghuri/orm'
import { dummy } from '@ghuri/tasks'
import { useTaskBackend } from '@ghuri/tasks/testing'
import { runAsTenant, tenantModel, withoutTenant } from '@ghuri/tenants'
import { describe, expect, it } from 'vitest'
import {
  MemoryVectorStore,
  ModelIndex,
  TenantRequired,
  VECTOR_TASK_NAME,
  VectorIndex,
  vectorStores,
} from '../src/index'
import { storeCases } from './backends'
import { useHarness } from './helpers'

export class Org extends Model({ name: new CharField({ maxLength: 50 }) }) {
  declare static objects: Manager<Org>
  static Meta = { appLabel: 'orgs' }
}
const OrgModel = tenantModel({ field: 'org', to: () => Org })
export class Note extends OrgModel({ text: new TextField() }) {
  declare static objects: Manager<Note>
  static Meta = { appLabel: 'orgs' }
}
class OrgsConfig extends AppConfig {
  name = 'orgs'
  override label = 'orgs'
  override models = { Org, Note }
}

const docs = new VectorIndex('test.docs', { chunk: false })
const shared = new VectorIndex('test.shared', { chunk: false, tenant: false })
const notes = new ModelIndex(Note, { text: ['text'], chunk: false })

const cases = storeCases({ TENANTS: { MODEL: 'orgs.Org' } }).filter((c) =>
  ['memory', 'database (sqlite)', 'database (pglite)', 'pgvector', 'vectorize'].includes(c.name),
)

for (const c of cases) {
  describe(`tenants: ${c.name}`, () => {
    const h = useHarness(
      () => ({ ...c.harness(), apps: [OrgsConfig] }),
      async () => {
        const store = vectorStores.get()
        if (store instanceof MemoryVectorStore) store.clear()
        ;(h.env as any).VECTORIZE?.vectors.clear()
      },
    )
    const orgs = async () => {
      const a = await Org.objects.create({ name: 'a' })
      const b = await Org.objects.create({ name: 'b' })
      return { a, b }
    }

    it('setup: @ghuri/tenants registers the tenant carrier, so indexes are scoped', async () => {
      await c.prepare?.()
      expect(getContextCarriers()).toContain('tenant')
    })

    it('keeps tenants apart and fails closed without a tenant', async () => {
      const { a, b } = await orgs()
      await runAsTenant(a, () => docs.index({ key: 'plan', text: 'Acme pricing plan is gold' }))
      await runAsTenant(b, () => docs.index({ key: 'plan', text: 'Beta pricing plan is silver' }))
      const hitsA = await runAsTenant(a, () => docs.search('pricing plan', { k: 10 }))
      const hitsB = await runAsTenant(b, () => docs.search('pricing plan', { k: 10 }))
      expect(hitsA.map((x) => x.text)).toEqual(['Acme pricing plan is gold'])
      expect(hitsB.map((x) => x.text)).toEqual(['Beta pricing plan is silver'])
      expect(hitsA[0].id).not.toBe(hitsB[0].id)

      await expect(docs.search('pricing')).rejects.toThrow(TenantRequired)
      await expect(docs.index({ key: 'x', text: 'x' })).rejects.toThrow(TenantRequired)
      await expect(docs.delete('plan')).rejects.toThrow(TenantRequired)
      await expect(docs.clear()).rejects.toThrow(TenantRequired)
      await expect(withoutTenant(() => docs.search('pricing'))).rejects.toThrow(TenantRequired)

      await runAsTenant(a, () => docs.clear())
      expect(await runAsTenant(a, () => docs.search('pricing plan'))).toEqual([])
      expect((await runAsTenant(b, () => docs.search('pricing plan'))).length).toBe(1)

      // tenant: false indexes are shared by every tenant.
      await shared.index({ key: 'tos', text: 'Terms of service' })
      expect((await runAsTenant(b, () => shared.search('terms')))[0].key).toBe('tos')
    })

    it('indexes tenant models on save, in the background with the tenant carried', async () => {
      const { a, b } = await orgs()
      const { backend, restore } = useTaskBackend(dummy())
      try {
        await runAsTenant(a, () => Note.objects.create({ text: 'Refunds take five days' }))
        await runAsTenant(b, () => Note.objects.create({ text: 'Refunds take ten days' }))
        expect(backend.results.map((r: any) => r.task.modulePath)).toEqual([
          VECTOR_TASK_NAME,
          VECTOR_TASK_NAME,
        ])
        expect(await runAsTenant(a, () => notes.search('refunds'))).toEqual([])
        await backend.runPending()
        const hitsA = await runAsTenant(a, () => notes.search('refunds', { k: 5 }))
        expect(hitsA.map((x) => x.object.text)).toEqual(['Refunds take five days'])
        const hitsB = await runAsTenant(b, () => notes.search('refunds', { k: 5 }))
        expect(hitsB.map((x) => x.object.text)).toEqual(['Refunds take ten days'])
      } finally {
        restore()
      }
    })

    it('update_index --tenant indexes one tenant; without it the command fails closed', async () => {
      const { a, b } = await orgs()
      notes.disconnectSignals()
      try {
        await runAsTenant(a, () => Note.objects.create({ text: 'alpha note' }))
        await runAsTenant(b, () => Note.objects.create({ text: 'beta note' }))
        await expect(callCommand('update_index', { index: ['orgs.note'] }, bufferIO())).rejects.toThrow(
          TenantRequired,
        )
        const io = bufferIO()
        await callCommand('update_index', { index: ['orgs.note'], tenant: String(a.pk) }, io)
        expect(io.out.join('\n')).toMatch(/Indexed 1 Note rows/)
        expect((await runAsTenant(a, () => notes.search('note'))).map((x) => x.object.text)).toEqual([
          'alpha note',
        ])
        expect(await runAsTenant(b, () => notes.search('note'))).toEqual([])
      } finally {
        ;(notes as any).connectSignals()
      }
    })
  })
}
