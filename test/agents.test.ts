import { Agent, isAgentTool } from '@ghuri/agents'
import { AnonymousUser } from '@ghuri/core'
import { describe, expect, it } from 'vitest'
import { formatHits, retrieve, searchTool } from '../src/agents'
import { memoryStore, VectorIndex } from '../src/index'
import { fakeEmbedding } from '../src/testing'
import { useHarness } from './helpers'

const handbook = new VectorIndex('support.handbook', { chunk: false, tenant: false })

class SupportAgent extends Agent {
  override model = 'openai:gpt-test'
  override tools = [
    searchTool(handbook, {
      description: 'Search the support handbook.',
      filter: (ctx) => (ctx.user.isAuthenticated ? undefined : { audience: 'public' }),
    }),
  ]
}

describe('@ghuri/vectors/agents', () => {
  useHarness(() => ({ settings: { VECTORS: { default: memoryStore({ embedding: fakeEmbedding() }) } } }))

  it('searchTool is an AgentTool that agents accept', async () => {
    const tools = new SupportAgent().getTools()
    expect(tools.map((t) => t.name)).toEqual(['search_support_handbook'])
    const [tool] = tools
    expect(isAgentTool(tool)).toBe(true)
    expect(tool.annotations?.readOnlyHint).toBe(true)
    expect(tool.inputSchema).toMatchObject({ type: 'object', required: ['query'] })
    expect(() => tool.parse!({})).toThrow(/query/)
    expect(() => tool.parse!({ query: 'x', k: 99 })).toThrow(/k/)
    expect(tool.parse!({ query: 'refunds' })).toEqual({ query: 'refunds', k: 5 })
  })

  it('runs searches with the tool context filter', async () => {
    await handbook.index([
      { key: 'refunds-public', text: 'Refunds are paid in five days', metadata: { audience: 'public' } },
      { key: 'refunds-staff', text: 'Refunds over 500 need a manager', metadata: { audience: 'staff' } },
    ])
    const [tool] = new SupportAgent().getTools()
    const anon = (await tool.execute({ query: 'refunds', k: 5 }, { user: new AnonymousUser() })) as any
    expect(anon.results.map((r: any) => r.key)).toEqual(['refunds-public'])
    const staff = (await tool.execute(
      { query: 'refunds', k: 5 },
      { user: { isAuthenticated: true, pk: 1 } as any },
    )) as any
    expect(staff.results.map((r: any) => r.key).sort()).toEqual(['refunds-public', 'refunds-staff'])
    expect(staff.results[0]).toEqual({
      key: expect.any(String),
      score: expect.any(Number),
      text: expect.stringContaining('Refunds'),
      metadata: { audience: expect.any(String) },
    })
  })

  it('formats hits as a prompt context block', async () => {
    await handbook.index({ key: 'shipping', text: 'Shipping takes two weeks' })
    const text = await retrieve(handbook, 'shipping', { k: 1 })
    expect(text).toBe('[1] (shipping) Shipping takes two weeks')
    expect(formatHits([])).toBe('')
  })
})
