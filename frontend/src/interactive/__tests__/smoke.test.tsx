import { describe, expect, it } from 'vitest'

describe('InteractivePage module smoke', () => {
  it('exports the InteractivePage component', async () => {
    const mod = await import('../InteractivePage')
    expect(typeof mod.InteractivePage).toBe('function')
  })
})
