import { describe, expect, it } from 'vitest'
import { probeWorldAvailability, type WorldEntry } from '../WorldSelector'

const makeWorld = (slug: string): WorldEntry => ({
  slug,
  displayName: slug,
  plyUrl: `/worlds/${slug}/0-world.ply`,
  colliderJsonUrl: `/worlds/${slug}/0-world-collider.json`,
})

describe('probeWorldAvailability', () => {
  it('marks a world as available when fetch returns ok=true', async () => {
    const worlds = [makeWorld('garden')]
    const mockFetch = async (_url: string) => ({ ok: true })
    const result = await probeWorldAvailability(worlds, mockFetch)
    expect(result['garden']).toBe(true)
  })

  it('marks a world as unavailable when fetch returns ok=false (404)', async () => {
    const worlds = [makeWorld('myroom')]
    const mockFetch = async (_url: string) => ({ ok: false })
    const result = await probeWorldAvailability(worlds, mockFetch)
    expect(result['myroom']).toBe(false)
  })

  it('marks a world as unavailable when fetch throws (network error)', async () => {
    const worlds = [makeWorld('missing')]
    const mockFetch = async (_url: string): Promise<{ ok: boolean }> => {
      throw new Error('network error')
    }
    const result = await probeWorldAvailability(worlds, mockFetch)
    expect(result['missing']).toBe(false)
  })

  it('handles multiple worlds independently', async () => {
    const worlds = [makeWorld('garden'), makeWorld('myroom'), makeWorld('gone')]
    const mockFetch = async (url: string) => ({
      ok: !url.includes('myroom') && !url.includes('gone'),
    })
    const result = await probeWorldAvailability(worlds, mockFetch)
    expect(result['garden']).toBe(true)
    expect(result['myroom']).toBe(false)
    expect(result['gone']).toBe(false)
  })

  it('returns all slugs present in the input array', async () => {
    const worlds = [makeWorld('a'), makeWorld('b'), makeWorld('c')]
    const mockFetch = async (_url: string) => ({ ok: true })
    const result = await probeWorldAvailability(worlds, mockFetch)
    expect(Object.keys(result).sort()).toEqual(['a', 'b', 'c'])
  })

  it('returns empty record for empty input', async () => {
    const result = await probeWorldAvailability([], async (_url) => ({ ok: true }))
    expect(result).toEqual({})
  })
})
