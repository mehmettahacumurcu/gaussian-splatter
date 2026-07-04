import { useEffect, useState } from 'react'

export interface WorldEntry {
  slug: string
  displayName: string
  plyUrl: string
  colliderJsonUrl: string
}

interface Props {
  value: string | null
  onChange: (entry: WorldEntry | null) => void
}

// MVP: worlds hardcoded. Future: fetch index from backend.
// Real-quality entries from the project's static 3DGS pipeline come first;
// the early single-image LucidDreamer fixtures are kept below for comparison.
const FIXTURE_WORLDS: WorldEntry[] = [
  {
    slug: 'garden',
    displayName: 'Garden (Mip-NeRF360, Colab SOTA)',
    plyUrl: '/worlds/garden/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/garden/output/world/0-world-collider.json',
  },
  {
    slug: 'myroom',
    displayName: 'My Room (static 3DGS, Colab)',
    plyUrl: '/worlds/myroom/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/myroom/output/world/0-world-collider.json',
  },
  {
    slug: 'myroom-v2',
    displayName: 'My Room v2 (static 3DGS)',
    plyUrl: '/worlds/myroom-v2/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/myroom-v2/output/world/0-world-collider.json',
  },
  {
    slug: 'banana-demo',
    displayName: 'Banana Demo (static 3DGS)',
    plyUrl: '/worlds/banana-demo/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/banana-demo/output/world/0-world-collider.json',
  },
  {
    slug: 'fixture-a-render',
    displayName: 'Fixture A (single-image, low quality)',
    plyUrl: '/worlds/fixture-a-render/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/fixture-a-render/output/world/0-world-collider.json',
  },
  {
    slug: 'fixture-b-empty-photo',
    displayName: 'Fixture B (single-image, low quality)',
    plyUrl: '/worlds/fixture-b-empty-photo/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/fixture-b-empty-photo/output/world/0-world-collider.json',
  },
  {
    slug: 'fixture-c-photo-with-objects',
    displayName: 'Fixture C (single-image, low quality)',
    plyUrl: '/worlds/fixture-c-photo-with-objects/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/fixture-c-photo-with-objects/output/world/0-world-collider.json',
  },
]

// Probe each world's collider JSON (cheap ~400 B HEAD request) to detect
// which worlds have files on disk. Accepts an injectable fetchFn for testing.
export async function probeWorldAvailability(
  worlds: WorldEntry[],
  fetchFn: (url: string) => Promise<Pick<Response, 'ok'>> = (url) =>
    fetch(url, { method: 'HEAD' }),
): Promise<Record<string, boolean>> {
  const pairs = await Promise.all(
    worlds.map(async (w) => {
      try {
        const r = await fetchFn(w.colliderJsonUrl)
        return [w.slug, r.ok] as const
      } catch {
        return [w.slug, false] as const
      }
    }),
  )
  return Object.fromEntries(pairs)
}

// WorldSelector renders just the <select> — positioning is handled by the
// parent (InteractivePage) so both the selector and the fly-mode toggle can
// share a single top-right flex row without double absolute positioning.
export function WorldSelector({ value, onChange }: Props) {
  const [availability, setAvailability] = useState<Record<string, boolean>>({})
  const [probesDone, setProbesDone] = useState(false)

  useEffect(() => {
    let cancelled = false
    probeWorldAvailability(FIXTURE_WORLDS).then((result) => {
      if (!cancelled) {
        setAvailability(result)
        setProbesDone(true)
      }
    })
    return () => { cancelled = true }
  }, [])

  return (
    <select
      data-testid="world-selector"
      value={value ?? ''}
      onChange={(e) => {
        const slug = e.target.value
        onChange(slug ? (FIXTURE_WORLDS.find((w) => w.slug === slug) ?? null) : null)
      }}
      style={{ padding: '4px 8px', background: '#222', color: '#eee', border: '1px solid #444' }}
    >
      <option value="">— built-in test room —</option>
      {FIXTURE_WORLDS.map((w) => {
        const available = !probesDone || (availability[w.slug] ?? true)
        return (
          <option key={w.slug} value={w.slug} disabled={!available}>
            {w.displayName}{!available ? ' (files missing)' : ''}
          </option>
        )
      })}
    </select>
  )
}
