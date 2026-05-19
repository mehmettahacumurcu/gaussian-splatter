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

// MVP: fixture worlds hardcoded. Future: fetch index from backend.
const FIXTURE_WORLDS: WorldEntry[] = [
  {
    slug: 'fixture-a-render',
    displayName: 'Fixture A (render)',
    plyUrl: '/worlds/fixture-a-render/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/fixture-a-render/output/world/0-world-collider.json',
  },
  {
    slug: 'fixture-b-empty-photo',
    displayName: 'Fixture B (empty photo)',
    plyUrl: '/worlds/fixture-b-empty-photo/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/fixture-b-empty-photo/output/world/0-world-collider.json',
  },
  {
    slug: 'fixture-c-photo-with-objects',
    displayName: 'Fixture C (photo + objects)',
    plyUrl: '/worlds/fixture-c-photo-with-objects/output/world/0-world.ply',
    colliderJsonUrl: '/worlds/fixture-c-photo-with-objects/output/world/0-world-collider.json',
  },
]

export function WorldSelector({ value, onChange }: Props) {
  return (
    <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 10 }}>
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
        {FIXTURE_WORLDS.map((w) => (
          <option key={w.slug} value={w.slug}>
            {w.displayName}
          </option>
        ))}
      </select>
    </div>
  )
}
