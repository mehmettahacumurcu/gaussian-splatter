import { useEffect, useState } from 'react'
import { Scene } from './Scene'
import { HUD } from './HUD'
import { WorldSelector, type WorldEntry } from './WorldSelector'
import type { PickupState } from './types'

export function InteractivePage() {
  const [pickupState, setPickupState] = useState<PickupState>({ kind: 'IDLE' })
  const [selectedWorld, setSelectedWorld] = useState<WorldEntry | null>(null)
  const [flyMode, setFlyMode] = useState(false)

  // [F] toggles collision/fly mode. Guard against key repeat so a held F
  // does not rapidly toggle. (KeyE and KeyG are taken by the pickup system.)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'KeyF' && !e.repeat) setFlyMode((prev) => !prev)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Top-right control bar: fly toggle + world selector share one row */}
      <div
        style={{
          position: 'absolute', top: 8, right: 8, zIndex: 10,
          display: 'flex', alignItems: 'center', gap: 6,
        }}
      >
        <button
          onClick={() => setFlyMode((prev) => !prev)}
          title="Toggle fly / collision mode (F)"
          style={{
            padding: '4px 8px',
            background: flyMode ? '#1a3a1a' : '#222',
            color: flyMode ? '#6f6' : '#eee',
            border: `1px solid ${flyMode ? '#4a4' : '#444'}`,
            cursor: 'pointer',
            fontSize: 13,
            fontFamily: 'monospace',
            userSelect: 'none',
          }}
        >
          {flyMode ? 'FLY [F]' : 'Walk [F]'}
        </button>
        <WorldSelector value={selectedWorld?.slug ?? null} onChange={setSelectedWorld} />
      </div>
      <Scene world={selectedWorld} onStateChange={setPickupState} flyMode={flyMode} />
      <HUD pickupState={pickupState} flyMode={flyMode} />
    </div>
  )
}
