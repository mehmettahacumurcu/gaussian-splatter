import { useState } from 'react'
import { Scene } from './Scene'
import { HUD } from './HUD'
import { WorldSelector, type WorldEntry } from './WorldSelector'
import type { PickupState } from './types'

export function InteractivePage() {
  const [pickupState, setPickupState] = useState<PickupState>({ kind: 'IDLE' })
  const [selectedWorld, setSelectedWorld] = useState<WorldEntry | null>(null)
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <WorldSelector value={selectedWorld?.slug ?? null} onChange={setSelectedWorld} />
      <Scene world={selectedWorld} onStateChange={setPickupState} />
      <HUD pickupState={pickupState} />
    </div>
  )
}
