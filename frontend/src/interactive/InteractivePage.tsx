import { useState } from 'react'
import { Scene } from './Scene'
import { HUD } from './HUD'
import type { PickupState } from './types'

export function InteractivePage() {
  const [pickupState, setPickupState] = useState<PickupState>({ kind: 'IDLE' })
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Scene onStateChange={setPickupState} />
      <HUD pickupState={pickupState} />
    </div>
  )
}
