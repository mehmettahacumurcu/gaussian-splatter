import { useEffect, useState } from 'react'
import { DEFAULTS } from './config'
import type { PickupState } from './types'

interface Props {
  pickupState: PickupState
  flyMode?: boolean
}

export function HUD({ pickupState, flyMode = false }: Props) {
  const [locked, setLocked] = useState(false)

  useEffect(() => {
    const onLockChange = () => setLocked(document.pointerLockElement !== null)
    document.addEventListener('pointerlockchange', onLockChange)
    return () => document.removeEventListener('pointerlockchange', onLockChange)
  }, [])

  const aiming = pickupState.kind === 'AIMING_AT_OBJ'

  return (
    <>
      {!locked && (
        <div
          style={{
            position: 'absolute', inset: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(0,0,0,0.6)', color: 'white', fontSize: 24,
            zIndex: 10, pointerEvents: 'none',
          }}
        >
          Click to start
        </div>
      )}
      {locked && (
        <>
          <div
            style={{
              position: 'absolute', top: '50%', left: '50%',
              transform: 'translate(-50%, -50%)',
              width: DEFAULTS.ui.crosshairSize,
              height: DEFAULTS.ui.crosshairSize,
              borderRadius: '50%',
              background: aiming ? '#ffe44d' : 'white',
              pointerEvents: 'none', zIndex: 10,
              transition: `background ${DEFAULTS.ui.promptFadeMs}ms`,
            }}
          />
          {aiming && (
            <div
              style={{
                position: 'absolute',
                top: 'calc(50% + 20px)', left: '50%',
                transform: 'translateX(-50%)',
                color: 'white', fontSize: 14,
                background: 'rgba(0,0,0,0.5)',
                padding: '4px 8px', borderRadius: 4,
                pointerEvents: 'none', zIndex: 10,
              }}
            >
              [E] pick up
            </div>
          )}
          {pickupState.kind === 'HOLDING' && (
            <div
              style={{
                position: 'absolute',
                top: 'calc(50% + 20px)', left: '50%',
                transform: 'translateX(-50%)',
                color: 'white', fontSize: 14,
                background: 'rgba(0,0,0,0.5)',
                padding: '4px 8px', borderRadius: 4,
                pointerEvents: 'none', zIndex: 10,
              }}
            >
              [G] drop &nbsp; [LClick] throw
            </div>
          )}
          {flyMode && (
            <div
              style={{
                position: 'absolute', bottom: 8, left: 8,
                color: '#6f6', fontSize: 13,
                background: 'rgba(0,0,0,0.55)',
                padding: '4px 8px', borderRadius: 4,
                pointerEvents: 'none', zIndex: 10,
                fontFamily: 'monospace',
              }}
            >
              FLY MODE &nbsp; [F] walk &nbsp; Space/Shift = up/down
            </div>
          )}
        </>
      )}
    </>
  )
}
