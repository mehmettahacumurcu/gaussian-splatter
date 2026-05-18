import { useEffect, useState } from 'react'
import { DEFAULTS } from './config'

export function HUD() {
  const [locked, setLocked] = useState(false)

  useEffect(() => {
    const onLockChange = () => setLocked(document.pointerLockElement !== null)
    document.addEventListener('pointerlockchange', onLockChange)
    return () => document.removeEventListener('pointerlockchange', onLockChange)
  }, [])

  return (
    <>
      {!locked && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(0,0,0,0.6)',
            color: 'white',
            fontSize: 24,
            zIndex: 10,
            pointerEvents: 'none',
          }}
        >
          Click to start
        </div>
      )}
      {locked && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            width: DEFAULTS.ui.crosshairSize,
            height: DEFAULTS.ui.crosshairSize,
            borderRadius: '50%',
            background: 'white',
            pointerEvents: 'none',
            zIndex: 10,
          }}
        />
      )}
    </>
  )
}
