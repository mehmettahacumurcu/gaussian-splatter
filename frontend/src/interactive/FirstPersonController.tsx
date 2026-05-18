import { PointerLockControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { useEffect, useRef } from 'react'
import { Vector3 } from 'three'
import { DEFAULTS } from './config'

export function FirstPersonController() {
  const { camera } = useThree()
  const keys = useRef<Record<string, boolean>>({})

  useEffect(() => {
    const down = (e: KeyboardEvent) => { keys.current[e.code] = true }
    const up = (e: KeyboardEvent) => { keys.current[e.code] = false }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
    }
  }, [])

  useFrame((_, dt) => {
    const speed = DEFAULTS.player.walkSpeed * dt
    const forward = new Vector3()
    camera.getWorldDirection(forward)
    forward.y = 0
    forward.normalize()
    const right = new Vector3().crossVectors(forward, camera.up).normalize()

    if (keys.current['KeyW']) camera.position.addScaledVector(forward, speed)
    if (keys.current['KeyS']) camera.position.addScaledVector(forward, -speed)
    if (keys.current['KeyA']) camera.position.addScaledVector(right, -speed)
    if (keys.current['KeyD']) camera.position.addScaledVector(right, speed)
  })

  return <PointerLockControls />
}
