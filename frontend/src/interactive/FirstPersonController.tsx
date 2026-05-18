import { PointerLockControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { CapsuleCollider, RigidBody, type RapierRigidBody } from '@react-three/rapier'
import { useEffect, useRef } from 'react'
import { Vector3 } from 'three'
import { DEFAULTS } from './config'

interface Props {
  spawn: [number, number, number]
}

export function FirstPersonController({ spawn }: Props) {
  const { camera } = useThree()
  const bodyRef = useRef<RapierRigidBody>(null)
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

  useFrame(() => {
    const body = bodyRef.current
    if (!body) return

    const forward = new Vector3()
    camera.getWorldDirection(forward)
    forward.y = 0
    forward.normalize()
    const right = new Vector3().crossVectors(forward, new Vector3(0, 1, 0)).normalize()

    const dir = new Vector3()
    if (keys.current['KeyW']) dir.add(forward)
    if (keys.current['KeyS']) dir.sub(forward)
    if (keys.current['KeyD']) dir.add(right)
    if (keys.current['KeyA']) dir.sub(right)

    if (dir.lengthSq() > 0) dir.normalize().multiplyScalar(DEFAULTS.player.walkSpeed)

    const linvel = body.linvel()
    body.setLinvel({ x: dir.x, y: linvel.y, z: dir.z }, true)

    // Rapier CapsuleCollider total height = halfHeight*2 + radius*2. To get a
    // human-sized capsule (capsuleHeight ≈ 1.7m), the cylinder halfHeight is
    // (capsuleHeight - 2*radius) / 2. At rest the body center sits at radius +
    // halfHeight = capsuleHeight/2 above the floor, so the eye offset from
    // body center is (capsuleHeight/2 - 2*radius/2) — but simpler: place eye
    // at body_y + (capsuleHeight/2 - radius) so the eye lands at capsuleHeight
    // above the floor when standing on it.
    const t = body.translation()
    const eyeOffset = DEFAULTS.player.capsuleHeight / 2 - DEFAULTS.player.capsuleRadius
    camera.position.set(t.x, t.y + eyeOffset, t.z)
  })

  // Cylinder halfHeight for CapsuleCollider — total capsule height is
  // halfHeight*2 + radius*2, which should equal capsuleHeight.
  const capsuleHalfHeight = DEFAULTS.player.capsuleHeight / 2 - DEFAULTS.player.capsuleRadius

  return (
    <>
      <RigidBody
        ref={bodyRef}
        type="dynamic"
        position={spawn}
        lockRotations
        colliders={false}
        mass={70}
      >
        <CapsuleCollider args={[capsuleHalfHeight, DEFAULTS.player.capsuleRadius]} />
      </RigidBody>
      <PointerLockControls />
    </>
  )
}
