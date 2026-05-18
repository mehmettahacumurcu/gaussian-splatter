import { PointerLockControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { CapsuleCollider, RigidBody, type RapierRigidBody } from '@react-three/rapier'
import { useEffect, useMemo, useRef } from 'react'
import { Vector3 } from 'three'
import { DEFAULTS } from './config'

interface Props {
  spawn: [number, number, number]
}

// Rapier CapsuleCollider total height = halfHeight*2 + radius*2. Solving for
// halfHeight given the desired total capsuleHeight: halfHeight =
// capsuleHeight/2 - radius. With capsuleHeight=1.7 and radius=0.4 that's
// halfHeight=0.45 → total 1.7m human-size capsule.
const CAPSULE_HALF_HEIGHT = DEFAULTS.player.capsuleHeight / 2 - DEFAULTS.player.capsuleRadius

// Body center sits at radius + halfHeight = capsuleHeight/2 above the floor
// at rest. Spawning at this Y avoids a visible drop on scene load.
const BODY_REST_Y = DEFAULTS.player.capsuleRadius + CAPSULE_HALF_HEIGHT

// Eye offset from body center. With eyeHeight === capsuleHeight (typical
// human), this resolves to capsuleHeight/2 and puts the eye at eyeHeight
// above the floor when the player is resting on it.
const EYE_OFFSET = DEFAULTS.player.eyeHeight - BODY_REST_Y

export function FirstPersonController({ spawn }: Props) {
  const { camera } = useThree()
  const bodyRef = useRef<RapierRigidBody>(null)
  const keys = useRef<Record<string, boolean>>({})

  // Hoist per-frame vectors so useFrame does not allocate every tick.
  const forward = useMemo(() => new Vector3(), [])
  const right = useMemo(() => new Vector3(), [])
  const dir = useMemo(() => new Vector3(), [])
  const UP = useMemo(() => new Vector3(0, 1, 0), [])

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

    camera.getWorldDirection(forward)
    forward.y = 0
    forward.normalize()
    right.crossVectors(forward, UP).normalize()

    dir.set(0, 0, 0)
    if (keys.current['KeyW']) dir.add(forward)
    if (keys.current['KeyS']) dir.sub(forward)
    if (keys.current['KeyD']) dir.add(right)
    if (keys.current['KeyA']) dir.sub(right)

    if (dir.lengthSq() > 0) dir.normalize().multiplyScalar(DEFAULTS.player.walkSpeed)

    const linvel = body.linvel()
    body.setLinvel({ x: dir.x, y: linvel.y, z: dir.z }, true)

    const t = body.translation()
    camera.position.set(t.x, t.y + EYE_OFFSET, t.z)
  })

  // Body spawn Y is body-center, not eye-level. Caller passes the camera
  // spawn (eye-level); we translate down to body-center.
  const bodySpawn: [number, number, number] = [spawn[0], BODY_REST_Y, spawn[2]]

  return (
    <>
      <RigidBody
        ref={bodyRef}
        type="dynamic"
        position={bodySpawn}
        lockRotations
        colliders={false}
        mass={70}
      >
        <CapsuleCollider args={[CAPSULE_HALF_HEIGHT, DEFAULTS.player.capsuleRadius]} />
      </RigidBody>
      <PointerLockControls />
    </>
  )
}
