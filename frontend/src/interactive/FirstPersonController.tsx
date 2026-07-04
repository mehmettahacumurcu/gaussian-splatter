import { PointerLockControls } from '@react-three/drei'
import { useFrame, useThree } from '@react-three/fiber'
import { CapsuleCollider, RigidBody, type RapierRigidBody } from '@react-three/rapier'
import { useEffect, useMemo, useRef } from 'react'
import { Vector3 } from 'three'
import { DEFAULTS } from './config'

interface Props {
  spawn: [number, number, number]
  flyMode?: boolean
  // groundPlane.y from the active world collider; used to teleport the player
  // back to spawn when re-enabling collision if they went underground in fly mode.
  groundY?: number
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

// Convert an eye-level Y coordinate to the corresponding physics body-center Y.
// Inverse of the per-frame camera update: camera.y = body.y + EYE_OFFSET.
// Exported for unit testing.
export function eyeToBody(eyeY: number): number {
  return eyeY - EYE_OFFSET
}

export function FirstPersonController({ spawn, flyMode = false, groundY }: Props) {
  const { camera } = useThree()
  const bodyRef = useRef<RapierRigidBody>(null)
  const keys = useRef<Record<string, boolean>>({})

  // Body spawn Y is body-center, not eye-level. Caller passes the camera
  // spawn (eye-level); we translate down to body-center.
  const bodySpawn: [number, number, number] = [spawn[0], eyeToBody(spawn[1]), spawn[2]]

  // Hoist per-frame vectors so useFrame does not allocate every tick.
  const forward = useMemo(() => new Vector3(), [])
  const forward3D = useMemo(() => new Vector3(), [])
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

  // Toggle fly-mode physics at runtime so the player's position is preserved
  // across the toggle. bodySpawn and spawn are stable for this component's
  // lifetime (the controller remounts whenever the spawn changes via key).
  useEffect(() => {
    const body = bodyRef.current
    if (!body) return
    if (flyMode) {
      body.setGravityScale(0, true)
      body.setLinvel({ x: 0, y: 0, z: 0 }, true)
      // Sensor mode lets the capsule pass through all world geometry.
      if (body.numColliders() > 0) body.collider(0).setSensor(true)
    } else {
      body.setGravityScale(1, true)
      if (body.numColliders() > 0) body.collider(0).setSensor(false)
      // If the player went underground while flying, teleport back to spawn
      // before gravity re-enables so they do not fall into the void.
      if (groundY !== undefined) {
        const t = body.translation()
        if (t.y < groundY) {
          body.setTranslation({ x: bodySpawn[0], y: bodySpawn[1], z: bodySpawn[2] }, true)
          body.setLinvel({ x: 0, y: 0, z: 0 }, true)
        }
      }
    }
  }, [flyMode, groundY]) // eslint-disable-line react-hooks/exhaustive-deps

  useFrame(() => {
    const body = bodyRef.current
    if (!body) return

    if (flyMode) {
      // --- Fly / noclip: full 3D movement along camera direction ---
      camera.getWorldDirection(forward3D)  // includes pitch
      right.crossVectors(forward3D, UP)
      // Guard against looking straight up/down (cross product → zero vector).
      if (right.lengthSq() < 1e-6) right.set(1, 0, 0)
      else right.normalize()

      dir.set(0, 0, 0)
      if (keys.current['KeyW']) dir.add(forward3D)
      if (keys.current['KeyS']) dir.sub(forward3D)
      if (keys.current['KeyD']) dir.add(right)
      if (keys.current['KeyA']) dir.sub(right)

      let vy = 0
      if (keys.current['Space']) vy += DEFAULTS.player.flySpeed
      if (keys.current['ShiftLeft']) vy -= DEFAULTS.player.flySpeed

      if (dir.lengthSq() > 0) dir.normalize().multiplyScalar(DEFAULTS.player.flySpeed)
      body.setLinvel({ x: dir.x, y: dir.y + vy, z: dir.z }, true)
    } else {
      // --- Normal grounded walk: horizontal movement only ---
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
    }

    const t = body.translation()
    camera.position.set(t.x, t.y + EYE_OFFSET, t.z)
  })

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
