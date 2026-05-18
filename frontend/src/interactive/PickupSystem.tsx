import { useFrame, useThree } from '@react-three/fiber'
import { useRapier, type RapierRigidBody } from '@react-three/rapier'
import { useEffect, useReducer, useRef } from 'react'
import { Vector3 } from 'three'
import { DEFAULTS } from './config'
import { computeThrowVelocity, initialPickupState, pickupReducer } from './reducer'
import type { PickupState } from './types'

export interface DynamicBodyEntry {
  body: RapierRigidBody
  mass: number
}

interface Props {
  dynamicBodies: Map<string, DynamicBodyEntry>
  onStateChange?: (state: PickupState) => void
}

function findByHandle(
  bodies: Map<string, DynamicBodyEntry>,
  handle: number,
): { id: string; entry: DynamicBodyEntry } | null {
  for (const [id, entry] of bodies.entries()) {
    if (entry.body.handle === handle) return { id, entry }
  }
  return null
}

export function PickupSystem({ dynamicBodies, onStateChange }: Props) {
  const { camera } = useThree()
  const { world, rapier } = useRapier()
  const [state, dispatch] = useReducer(pickupReducer, initialPickupState)
  const prevStateRef = useRef<PickupState>(initialPickupState)
  const stateRef = useRef<PickupState>(state)

  useEffect(() => {
    stateRef.current = state
    onStateChange?.(state)
  }, [state, onStateChange])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === 'KeyE') dispatch({ type: 'PRESS_E' })
      if (e.code === 'KeyG') dispatch({ type: 'PRESS_G' })
    }
    const onClick = (e: MouseEvent) => {
      if (e.button === 0 && document.pointerLockElement) {
        dispatch({ type: 'LEFT_CLICK' })
      }
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('mousedown', onClick)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('mousedown', onClick)
    }
  }, [])

  useFrame(() => {
    const current = stateRef.current
    const prev = prevStateRef.current

    // --- Transitions: set body type when entering/leaving HOLDING ---
    if (prev.kind !== 'HOLDING' && current.kind === 'HOLDING') {
      const entry = dynamicBodies.get(current.targetId)
      if (entry) entry.body.setBodyType(2, true) // 2 = kinematicPosition
    }
    if (prev.kind === 'HOLDING' && current.kind === 'IDLE') {
      // Drop — body type back to dynamic, no initial velocity
      const entry = dynamicBodies.get(prev.targetId)
      if (entry) {
        entry.body.setBodyType(0, true) // 0 = dynamic
        entry.body.setLinvel({ x: 0, y: 0, z: 0 }, true)
        entry.body.setAngvel({ x: 0, y: 0, z: 0 }, true)
      }
    }
    if (prev.kind === 'HOLDING' && current.kind === 'THROWING') {
      // Throw — body type back to dynamic, apply forward velocity
      const entry = dynamicBodies.get(prev.targetId)
      if (entry) {
        entry.body.setBodyType(0, true)
        const forward = new Vector3()
        camera.getWorldDirection(forward)
        const [vx, vy, vz] = computeThrowVelocity({
          forward: [forward.x, forward.y, forward.z],
          throwForce: DEFAULTS.pickup.throwForce,
        })
        entry.body.setLinvel({ x: vx, y: vy, z: vz }, true)
        entry.body.setAngvel({
          x: (Math.random() - 0.5),
          y: (Math.random() - 0.5),
          z: (Math.random() - 0.5),
        }, true)
      }
      // Auto-advance THROWING → IDLE on next tick
      requestAnimationFrame(() => dispatch({ type: 'THROW_COMPLETE' }))
    }

    prevStateRef.current = current

    // --- Per-frame logic ---
    if (current.kind === 'IDLE' || current.kind === 'AIMING_AT_OBJ') {
      const dir = new Vector3()
      camera.getWorldDirection(dir)
      const ray = new rapier.Ray(
        { x: camera.position.x, y: camera.position.y, z: camera.position.z },
        { x: dir.x, y: dir.y, z: dir.z },
      )
      const hit = world.castRay(ray, DEFAULTS.pickup.rayMaxDistance, true)
      const hitBody = hit?.collider.parent()
      if (hitBody) {
        const found = findByHandle(dynamicBodies, hitBody.handle)
        if (found) {
          dispatch({ type: 'RAY_HIT', targetId: found.id, mass: found.entry.mass })
        } else {
          dispatch({ type: 'RAY_MISS' })
        }
      } else {
        dispatch({ type: 'RAY_MISS' })
      }
    }

    if (current.kind === 'HOLDING') {
      const entry = dynamicBodies.get(current.targetId)
      if (entry) {
        const target = new Vector3()
        camera.getWorldDirection(target)
        target.multiplyScalar(DEFAULTS.pickup.holdDistance).add(camera.position)
        const cur = entry.body.translation()
        const next = new Vector3(cur.x, cur.y, cur.z).lerp(target, DEFAULTS.pickup.holdSmoothing)
        entry.body.setNextKinematicTranslation({ x: next.x, y: next.y, z: next.z })
      }
    }
  })

  return null
}
