import { useFrame, useThree } from '@react-three/fiber'
import { useRapier, type RapierRigidBody } from '@react-three/rapier'
import { useEffect, useReducer, useRef } from 'react'
import { Vector3 } from 'three'
import { DEFAULTS } from './config'
import { initialPickupState, pickupReducer } from './reducer'
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

    // Raycast only in IDLE / AIMING_AT_OBJ
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

    // Hold logic (no throw yet — body type swap happens here defensively)
    if (current.kind === 'HOLDING') {
      const entry = dynamicBodies.get(current.targetId)
      if (entry) {
        if (entry.body.bodyType() !== 2) {
          entry.body.setBodyType(2, true) // 2 = kinematicPosition
        }
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
