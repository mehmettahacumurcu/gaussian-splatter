import { Canvas } from '@react-three/fiber'
import { Physics, type RapierRigidBody } from '@react-three/rapier'
import { Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { FirstPersonController } from './FirstPersonController'
import { StaticEnvironment } from './loaders/StaticEnvironment'
import { SplatBackground } from './loaders/SplatBackground'
import { WorldCollider, type WorldColliderData } from './loaders/WorldCollider'
import { DynamicObject, type DynamicObjectSpec } from './loaders/DynamicObject'
import { PickupSystem, type DynamicBodyEntry } from './PickupSystem'
import { DEFAULTS } from './config'
import type { PickupState } from './types'
import type { WorldEntry } from './WorldSelector'

const SPAWN_POSITION: [number, number, number] = [0, 1.7, 3]

const TEST_OBJECTS: DynamicObjectSpec[] = [
  {
    id: 'ball',
    shape: 'sphere',
    size: [0.12, 0, 0],
    position: [0, 1.5, 0],
    mass: 0.2,
    color: '#d94f4f',
  },
  {
    id: 'book',
    shape: 'box',
    size: [0.12, 0.02, 0.18],
    position: [0.5, 1.5, 0],
    mass: 0.5,
    color: '#2a6db5',
  },
  {
    id: 'mug',
    shape: 'cylinder',
    size: [0.05, 0.06, 0],
    position: [-0.5, 1.5, 0],
    mass: 0.3,
    color: '#e0e0d0',
  },
]

interface SceneProps {
  world?: WorldEntry | null
  onStateChange?: (s: PickupState) => void
}

export function Scene({ world, onStateChange }: SceneProps) {
  const bodies = useRef<Map<string, DynamicBodyEntry>>(new Map())
  const [colliderData, setColliderData] = useState<WorldColliderData | null>(null)

  useEffect(() => {
    if (!world) {
      setColliderData(null)
      return
    }
    let cancelled = false
    console.log(`[Scene] fetching collider ${world.colliderJsonUrl}`)
    fetch(world.colliderJsonUrl)
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`)
        return (await r.json()) as WorldColliderData
      })
      .then((data) => {
        if (!cancelled) {
          console.log('[Scene] collider loaded', data)
          setColliderData(data)
        }
      })
      .catch((e) => console.warn('[Scene] collider fetch failed; splat will render without physics walls:', e))
    return () => {
      cancelled = true
    }
  }, [world])

  const registerBody = useCallback((id: string, mass: number, body: RapierRigidBody | null) => {
    if (body) bodies.current.set(id, { body, mass })
    else bodies.current.delete(id)
  }, [])

  return (
    <Canvas
      camera={{ position: SPAWN_POSITION, fov: 75 }}
      style={{ width: '100%', height: '100%', background: '#101015' }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 10, 5]} intensity={1.0} />
      <Suspense fallback={null}>
        <Physics gravity={DEFAULTS.physics.gravity} timeStep={DEFAULTS.physics.fixedTimestep}>
          <FirstPersonController spawn={SPAWN_POSITION} />
          {world ? (
            <>
              {/* Visual splat — independent of collider, so it renders even
                  when the collider JSON 404s. */}
              <SplatBackground url={world.plyUrl} />
              {colliderData ? (
                <WorldCollider data={colliderData} />
              ) : (
                // Fallback: a flat ground plane at y=0 so the player has
                // something to stand on while the collider JSON loads or
                // if it fails entirely.
                <StaticEnvironment />
              )}
            </>
          ) : (
            <StaticEnvironment />
          )}
          {TEST_OBJECTS.map((o) => (
            <DynamicObject
              key={o.id}
              {...o}
              ref={(body) => registerBody(o.id, o.mass, body)}
            />
          ))}
          <PickupSystem dynamicBodies={bodies.current} onStateChange={onStateChange} />
        </Physics>
      </Suspense>
    </Canvas>
  )
}
