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

const DEFAULT_SPAWN: [number, number, number] = [0, 1.7, 3]
// When a fixture world is loaded, spawn at the capture point (origin) facing
// +Z (into the scene) — that's where the input image was taken from, so the
// splat's actual content is right there.
const WORLD_SPAWN_FALLBACK: [number, number, number] = [0, 1.7, 0]

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
  flyMode?: boolean
}

export function Scene({ world, onStateChange, flyMode = false }: SceneProps) {
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

  // Spawn from collider JSON when present, else fallback. When NO world is
  // selected, stay at the historical D+E spawn so the D+E test scene still works.
  const effectiveSpawn: [number, number, number] = colliderData
    ? colliderData.spawn.position
    : world
      ? WORLD_SPAWN_FALLBACK
      : DEFAULT_SPAWN

  // Force-remount the controller when the spawn changes so the rigidbody
  // gets repositioned. Physics RigidBody only reads `position` at mount.
  // Including spawn coordinates in the key ensures a remount when the
  // collider JSON loads and delivers the real spawn (fixes the case where
  // WORLD_SPAWN_FALLBACK was used until collider data arrived).
  const controllerKey = world
    ? `world:${world.slug}:${effectiveSpawn.join(',')}`
    : 'default'

  return (
    <Canvas
      camera={{ position: effectiveSpawn, fov: 75 }}
      style={{ width: '100%', height: '100%', background: '#101015' }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 10, 5]} intensity={1.0} />
      <Suspense fallback={null}>
        <Physics gravity={DEFAULTS.physics.gravity} timeStep={DEFAULTS.physics.fixedTimestep}>
          <FirstPersonController
            key={controllerKey}
            spawn={effectiveSpawn}
            flyMode={flyMode}
            groundY={colliderData?.groundPlane.y}
          />
          {world ? (
            <>
              {/* Visual splat — independent of collider, so it renders even
                  when the collider JSON 404s. The worldRotation gravity-aligns
                  the raw splat frame once the collider JSON arrives. */}
              <SplatBackground
                url={world.plyUrl}
                quaternion={colliderData?.worldRotation?.quaternion}
              />
              {colliderData ? (
                <WorldCollider data={colliderData} />
              ) : (
                // Fallback: flat ground at y=0 + four invisible far walls so
                // the player doesn't fall infinitely while collider JSON loads.
                <StaticEnvironment />
              )}
            </>
          ) : (
            <StaticEnvironment />
          )}
          {/* Hide the D+E primitive test props when a fixture world is loaded;
              they were obscuring the splat. They stay in the built-in scene. */}
          {!world && TEST_OBJECTS.map((o) => (
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
