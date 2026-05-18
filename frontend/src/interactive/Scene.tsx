import { Canvas } from '@react-three/fiber'
import { Physics, type RapierRigidBody } from '@react-three/rapier'
import { Suspense, useCallback, useRef } from 'react'
import { FirstPersonController } from './FirstPersonController'
import { StaticEnvironment } from './loaders/StaticEnvironment'
import { DynamicObject, type DynamicObjectSpec } from './loaders/DynamicObject'
import { PickupSystem, type DynamicBodyEntry } from './PickupSystem'
import { DEFAULTS } from './config'
import type { PickupState } from './types'

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

export function Scene({ onStateChange }: { onStateChange?: (s: PickupState) => void }) {
  const bodies = useRef<Map<string, DynamicBodyEntry>>(new Map())

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
          <StaticEnvironment />
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
