import { Canvas } from '@react-three/fiber'
import { Physics } from '@react-three/rapier'
import { Suspense } from 'react'
import { FirstPersonController } from './FirstPersonController'
import { StaticEnvironment } from './loaders/StaticEnvironment'
import { DEFAULTS } from './config'

const SPAWN_POSITION: [number, number, number] = [0, 1.7, 3]

export function Scene() {
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
        </Physics>
      </Suspense>
    </Canvas>
  )
}
