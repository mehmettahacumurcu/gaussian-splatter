import { Canvas } from '@react-three/fiber'
import { FirstPersonController } from './FirstPersonController'

export function Scene() {
  return (
    <Canvas
      camera={{ position: [0, 1.7, 3], fov: 75 }}
      style={{ width: '100%', height: '100%', background: '#101015' }}
    >
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 10, 5]} intensity={1.0} />
      <FirstPersonController />
      <mesh position={[0, 0, 0]}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial color="hotpink" />
      </mesh>
      <gridHelper args={[20, 20]} />
    </Canvas>
  )
}
