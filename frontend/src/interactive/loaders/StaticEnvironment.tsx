import { CuboidCollider, RigidBody } from '@react-three/rapier'

const ROOM_HALF = 4       // 8m × 8m room (each axis from -4 to +4)
const WALL_HEIGHT = 3
const WALL_THICK = 0.2

export function StaticEnvironment() {
  return (
    <RigidBody type="fixed" colliders={false}>
      {/* Floor */}
      <mesh position={[0, -WALL_THICK / 2, 0]} receiveShadow>
        <boxGeometry args={[ROOM_HALF * 2, WALL_THICK, ROOM_HALF * 2]} />
        <meshStandardMaterial color="#6b5a4a" />
      </mesh>
      <CuboidCollider args={[ROOM_HALF, WALL_THICK / 2, ROOM_HALF]} position={[0, -WALL_THICK / 2, 0]} />

      {/* North wall (-Z) */}
      <mesh position={[0, WALL_HEIGHT / 2, -ROOM_HALF]}>
        <boxGeometry args={[ROOM_HALF * 2, WALL_HEIGHT, WALL_THICK]} />
        <meshStandardMaterial color="#8a7a6a" />
      </mesh>
      <CuboidCollider args={[ROOM_HALF, WALL_HEIGHT / 2, WALL_THICK / 2]} position={[0, WALL_HEIGHT / 2, -ROOM_HALF]} />

      {/* South wall (+Z) */}
      <mesh position={[0, WALL_HEIGHT / 2, ROOM_HALF]}>
        <boxGeometry args={[ROOM_HALF * 2, WALL_HEIGHT, WALL_THICK]} />
        <meshStandardMaterial color="#8a7a6a" />
      </mesh>
      <CuboidCollider args={[ROOM_HALF, WALL_HEIGHT / 2, WALL_THICK / 2]} position={[0, WALL_HEIGHT / 2, ROOM_HALF]} />

      {/* West wall (-X) */}
      <mesh position={[-ROOM_HALF, WALL_HEIGHT / 2, 0]}>
        <boxGeometry args={[WALL_THICK, WALL_HEIGHT, ROOM_HALF * 2]} />
        <meshStandardMaterial color="#7a6a5a" />
      </mesh>
      <CuboidCollider args={[WALL_THICK / 2, WALL_HEIGHT / 2, ROOM_HALF]} position={[-ROOM_HALF, WALL_HEIGHT / 2, 0]} />

      {/* East wall (+X) */}
      <mesh position={[ROOM_HALF, WALL_HEIGHT / 2, 0]}>
        <boxGeometry args={[WALL_THICK, WALL_HEIGHT, ROOM_HALF * 2]} />
        <meshStandardMaterial color="#7a6a5a" />
      </mesh>
      <CuboidCollider args={[WALL_THICK / 2, WALL_HEIGHT / 2, ROOM_HALF]} position={[ROOM_HALF, WALL_HEIGHT / 2, 0]} />
    </RigidBody>
  )
}
