import { RigidBody, CuboidCollider } from '@react-three/rapier'

export interface WorldColliderData {
  schema_version: 1
  groundPlane: { y: number }
  boundingWalls: { xMin: number; xMax: number; zMin: number; zMax: number; yMax: number }
  spawn: { position: [number, number, number]; lookDirection: [number, number, number] }
  // Rotation (x, y, z, w) that maps the raw splat frame into the frame the
  // collider values above are expressed in (viewer space, +Y up). The splat
  // object must be rendered with this quaternion; absent on old bundles.
  worldRotation?: { quaternion: [number, number, number, number] }
}

interface Props {
  data: WorldColliderData
  wallThickness?: number
}

export function WorldCollider({ data, wallThickness = 0.5 }: Props) {
  const { groundPlane, boundingWalls } = data
  const xCenter = (boundingWalls.xMin + boundingWalls.xMax) / 2
  const zCenter = (boundingWalls.zMin + boundingWalls.zMax) / 2
  const xExtent = Math.max(0.1, (boundingWalls.xMax - boundingWalls.xMin) / 2)
  const zExtent = Math.max(0.1, (boundingWalls.zMax - boundingWalls.zMin) / 2)
  const wallHeight = Math.max(0.1, boundingWalls.yMax - groundPlane.y)
  const wallYCenter = groundPlane.y + wallHeight / 2

  return (
    <RigidBody type="fixed" colliders={false}>
      {/* Ground */}
      <CuboidCollider
        args={[xExtent + wallThickness, wallThickness / 2, zExtent + wallThickness]}
        position={[xCenter, groundPlane.y - wallThickness / 2, zCenter]}
      />
      {/* +X wall */}
      <CuboidCollider
        args={[wallThickness / 2, wallHeight / 2, zExtent]}
        position={[boundingWalls.xMax + wallThickness / 2, wallYCenter, zCenter]}
      />
      {/* -X wall */}
      <CuboidCollider
        args={[wallThickness / 2, wallHeight / 2, zExtent]}
        position={[boundingWalls.xMin - wallThickness / 2, wallYCenter, zCenter]}
      />
      {/* +Z wall */}
      <CuboidCollider
        args={[xExtent + wallThickness, wallHeight / 2, wallThickness / 2]}
        position={[xCenter, wallYCenter, boundingWalls.zMax + wallThickness / 2]}
      />
      {/* -Z wall */}
      <CuboidCollider
        args={[xExtent + wallThickness, wallHeight / 2, wallThickness / 2]}
        position={[xCenter, wallYCenter, boundingWalls.zMin - wallThickness / 2]}
      />
    </RigidBody>
  )
}
