import { BallCollider, CuboidCollider, CylinderCollider, RigidBody, type RapierRigidBody } from '@react-three/rapier'
import { forwardRef } from 'react'

export type PrimitiveShape = 'sphere' | 'box' | 'cylinder'

export interface DynamicObjectSpec {
  id: string
  shape: PrimitiveShape
  /**
   * For sphere: [radius, _, _] — only first element used.
   * For box: half-extents along each axis [hx, hy, hz].
   * For cylinder: [radius, halfHeight, _].
   */
  size: [number, number, number]
  position: [number, number, number]
  mass: number
  color: string
}

interface Props extends DynamicObjectSpec {}

export const DynamicObject = forwardRef<RapierRigidBody, Props>(function DynamicObject(
  { id, shape, size, position, mass, color },
  ref,
) {
  return (
    <RigidBody
      ref={ref}
      type="dynamic"
      mass={mass}
      position={position}
      colliders={false}
      userData={{ id, mass }}
    >
      {shape === 'sphere' && (
        <>
          <mesh>
            <sphereGeometry args={[size[0], 24, 16]} />
            <meshStandardMaterial color={color} />
          </mesh>
          <BallCollider args={[size[0]]} />
        </>
      )}
      {shape === 'box' && (
        <>
          <mesh>
            <boxGeometry args={[size[0] * 2, size[1] * 2, size[2] * 2]} />
            <meshStandardMaterial color={color} />
          </mesh>
          <CuboidCollider args={size} />
        </>
      )}
      {shape === 'cylinder' && (
        <>
          <mesh>
            <cylinderGeometry args={[size[0], size[0], size[1] * 2, 24]} />
            <meshStandardMaterial color={color} />
          </mesh>
          <CylinderCollider args={[size[1], size[0]]} />
        </>
      )}
    </RigidBody>
  )
})
