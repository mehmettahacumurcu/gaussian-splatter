import type { Matrix4Tuple } from 'three'

export type Vec3 = [number, number, number]

export type ColliderType = 'convexHull' | 'trimesh' | 'box'

export interface InteractiveScene {
  splat?: { url: string; transform?: Matrix4Tuple }
  visual?: { url: string; transform?: Matrix4Tuple }
  collider: { url: string; transform?: Matrix4Tuple }
  dynamicObjects: Array<{
    url: string
    transform: Matrix4Tuple
    mass: number
    colliderType: ColliderType
  }>
  spawn: { position: Vec3; lookAt: Vec3 }
}

export type PickupState =
  | { kind: 'IDLE' }
  | { kind: 'AIMING_AT_OBJ'; targetId: string }
  | { kind: 'HOLDING'; targetId: string }
  | { kind: 'THROWING'; targetId: string }

export type PickupAction =
  | { type: 'RAY_HIT'; targetId: string; mass: number }
  | { type: 'RAY_MISS' }
  | { type: 'PRESS_E' }
  | { type: 'PRESS_G' }
  | { type: 'LEFT_CLICK' }
  | { type: 'THROW_COMPLETE' }
