import { canPickup } from './config'
import type { PickupAction, PickupState } from './types'

export const initialPickupState: PickupState = { kind: 'IDLE' }

export function pickupReducer(state: PickupState, action: PickupAction): PickupState {
  switch (state.kind) {
    case 'IDLE':
      if (action.type === 'RAY_HIT' && canPickup(action.mass)) {
        return { kind: 'AIMING_AT_OBJ', targetId: action.targetId }
      }
      return state

    case 'AIMING_AT_OBJ':
      if (action.type === 'RAY_MISS') {
        return { kind: 'IDLE' }
      }
      if (action.type === 'RAY_HIT') {
        if (!canPickup(action.mass)) {
          return { kind: 'IDLE' }
        }
        return { kind: 'AIMING_AT_OBJ', targetId: action.targetId }
      }
      if (action.type === 'PRESS_E') {
        return { kind: 'HOLDING', targetId: state.targetId }
      }
      return state

    case 'HOLDING':
      if (action.type === 'LEFT_CLICK') {
        return { kind: 'THROWING', targetId: state.targetId }
      }
      if (action.type === 'PRESS_G') {
        return { kind: 'IDLE' }
      }
      return state

    case 'THROWING':
      if (action.type === 'THROW_COMPLETE') {
        return { kind: 'IDLE' }
      }
      return state

    default: {
      const _exhaustive: never = state
      return _exhaustive
    }
  }
}

export function computeThrowVelocity(args: {
  forward: [number, number, number]
  throwForce: number
}): [number, number, number] {
  const [fx, fy, fz] = args.forward
  const f = args.throwForce
  return [fx * f, fy * f, fz * f]
}
