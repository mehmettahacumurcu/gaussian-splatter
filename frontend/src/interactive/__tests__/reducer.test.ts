import { describe, expect, it } from 'vitest'
import { computeThrowVelocity, initialPickupState, pickupReducer } from '../reducer'

describe('pickupReducer', () => {
  it('starts in IDLE', () => {
    expect(initialPickupState).toEqual({ kind: 'IDLE' })
  })

  it('IDLE → AIMING_AT_OBJ on RAY_HIT with pickable mass', () => {
    const next = pickupReducer({ kind: 'IDLE' }, { type: 'RAY_HIT', targetId: 'mug', mass: 0.3 })
    expect(next).toEqual({ kind: 'AIMING_AT_OBJ', targetId: 'mug' })
  })

  it('IDLE stays IDLE on RAY_HIT with too-heavy mass', () => {
    const next = pickupReducer({ kind: 'IDLE' }, { type: 'RAY_HIT', targetId: 'piano', mass: 500 })
    expect(next).toEqual({ kind: 'IDLE' })
  })

  it('IDLE stays IDLE on RAY_MISS', () => {
    const next = pickupReducer({ kind: 'IDLE' }, { type: 'RAY_MISS' })
    expect(next).toEqual({ kind: 'IDLE' })
  })
})

describe('AIMING_AT_OBJ transitions', () => {
  const aiming = { kind: 'AIMING_AT_OBJ' as const, targetId: 'mug' }

  it('AIMING_AT_OBJ → IDLE on RAY_MISS', () => {
    expect(pickupReducer(aiming, { type: 'RAY_MISS' })).toEqual({ kind: 'IDLE' })
  })

  it('AIMING_AT_OBJ updates target on RAY_HIT for different object', () => {
    const next = pickupReducer(aiming, { type: 'RAY_HIT', targetId: 'book', mass: 0.5 })
    expect(next).toEqual({ kind: 'AIMING_AT_OBJ', targetId: 'book' })
  })

  it('AIMING_AT_OBJ → IDLE if newly aimed object is too heavy', () => {
    const next = pickupReducer(aiming, { type: 'RAY_HIT', targetId: 'piano', mass: 500 })
    expect(next).toEqual({ kind: 'IDLE' })
  })

  it('AIMING_AT_OBJ → HOLDING on PRESS_E', () => {
    expect(pickupReducer(aiming, { type: 'PRESS_E' })).toEqual({ kind: 'HOLDING', targetId: 'mug' })
  })

  it('AIMING_AT_OBJ ignores PRESS_G', () => {
    expect(pickupReducer(aiming, { type: 'PRESS_G' })).toEqual(aiming)
  })

  it('AIMING_AT_OBJ ignores LEFT_CLICK', () => {
    expect(pickupReducer(aiming, { type: 'LEFT_CLICK' })).toEqual(aiming)
  })
})

describe('HOLDING transitions', () => {
  const holding = { kind: 'HOLDING' as const, targetId: 'mug' }

  it('HOLDING → THROWING on LEFT_CLICK', () => {
    expect(pickupReducer(holding, { type: 'LEFT_CLICK' })).toEqual({ kind: 'THROWING', targetId: 'mug' })
  })

  it('HOLDING → IDLE on PRESS_G (drop)', () => {
    expect(pickupReducer(holding, { type: 'PRESS_G' })).toEqual({ kind: 'IDLE' })
  })

  it('HOLDING ignores RAY_HIT (raycast does not change target while held)', () => {
    expect(pickupReducer(holding, { type: 'RAY_HIT', targetId: 'book', mass: 0.5 })).toEqual(holding)
  })

  it('HOLDING ignores RAY_MISS', () => {
    expect(pickupReducer(holding, { type: 'RAY_MISS' })).toEqual(holding)
  })

  it('HOLDING ignores PRESS_E (already held)', () => {
    expect(pickupReducer(holding, { type: 'PRESS_E' })).toEqual(holding)
  })
})

describe('THROWING transitions', () => {
  const throwing = { kind: 'THROWING' as const, targetId: 'mug' }

  it('THROWING → IDLE on THROW_COMPLETE', () => {
    expect(pickupReducer(throwing, { type: 'THROW_COMPLETE' })).toEqual({ kind: 'IDLE' })
  })

  it('THROWING ignores other actions', () => {
    expect(pickupReducer(throwing, { type: 'RAY_HIT', targetId: 'book', mass: 0.5 })).toEqual(throwing)
    expect(pickupReducer(throwing, { type: 'PRESS_E' })).toEqual(throwing)
    expect(pickupReducer(throwing, { type: 'PRESS_G' })).toEqual(throwing)
  })
})

describe('computeThrowVelocity', () => {
  it('multiplies camera forward by throwForce', () => {
    const result = computeThrowVelocity({ forward: [0, 0, -1], throwForce: 8 })
    expect(result).toEqual([0, 0, -8])
  })

  it('handles arbitrary forward direction', () => {
    const result = computeThrowVelocity({ forward: [1, 0, 0], throwForce: 5 })
    expect(result).toEqual([5, 0, 0])
  })

  it('handles diagonal forward', () => {
    const inv = 1 / Math.sqrt(2)
    const result = computeThrowVelocity({ forward: [inv, 0, -inv], throwForce: 10 })
    expect(result[0]).toBeCloseTo(10 * inv, 5)
    expect(result[1]).toBeCloseTo(0, 5)
    expect(result[2]).toBeCloseTo(-10 * inv, 5)
  })
})
