import { describe, expect, it } from 'vitest'
import { DEFAULTS } from '../config'

// Import the exported pure function. The module constants are derived from
// DEFAULTS, so we reproduce them here to make test expectations self-contained.
const CAPSULE_HALF_HEIGHT = DEFAULTS.player.capsuleHeight / 2 - DEFAULTS.player.capsuleRadius
const BODY_REST_Y = DEFAULTS.player.capsuleRadius + CAPSULE_HALF_HEIGHT
const EYE_OFFSET = DEFAULTS.player.eyeHeight - BODY_REST_Y

describe('eyeToBody', () => {
  // We import the real function to validate against, while keeping the
  // expected-value derivation explicit so a DEFAULTS change breaks the test.
  let eyeToBody: (eyeY: number) => number

  beforeAll(async () => {
    const mod = await import('../FirstPersonController')
    eyeToBody = mod.eyeToBody
  })

  it('converts the default test-room eye-level spawn to the correct body-center Y', () => {
    // DEFAULT_SPAWN eye Y = 1.7 (floor at y=0, eye at eyeHeight above floor)
    // Expected body center = 1.7 - EYE_OFFSET = BODY_REST_Y
    const result = eyeToBody(1.7)
    expect(result).toBeCloseTo(BODY_REST_Y, 10)
  })

  it('body center is BODY_REST_Y above y=0 for the test-room spawn', () => {
    // Confirms the capsule rests on the y=0 StaticEnvironment floor.
    expect(eyeToBody(1.7)).toBeCloseTo(DEFAULTS.player.capsuleRadius + CAPSULE_HALF_HEIGHT, 10)
  })

  it('converts garden eye-level spawn to correct body-center Y', () => {
    // Garden: groundPlane.y = -3.21, spawn.position[1] = -1.51 (eye level)
    // Body center should be BODY_REST_Y above -3.21 = -3.21 + 0.85 = -2.36
    const gardenEyeY = -1.51
    const gardenGroundY = -3.21
    const result = eyeToBody(gardenEyeY)
    // Body center = eye - EYE_OFFSET
    expect(result).toBeCloseTo(gardenEyeY - EYE_OFFSET, 10)
    // And it should be BODY_REST_Y above the ground plane.
    expect(result - gardenGroundY).toBeCloseTo(BODY_REST_Y, 5)
  })

  it('is the exact inverse of the per-frame camera update (camera.y = body.y + EYE_OFFSET)', () => {
    const testCases = [0, 1.7, -1.51, 5.0, -10.0]
    for (const eyeY of testCases) {
      const bodyY = eyeToBody(eyeY)
      // Inverse: bodyY + EYE_OFFSET should equal eyeY
      expect(bodyY + EYE_OFFSET).toBeCloseTo(eyeY, 10)
    }
  })
})
