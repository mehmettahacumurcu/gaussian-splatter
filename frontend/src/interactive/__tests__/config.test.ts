import { describe, expect, it } from 'vitest'
import { DEFAULTS, canPickup, validateMass } from '../config'

describe('config', () => {
  describe('validateMass', () => {
    it('passes through positive finite mass', () => {
      expect(validateMass(0.3)).toBe(0.3)
      expect(validateMass(15)).toBe(15)
    })

    it('rejects zero', () => {
      expect(() => validateMass(0)).toThrow(/Invalid mass/)
    })

    it('rejects negative', () => {
      expect(() => validateMass(-1)).toThrow(/Invalid mass/)
    })

    it('rejects NaN', () => {
      expect(() => validateMass(NaN)).toThrow(/Invalid mass/)
    })

    it('rejects Infinity', () => {
      expect(() => validateMass(Infinity)).toThrow(/Invalid mass/)
    })
  })

  describe('canPickup', () => {
    it('allows mass at the boundary', () => {
      expect(canPickup(DEFAULTS.pickup.maxPickupMass)).toBe(true)
    })

    it('allows light mass', () => {
      expect(canPickup(0.3)).toBe(true)
    })

    it('rejects mass above limit', () => {
      expect(canPickup(DEFAULTS.pickup.maxPickupMass + 0.01)).toBe(false)
    })

    it('rejects very heavy mass', () => {
      expect(canPickup(1000)).toBe(false)
    })
  })
})
