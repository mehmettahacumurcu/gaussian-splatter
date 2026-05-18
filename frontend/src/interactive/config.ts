export const DEFAULTS = {
  player: {
    walkSpeed: 3.0,
    mouseSensitivity: 0.002,
    eyeHeight: 1.7,
    capsuleRadius: 0.4,
    capsuleHeight: 1.7,
  },
  pickup: {
    rayMaxDistance: 5.0,
    holdDistance: 1.5,
    holdSmoothing: 0.3,
    maxPickupMass: 20.0,
    throwForce: 8.0,
    heldOrientation: 'faceCamera' as 'faceCamera' | 'preserveOriginal',
  },
  physics: {
    gravity: [0, -9.81, 0] as [number, number, number],
    fixedTimestep: 1 / 60,
  },
  ui: {
    crosshairSize: 8,
    promptFadeMs: 150,
  },
} as const

export function validateMass(mass: number): number {
  if (!Number.isFinite(mass) || mass <= 0) {
    throw new Error(`Invalid mass: ${mass}; must be a positive finite number`)
  }
  return mass
}

export function canPickup(mass: number): boolean {
  return mass <= DEFAULTS.pickup.maxPickupMass
}
