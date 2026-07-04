import { useMemo, useRef, useEffect } from 'react'
import { extend, useThree } from '@react-three/fiber'
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark'
import * as THREE from 'three'

// Register Spark classes as R3F primitives.
// Both SparkRenderer (extends THREE.Mesh) and SplatMesh (extends THREE.Object3D).
extend({ SparkRenderer, SplatMesh })

const ignoreRaycast: THREE.Object3D['raycast'] = () => {}

interface Props {
  url: string
  visible?: boolean
  // World rotation (x, y, z, w) from the bundle's collider JSON: maps the raw
  // splat frame into viewer space (+Y up). The collider is derived in the
  // rotated frame, so physics only lines up when this is applied to the splat.
  quaternion?: [number, number, number, number]
}

const IDENTITY_QUATERNION: [number, number, number, number] = [0, 0, 0, 1]

// Augment JSX intrinsics so TypeScript accepts the extended elements.
declare module '@react-three/fiber' {
  interface ThreeElements {
    sparkRenderer: any
    splatMesh: any
  }
}

export function SplatBackground({ url, visible = true, quaternion }: Props) {
  const renderer = useThree((state) => state.gl)
  const sparkRef = useRef<SparkRenderer>(null)
  const meshRef = useRef<SplatMesh>(null)

  // Memoize options objects to avoid re-creating the Spark objects on every render.
  const sparkArgs = useMemo<[ConstructorParameters<typeof SparkRenderer>[0]]>(
    () => [{ renderer }],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )
  const splatArgs = useMemo<[ConstructorParameters<typeof SplatMesh>[0]]>(
    () => [{ url }],
    [url],
  )

  // Keyed by url: a url change makes R3F construct a NEW SplatMesh (args
  // identity), so the new instance needs raycast disabled again.
  useEffect(() => {
    if (meshRef.current) meshRef.current.raycast = ignoreRaycast
    if (sparkRef.current) sparkRef.current.raycast = ignoreRaycast
  }, [url])

  // Explicit dispose of the outgoing SplatMesh (GPU buffers) on url change
  // and on unmount. Capture the instance in the effect body: React detaches
  // refs before passive cleanup runs, so `meshRef.current` is already null
  // inside the cleanup. dispose={null} below makes this the single owner.
  useEffect(() => {
    const mesh = meshRef.current
    return () => {
      mesh?.dispose()
    }
  }, [url])

  return (
    <sparkRenderer ref={sparkRef} args={sparkArgs} visible={visible}>
      <splatMesh
        ref={meshRef}
        args={splatArgs}
        quaternion={quaternion ?? IDENTITY_QUATERNION}
        dispose={null}
      />
    </sparkRenderer>
  )
}
