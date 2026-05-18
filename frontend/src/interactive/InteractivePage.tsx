import { Scene } from './Scene'
import { HUD } from './HUD'

export function InteractivePage() {
  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Scene />
      <HUD />
    </div>
  )
}
