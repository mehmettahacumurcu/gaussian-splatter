import { Scene } from './Scene'

export function InteractivePage() {
  return (
    <div style={{ width: '100%', height: 'calc(100vh - 60px)', position: 'relative' }}>
      <Scene />
    </div>
  )
}
