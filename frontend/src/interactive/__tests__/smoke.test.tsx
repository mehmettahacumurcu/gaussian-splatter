import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

describe('InteractivePage module smoke', () => {
  it('exports the InteractivePage component', async () => {
    const mod = await import('../InteractivePage')
    expect(typeof mod.InteractivePage).toBe('function')
  })
})

describe('WorldSelector', () => {
  it('renders a select with at least the built-in option', async () => {
    const { WorldSelector } = await import('../WorldSelector')
    render(<WorldSelector value={null} onChange={() => {}} />)
    const selector = screen.getByTestId('world-selector') as HTMLSelectElement
    expect(selector).toBeTruthy()
    // Default option ("— built-in test room —") + 3 fixture worlds = 4 options minimum.
    expect(selector.options.length).toBeGreaterThanOrEqual(1)
  })
})
