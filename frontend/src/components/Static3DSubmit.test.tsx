import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { Static3DSubmit } from './Static3DSubmit'

describe('Static3DSubmit', () => {
  it('renders all static runner presets including SOTA and Ultra', () => {
    render(<Static3DSubmit onJobSubmitted={() => {}} />)

    expect(screen.getByText(/SOTA/i)).toBeInTheDocument()
    expect(screen.getByText(/Ultra/i)).toBeInTheDocument()
  })
})
