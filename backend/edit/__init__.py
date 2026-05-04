"""Object editing package — single-shot delete with AI inpaint + local refit.

Modules:
  sam_service       — SAM 2 wrapper (mask from click, video propagation)
  inpainter         — LaMa + SD inpainting backends with a common interface
  warp              — depth-warp anchor inpaints into target frames
  gauss_classifier  — projection-vote-threshold for Gaussian deletion
  refit             — local-region refit using inpainted views as new GT
  runner            — EditJobRunner: orchestrates the 5-phase edit pipeline

Spec: docs/superpowers/specs/2026-05-04-object-deletion-design.md
"""
