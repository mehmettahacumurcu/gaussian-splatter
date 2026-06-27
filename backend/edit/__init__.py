"""Inpainting backends (LaMa + Stable Diffusion) with a common interface.

After the strip-to-core cleanup the object-deletion edit pipeline
(sam_service / warp / gauss_classifier / refit / runner) was parked —
recoverable via the `milestone/edit-pipeline` tag. Only `inpainter`
remains in the trunk, because backend.image_to_scene (sub-project B)
depends on it (SDInpainter / InpainterBase).
"""
