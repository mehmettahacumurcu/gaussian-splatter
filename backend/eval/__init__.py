"""4D Quality v6.1 — Evaluation framework.

End-of-training evaluation:
  * nvs_eval — held-out cam metrics (PSNR/SSIM/LPIPS) — Madde 1+8
  * orbit_render — Catmull-Rom spline cam path → smooth orbit render
  * video_export — frames → mp4 (imageio/ffmpeg)
"""
