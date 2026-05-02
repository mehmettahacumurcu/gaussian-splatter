"""Faz 7 — FastAPI backend.

Video upload → background pipeline → .ply indirme.

Çalıştırma:
    uvicorn backend.api:app --host 127.0.0.1 --port 8000 --reload

Swagger UI:
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations
import io
import json
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from .api_models import (
    HealthResponse,
    Job,
    JobListResponse,
    JobStatus,
    ProcessResponse,
)
from .config import default_config, cloud_config, scene_paths
from .job_manager import JobManager, get_manager
from .pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Uygulama yaşam döngüsü
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup + shutdown hook'ları."""
    manager = get_manager()
    app.state.manager = manager
    print("[api] JobManager hazır (max_workers=1, GPU sıralaması aktif)")
    try:
        yield
    finally:
        print("[api] Shutdown — executor kapatılıyor (running job'lar bitene kadar beklenir)")
        manager.shutdown(wait=True)


app = FastAPI(
    title="4DGS Studio Backend",
    description="Video → 4D Gaussian Splatting pipeline (Faz 7 HTTP katmanı).",
    version="0.1.0",
    lifespan=lifespan,
)


# CORS — Tauri / localhost web client'ları için
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:*",
        "http://127.0.0.1:*",
        "tauri://localhost",
        "https://tauri.localhost",
    ],
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|tauri://localhost|https://tauri\.localhost)$",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Servis sağlığı + GPU durumu."""
    manager: JobManager = app.state.manager
    gpu_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_ok else None
    return HealthResponse(
        gpu_available=gpu_ok,
        gpu_name=gpu_name,
        active_jobs=manager.active_count(),
    )


# ---------------------------------------------------------------------------
# Process — yeni job oluştur
# ---------------------------------------------------------------------------
@app.post("/process", response_model=ProcessResponse, status_code=202, tags=["jobs"])
async def process_video(
    video: UploadFile | None = File(None, description="Girdi video dosyası (mp4/mov). Static modda OPSIYONEL — sahne klasoründe images/ varsa video gerekmez."),
    scene: str = Form("unnamed_scene", description="Sahne ismi — data/<scene>/ altında çalışılır"),
    # v6.0 — Mode + preset (single source of truth)
    mode: str = Form("dynamic", description="'static' (Static 3DGS) | 'dynamic' (4D, default)"),
    preset: str | None = Form(None, description="Mode'a göre preset adı. Static: fast/balanced/high/premium. Dynamic: micro/smoke/full/high/cloud/ultra/ultra_clean/static_max."),
    # Legacy boolean preset flag'leri (geriye uyumluluk — yeni clientlar mode+preset gönderir)
    smoke_test: bool = Form(False, description="[LEGACY] use mode='dynamic' + preset='smoke'"),
    micro_test: bool = Form(False, description="[LEGACY] use mode='dynamic' + preset='micro'"),
    cloud: bool = Form(False, description="[LEGACY] use mode='dynamic' + preset='cloud'"),
    high_test: bool = Form(False, description="[LEGACY] use mode='dynamic' + preset='high'"),
    ultra_test: bool = Form(False, description="[LEGACY] use mode='dynamic' + preset='ultra'"),
    ultra_clean: bool = Form(False, description="[LEGACY] use mode='dynamic' + preset='ultra_clean'"),
    static_max: bool = Form(False, description="[LEGACY] use mode='dynamic' + preset='static_max'"),
    skip_foundation: bool = Form(False, description="Foundation modellerini atla. Default False — foundation (depth/masks/flow/tracks) calistirilir; True verirsen atlanir (sadece smoke / mekanik test icin)."),
    # --- Override parametreleri (preset üzerine uygulanır) ---
    # Temel training
    iters: int | None = Form(None, description="Override training iter sayısı"),
    resolution: str | None = Form(None, description='"WxH" format, örn. "640x360"'),
    num_timestamps: int | None = Form(None, description="Export edilecek timestamp sayısı"),
    fps: int | None = Form(None, description="Frame extraction FPS"),
    # Loss weights
    lambda_ssim: float | None = Form(None, description="SSIM katkı oranı (0-1)"),
    lambda_deform_reg: float | None = Form(None, description="Deformation L2 reg"),
    lambda_smoothness: float | None = Form(None, description="Temporal smoothness"),
    lambda_rigidity: float | None = Form(None, description="Isometric rigidity"),
    lambda_depth: float | None = Form(None, description="Depth consistency (Metric3D)"),
    lambda_mask_motion: float | None = Form(None, description="Dynamic mask weight"),
    lambda_track: float | None = Form(None, description="CoTracker 3D-anchored track loss"),
    lambda_scale: float | None = Form(None, description="Scale regularizer (outlier blow-up önleme)"),
    lambda_aniso: float | None = Form(None, description="v3.8: Anisotropy regularizer (streak/needle gaussian fix)"),
    aniso_threshold: float | None = Form(None, description="v3.8: max/min scale ratio threshold (default 5)"),
    dpos_total_cap_frac: float | None = Form(None, description="v3.8: Per-iter dpos clamp × scene_extent (default 0.2)"),
    opacity_reset_interval: int | None = Form(None, description="Opacity reset aralığı (iter), 0=kapalı"),
    track_sample_k: int | None = Form(None, description="Her iter sample edilecek track sayısı"),
    warmup_iters: int | None = Form(None, description="Regularizer warmup süresi (iter)"),
    # Learning rates
    lr_deform: float | None = Form(None, description="Deformation field LR"),
    lr_means: float | None = Form(None, description="Gaussian means LR"),
    # Density control
    density_start_iter: int | None = Form(None, description="Density control başlangıç iter"),
    density_end_iter: int | None = Form(None, description="Density control bitiş iter"),
    density_interval: int | None = Form(None, description="Kaç iter'de bir clone/split/prune"),
    densify_grad_threshold: float | None = Form(None, description="Densify grad eşiği"),
    prune_min_opacity: float | None = Form(None, description="Prune: opacity altı"),
    prune_max_scale: float | None = Form(None, description="Prune: scale üstü"),
    # Model mimarisi
    sh_degree: int | None = Form(None, description="Spherical Harmonics derecesi (0-3)"),
    hexplane_resolution: int | None = Form(None, description="HexPlane grid çözünürlük"),
    hexplane_feat_dim: int | None = Form(None, description="HexPlane feature dim"),
    mlp_width: int | None = Form(None, description="Deformation MLP genişlik"),
    mlp_depth: int | None = Form(None, description="Deformation MLP hidden layer sayısı"),
    num_time_freqs: int | None = Form(None, description="Fourier time encoding frekans sayısı"),
    # v3.6 / Yol C — Per-gaussian Fourier trajectory
    deform_pos_mode: str | None = Form(None, description="mlp | fourier | hybrid (default: hybrid)"),
    fourier_K: int | None = Form(None, description="Per-gaussian Fourier trajectory frekans sayısı (default 8)"),
    lr_fourier: float | None = Form(None, description="Fourier coefficient LR"),
    lambda_fourier_reg: float | None = Form(None, description="Fourier high-freq L2 reg"),
    # Foundation models (Faz 3)
    metric3d_model: str | None = Form(None, description="metric3d_vit_small | _large | _giant2"),
    cotracker_num_points: int | None = Form(None, description="CoTracker nokta sayısı"),
    cotracker_grid_size: int | None = Form(None, description="CoTracker grid NxN"),
    sam2_threshold: float | None = Form(None, description="SAM2 confidence eşiği"),
    # v3.9 — Preprocessing
    resize_long_edge: int | None = Form(None, description="Frame extract long edge px (default 960)"),
    colmap_matching: str | None = Form(None, description="sequential | exhaustive (default sequential)"),
    init_subsample_mode: str | None = Form(None, description="random | confidence (default random)"),
    # v5.0 — Multi-view bootstrap COLMAP
    colmap_mv_timestamps: int | None = Form(None, description="Multi-view bootstrap COLMAP: kac timestep (1, 5, 10, 25). Default 5."),
    colmap_mv_dense_mvs: bool | None = Form(None, description="Multi-view: MVS dense reconstruction (premium, +30-90 dk, 500k-2M dense point)"),
    # Phase 1.1 — Cache infrastructure
    force_preprocess: bool = Form(False, description="True: tum cache'leri yoksay, preprocessing baştan kosulsun (frames/COLMAP/depth/tracks/masks). Default False — heavy preprocess once, train many times."),
    # New SOTA-tier knobs (added 2026-05-02)
    lambda_lpips: float | None = Form(None, description="LPIPS perceptual loss weight"),
    lpips_net: str | None = Form(None, description="'alex' or 'vgg' for LPIPS backbone"),
    lambda_flow: float | None = Form(None, description="Optical flow loss weight (requires foundation)"),
    lambda_multiview_consistency: float | None = Form(None, description="Multi-view cross-cam supervision (4D MV only)"),
    lr_cam_K: float | None = Form(None, description="Joint BA on intrinsics (1e-7 typical)"),
    lr_cam_w2c: float | None = Form(None, description="Joint BA on extrinsics (1e-7 typical)"),
    cam_refine_start_iter: int | None = Form(None, description="Iter to start joint BA"),
    multires_schedule: str | None = Form(None, description='Comma-separated "iter:long_edge" pairs, e.g., "0:480,10000:720,25000:1080"'),
) -> ProcessResponse:
    """
    Video'yu upload et ve pipeline'ı kuyruğa al.

    Response: 202 Accepted + job_id. İlerleme için /status/{job_id}'yi poll et.

    v6.0: mode + preset alanlari (single source of truth). Eski boolean flag'ler
    hala desteklenir (legacy clientlar) ama yeni client'lar sadece mode+preset
    gondermeli.
    """
    # Sahne adı sanitize — path traversal engelle
    safe_scene = _safe_scene_name(scene)
    paths = scene_paths(safe_scene)
    paths["base"].mkdir(parents=True, exist_ok=True)

    # ---- v6.0 mode + preset → legacy boolean translation ----
    # Eski boolean flag'leri override eder. Mode='static' verilirse
    # static_mode_active True yapilir, alt presetlerle birlikte islenir.
    mode_norm = (mode or "dynamic").strip().lower()
    if mode_norm not in ("static", "dynamic"):
        raise HTTPException(400, f"mode '{mode}' invalid — use 'static' or 'dynamic'")
    static_mode_active = (mode_norm == "static")
    # Default local_max preset flag — preset='local_max' verilmedikce False kalir
    local_max = False
    if preset:
        preset_lc = preset.strip().lower()
        if static_mode_active:
            valid_static = {"fast", "balanced", "high", "premium"}
            if preset_lc not in valid_static:
                raise HTTPException(400, f"static preset '{preset}' invalid — use one of {sorted(valid_static)}")
            # Static preset'leri legacy boolean'a translate ETMIYORUZ;
            # asagida static-only preset uygulayicisi ile islenir.
        else:
            # Dynamic mode — preset string'i legacy boolean'a cevir
            mapping = {
                "micro": "micro_test",
                "smoke": "smoke_test",
                "full": None,  # default config, ek flag yok
                "high": "high_test",
                "cloud": "cloud",
                "ultra": "ultra_test",
                "ultra_clean": "ultra_clean",
                "static_max": "static_max",
                "local_max": "local_max",
            }
            if preset_lc not in mapping:
                raise HTTPException(400, f"dynamic preset '{preset}' invalid — use one of {sorted(mapping.keys())}")
            target = mapping[preset_lc]
            # Tum legacy flag'leri False'a cek, sadece target'i True
            smoke_test = (target == "smoke_test")
            micro_test = (target == "micro_test")
            high_test = (target == "high_test")
            ultra_test = (target == "ultra_test")
            ultra_clean = (target == "ultra_clean")
            static_max = (target == "static_max")
            cloud = (target == "cloud")
            local_max = (target == "local_max")

    # ---- Video upload ----
    video_path = paths["base"] / "video.mp4"
    if video is not None and video.filename:
        # Chunked write (büyük dosya support)
        try:
            with open(video_path, "wb") as f:
                while chunk := await video.read(1024 * 1024):  # 1 MB
                    f.write(chunk)
        except Exception as e:
            raise HTTPException(500, f"Video yazılamadı: {e}")
    else:
        # Static mode + photo set / pre-loaded scene durumu — video gerekmez,
        # ama sahne klasorunde frames/ veya images/ olmali.
        if not static_mode_active:
            raise HTTPException(400, "Dynamic mode'da video upload zorunludur")
        has_images = (paths["base"] / "images").exists()
        has_frames = paths["frames"].exists() and any(paths["frames"].glob("frame_*.png"))
        has_video = video_path.exists()
        if not (has_images or has_frames or has_video):
            raise HTTPException(
                400,
                f"Static mode'da video yok ve {paths['base']}/ altinda images/, frames/ veya video.mp4 hiçbiri bulunamadı. "
                f"Photo set kullaniyorsan: data/{safe_scene}/images/IMG_*.jpg klasorunu hazirla."
            )

    # Job kaydını oluştur
    manager: JobManager = app.state.manager
    job = manager.create(scene=safe_scene, smoke_test=smoke_test)

    # Runner kapanı — pipeline.run_pipeline'ı callback'le çağırır
    def _runner(on_progress: Any) -> dict:
        if cloud:
            cfg = cloud_config()
        elif local_max:
            # Inline import — local_max_config() companion config branch'inde eklenir.
            # Top-level import yapmiyoruz ki branch merge edilmemisken sadece bu
            # preset patlasin, tum api.py degil.
            from backend.config import local_max_config
            cfg = local_max_config()
        else:
            cfg = default_config()

        # ---- v6.0 Static 3DGS preset uygulayicisi ----
        # mode='static' verildiyse Static 3DGS preset'leri kosulur, sonra
        # dynamic preset blok'lari (smoke_test/...) atlanir.
        if static_mode_active:
            from scripts.static_3dgs import _apply_preset as _apply_static_preset
            preset_name = (preset or "balanced").strip().lower()
            if preset_name not in ("fast", "balanced", "high", "premium"):
                preset_name = "balanced"
            _apply_static_preset(cfg, preset_name)
            print(f"[api.static] preset={preset_name}, static_mode={cfg.train.static_mode}, "
                  f"n_iters={cfg.train.n_iters}, max_gauss={cfg.train.max_gaussians}")
            # Static modda foundation otomatik atlanir, ancak skip_foundation flag'i
            # dogrudan run_pipeline'a gecirilir. Pipeline'da zaten static_mode=True
            # ise foundation skip + 4D lambda zero-out yapiliyor, double safety.
            effective_skip_foundation = True

            # Apply user hyperparam overrides on top of static preset
            # (en azindan training-shared olanlari; preset-specific olanlar
            # static-friendly olmaya zorlanir).
            if iters is not None:
                cfg.train.n_iters = iters
            if resolution:
                try:
                    w_str, h_str = resolution.lower().split("x")
                    cfg.train.image_resolution = (int(w_str), int(h_str))
                except ValueError as e:
                    raise HTTPException(400, f"resolution formati: WxH ({e})")
            if num_timestamps is not None:
                cfg.export.num_timestamps = max(1, num_timestamps)
            if lambda_ssim is not None:
                cfg.train.lambda_ssim = lambda_ssim
            if lambda_scale is not None:
                cfg.train.lambda_scale = lambda_scale
            if sh_degree is not None:
                cfg.model.sh_degree = sh_degree
            if density_start_iter is not None:
                cfg.train.density_start_iter = density_start_iter
            if density_end_iter is not None:
                cfg.train.density_end_iter = density_end_iter
            if densify_grad_threshold is not None:
                cfg.train.densify_grad_threshold = densify_grad_threshold
            if prune_min_opacity is not None:
                cfg.train.prune_min_opacity = prune_min_opacity
            if prune_max_scale is not None:
                cfg.train.prune_max_scale = prune_max_scale
            if resize_long_edge is not None:
                cfg.preprocess.resize_long_edge = resize_long_edge
            if colmap_matching is not None:
                if colmap_matching not in ("sequential", "exhaustive"):
                    raise HTTPException(400, f"colmap_matching invalid: {colmap_matching}")
                cfg.preprocess.colmap_matching = colmap_matching
            if init_subsample_mode is not None:
                if init_subsample_mode not in ("random", "confidence"):
                    raise HTTPException(400, f"init_subsample_mode invalid: {init_subsample_mode}")
                cfg.preprocess.init_subsample_mode = init_subsample_mode

            # New SOTA-tier overrides — static mode (added 2026-05-02).
            # Static 3DGS'te lambda_flow / lambda_multiview_consistency etkisiz
            # ama yine de cfg'e yaziyoruz ki user explicit isterse trainer gorsun.
            if lambda_lpips is not None:
                cfg.train.lambda_lpips = float(lambda_lpips)
            if lpips_net is not None:
                if lpips_net not in ("alex", "vgg"):
                    raise HTTPException(400, f"lpips_net invalid: {lpips_net}")
                cfg.train.lpips_net = lpips_net
            if lambda_flow is not None:
                cfg.train.lambda_flow = float(lambda_flow)
            if lambda_multiview_consistency is not None:
                cfg.train.lambda_multiview_consistency = float(lambda_multiview_consistency)
            if lr_cam_K is not None:
                cfg.train.lr_cam_K = float(lr_cam_K)
            if lr_cam_w2c is not None:
                cfg.train.lr_cam_w2c = float(lr_cam_w2c)
            if cam_refine_start_iter is not None:
                cfg.train.cam_refine_start_iter = int(cam_refine_start_iter)
            if opacity_reset_interval is not None:
                cfg.train.opacity_reset_interval = int(opacity_reset_interval)
            if multires_schedule is not None:
                try:
                    schedule = []
                    for pair in multires_schedule.split(","):
                        pair = pair.strip()
                        if not pair:
                            continue
                        k, v = pair.split(":")
                        schedule.append((int(k.strip()), int(v.strip())))
                    cfg.train.multires_schedule = schedule
                except Exception as e:
                    raise HTTPException(400, f"multires_schedule parse failed: {e}")

            return run_pipeline(
                str(video_path),
                safe_scene,
                cfg,
                skip_foundation=effective_skip_foundation,
                progress_callback=on_progress,
                force_preprocess=force_preprocess,
            )

        # ---- Dynamic mode (legacy path) ----
        if smoke_test:
            cfg.train.n_iters = 500
            cfg.train.image_resolution = (480, 270)
            cfg.train.ckpt_interval = 500
            cfg.train.log_interval = 25
            cfg.train.density_start_iter = 100
            cfg.train.density_end_iter = 400
            cfg.train.density_interval = 50
            cfg.train.warmup_iters = 100   # smoke'ta kısa: iter 200'de full
            cfg.export.num_timestamps = 10

        # MICRO TEST — ultra hızlı dev iteration / preflight
        # skip_foundation user'ın seçimine bırakıldı:
        #   - Foundation OFF: ~30-60 sn (cache hit) — pipeline mekanik kontrolü
        #   - Foundation ON:  ~5-10 dk — tam Stage 2 stack preflight (MiDaS + CoTracker + Farneback + tracks/depth loss)
        effective_skip_foundation = skip_foundation
        if micro_test:
            cfg.preprocess.fps = 2                       # 55sn × 2 = ~110 frame → COLMAP hızlı
            cfg.preprocess.resize_long_edge = 480        # yarı çözünürlük
            cfg.train.n_iters = 200
            cfg.train.image_resolution = (320, 180)
            cfg.train.ckpt_interval = 200
            cfg.train.log_interval = 10
            cfg.train.density_start_iter = 50
            cfg.train.density_end_iter = 150
            cfg.train.density_interval = 25
            cfg.train.warmup_iters = 50
            cfg.export.num_timestamps = 5
            # Foundation hafifletme — micro'da küçük modeller / az nokta:
            cfg.foundation.metric3d_model = "MiDaS_small"    # en küçük, en hızlı
            cfg.foundation.cotracker_grid_size = 15          # 30→15 (225 nokta, 4x hızlı)
            cfg.foundation.cotracker_num_points = 900

        # HIGH TEST — 3-4 saat enhanced quality (Full ile Ultra arası)
        # Full preset (30k iter, 640x360) yeterli motion için ama daha derin
        # iterasyon ve modest scale-up ile daha temiz sonuç:
        #   - n_iters 50k (Full 30k → Ultra 80k arası)
        #   - density_end 35k (uzatılmış density window)
        #   - max_gaussians 60k (cap)
        #   - Fourier K=10 (Full 8 → Ultra 12 arası)
        #   - num_timestamps 90 (Full 60 → Ultra 90)
        # Resolution Full ile aynı (640x360) — render hızı korunur.
        # Tahmini süre 3060 Ti: 3-4 saat
        if high_test:
            cfg.preprocess.fps = 10
            cfg.preprocess.resize_long_edge = 960
            cfg.train.n_iters = 50_000
            cfg.train.image_resolution = (640, 360)
            cfg.train.ckpt_interval = 5000
            cfg.train.log_interval = 100
            cfg.train.density_start_iter = 500
            cfg.train.density_end_iter = 35_000
            cfg.train.density_interval = 200
            cfg.train.densify_grad_threshold = 3e-4   # Default 2e-4'ten biraz sıkı
            cfg.train.warmup_iters = 1000
            cfg.train.max_gaussians = 60_000
            # Model — modest scale-up
            cfg.model.hexplane_resolution = 96    # Full default
            cfg.model.hexplane_feat_dim = 48
            cfg.model.mlp_width = 512
            cfg.model.mlp_depth = 4
            cfg.model.fourier_K = 10
            # Export
            cfg.export.num_timestamps = 90
            # Foundation — v3.7.4: 3060 Ti 8GB için CoTracker güvenli ayar
            # Önceden 25 idi ama banana_high'ta OOM oldu. 20 daha güvenli, fallback hazır.
            cfg.foundation.metric3d_model = "metric3d_vit_small"
            cfg.foundation.cotracker_grid_size = 20
            cfg.foundation.cotracker_num_points = 1024

        # ULTRA TEST v2 — 8-12 saat max-quality render (REVISED)
        # ÖNCEKİ ULTRA v1 BAŞARISIZ: banana'da N=164k, 0.08 it/s → 21 gün ETA.
        # Sebepler:
        #   1. densify_grad_threshold=2e-4 default → ultra res'te aşırı split
        #   2. N için hard cap yoktu
        #   3. resolution 960x540 + büyük model + Fourier K=16 = aşırı yük
        # ULTRA v2 fixleri:
        #   - max_gaussians = 80k cap (N hard limit)
        #   - densify_grad_threshold 2e-4 → 5e-4 (1/2.5 split rate)
        #   - resolution 720x405 (3.5× pixel base, manageable)
        #   - n_iters 80k (150k yerine)
        #   - density_end 50k (100k yerine)
        #   - HexPlane 112/56 (128/64 yerine — modest scale)
        #   - MLP 640/4 (768/5 yerine — daha hafif)
        #   - Fourier K=12 (16 yerine)
        # Tahmini süre 3060 Ti: 6-9 saat (cookie-banana scale scenes)
        if ultra_test:
            cfg.preprocess.fps = 10
            cfg.preprocess.resize_long_edge = 960
            cfg.train.n_iters = 80_000
            cfg.train.image_resolution = (720, 405)
            cfg.train.ckpt_interval = 4000
            cfg.train.log_interval = 100
            cfg.train.density_start_iter = 500
            cfg.train.density_end_iter = 50_000
            cfg.train.density_interval = 200
            cfg.train.densify_grad_threshold = 5e-4   # YÜKSELT! 2e-4'ten 2.5×
            cfg.train.warmup_iters = 1500
            # N hard cap — banana 164k olayı tekrarlamasın
            cfg.train.max_gaussians = 80_000
            # Model mimarisi — modest scale-up
            cfg.model.hexplane_resolution = 112
            cfg.model.hexplane_feat_dim = 56
            cfg.model.mlp_width = 640
            cfg.model.mlp_depth = 4
            cfg.model.fourier_K = 12
            # Export
            cfg.export.num_timestamps = 90
            # Foundation
            cfg.foundation.metric3d_model = "metric3d_vit_small"
            cfg.foundation.cotracker_grid_size = 25
            cfg.foundation.cotracker_num_points = 1600

        # ULTRA CLEAN PRESET — v3.8 anti-streak.
        # banana_demo Ultra (PSNR 19.9, görseldeki streak'ler) post-mortem fix'i.
        # Ultra v2 base + 8 değişiklik:
        #   1. lambda_aniso 0.02 (anisotropy regularizer açık — streak fix)
        #   2. aniso_threshold 5.0 (max/min ratio < 5 serbest, üstü ceza)
        #   3. dpos_total_cap_frac 0.05 (Ultra'da 0.2; 4× sıkı, motion daha kısıtlı)
        #   4. lambda_rigidity 1e-3 (Ultra 2e-4'ten 5×; KNN motion uniformity zorla)
        #   5. lambda_fourier_reg 1e-2 (Ultra 1e-3'ten 10×; Fourier coeff overfit bastır)
        #   6. density_end_iter 30000 (Ultra 50k; geri kalan 50k iter sadece refine)
        #   7. prune_max_scale 0.01 (Ultra 0.02; daha küçük max gaussian)
        #   8. sh_degree 2 (Ultra 3; daha smooth color, gaussian-spike incentive azalır)
        # Beklenen: PSNR 22+ (Ultra 19.9'dan +2-3 dB), streak'siz, kontrollü motion.
        # Tahmini süre 3060 Ti: 7-9 saat.
        if ultra_clean:
            cfg.preprocess.fps = 10
            cfg.preprocess.resize_long_edge = 960
            cfg.train.n_iters = 80_000
            cfg.train.image_resolution = (720, 405)
            cfg.train.ckpt_interval = 4000
            cfg.train.log_interval = 100
            cfg.train.density_start_iter = 500
            cfg.train.density_end_iter = 30_000        # Ultra 50k → 30k
            cfg.train.density_interval = 200
            cfg.train.densify_grad_threshold = 5e-4
            cfg.train.warmup_iters = 1500
            cfg.train.max_gaussians = 80_000
            cfg.train.prune_max_scale = 0.01            # Ultra 0.02 → 0.01
            # v3.8 ANTI-STREAK
            cfg.train.lambda_aniso = 0.02               # KAPALI (0) → 0.02 AÇIK
            cfg.train.aniso_threshold = 5.0
            cfg.train.dpos_total_cap_frac = 0.05        # default 0.2 → 0.05 (4× sıkı)
            cfg.train.lambda_rigidity = 1e-3            # default 2e-4 → 1e-3 (5× sıkı)
            cfg.train.lambda_fourier_reg = 1e-2         # default 1e-3 → 1e-2 (10× sıkı)
            # Model — sh_degree azalt
            cfg.model.sh_degree = 2                     # Ultra 3 → 2 (overfit azalt)
            cfg.model.hexplane_resolution = 112
            cfg.model.hexplane_feat_dim = 56
            cfg.model.mlp_width = 640
            cfg.model.mlp_depth = 4
            cfg.model.fourier_K = 12
            # Export
            cfg.export.num_timestamps = 90
            # Foundation
            cfg.foundation.metric3d_model = "metric3d_vit_small"
            cfg.foundation.cotracker_grid_size = 25
            cfg.foundation.cotracker_num_points = 1600

        # STATIC MAX PRESET — v3.9 input pipeline iyilestirmeleri.
        # Hipotez: 5k iter ve 80k iter ayni static quality verdigi icin
        # bottleneck training'de degil INPUT PIPELINE'da. Bu preset tum
        # input iyilestirmelerini AYNI ANDA uygular:
        #   1. fps 10 -> 20 (2x more frames, daha dense view)
        #   2. metric3d_vit_small -> vit_large (daha keskin depth)
        #   3. COLMAP sequential -> exhaustive (loop closure, orbital fix)
        #   4. init subsample random -> confidence (track + error tabanli)
        #   5. sh_degree 2 -> 3 (Ultra Clean'den geri, color expressiveness)
        # + Ultra Clean'in tum anti-streak fix'leri korunur.
        if static_max:
            cfg.preprocess.fps = 20
            cfg.preprocess.resize_long_edge = 1280
            cfg.preprocess.colmap_matching = "exhaustive"
            cfg.preprocess.init_subsample_mode = "confidence"
            cfg.train.n_iters = 80_000
            cfg.train.image_resolution = (720, 405)
            cfg.train.ckpt_interval = 4000
            cfg.train.log_interval = 100
            cfg.train.density_start_iter = 500
            cfg.train.density_end_iter = 30_000
            cfg.train.density_interval = 200
            cfg.train.densify_grad_threshold = 5e-4
            cfg.train.warmup_iters = 1500
            cfg.train.max_gaussians = 80_000
            cfg.train.prune_max_scale = 0.01
            cfg.train.lambda_aniso = 0.02
            cfg.train.aniso_threshold = 5.0
            cfg.train.dpos_total_cap_frac = 0.05
            cfg.train.lambda_rigidity = 1e-3
            cfg.train.lambda_fourier_reg = 1e-2
            cfg.model.sh_degree = 3
            cfg.model.hexplane_resolution = 112
            cfg.model.hexplane_feat_dim = 56
            cfg.model.mlp_width = 640
            cfg.model.mlp_depth = 4
            cfg.model.fourier_K = 12
            cfg.export.num_timestamps = 90
            cfg.foundation.metric3d_model = "metric3d_vit_large"
            cfg.foundation.cotracker_grid_size = 25
            cfg.foundation.cotracker_num_points = 1600

        # --- Override'lar (preset uzerine uygulanir) ---
        # Temel
        if iters is not None:
            cfg.train.n_iters = iters
            # Density control takvimini iter sayisina olcekle (daha spesifik
            # override'lar altta, bu sadece default hesaplama)
            cfg.train.density_start_iter = max(100, int(iters * 0.1))
            cfg.train.density_end_iter = max(
                cfg.train.density_start_iter + 1, int(iters * 0.8)
            )
            cfg.train.density_interval = max(50, int(iters * 0.02))
            cfg.train.ckpt_interval = iters
            cfg.train.log_interval = max(10, iters // 30)
        if resolution:
            try:
                w_str, h_str = resolution.lower().split("x")
                cfg.train.image_resolution = (int(w_str), int(h_str))
            except ValueError as e:
                raise HTTPException(400, f"resolution formati: WxH (orn. '640x360'). Hata: {e}")
        if num_timestamps is not None:
            cfg.export.num_timestamps = num_timestamps
        if fps is not None:
            cfg.preprocess.fps = fps
        # Loss weights
        if lambda_ssim is not None:
            cfg.train.lambda_ssim = lambda_ssim
        if lambda_deform_reg is not None:
            cfg.train.lambda_deform_reg = lambda_deform_reg
        if lambda_smoothness is not None:
            cfg.train.lambda_smoothness = lambda_smoothness
        if lambda_rigidity is not None:
            cfg.train.lambda_rigidity = lambda_rigidity
        if lambda_depth is not None:
            cfg.train.lambda_depth = lambda_depth
        if lambda_mask_motion is not None:
            cfg.train.lambda_mask_motion = lambda_mask_motion
        if lambda_track is not None:
            cfg.train.lambda_track = lambda_track
        if lambda_scale is not None:
            cfg.train.lambda_scale = lambda_scale
        if lambda_aniso is not None:
            cfg.train.lambda_aniso = lambda_aniso
        if aniso_threshold is not None:
            cfg.train.aniso_threshold = aniso_threshold
        if dpos_total_cap_frac is not None:
            cfg.train.dpos_total_cap_frac = dpos_total_cap_frac
        if opacity_reset_interval is not None:
            cfg.train.opacity_reset_interval = opacity_reset_interval
        if track_sample_k is not None:
            cfg.train.track_sample_k = track_sample_k
        if warmup_iters is not None:
            cfg.train.warmup_iters = warmup_iters
        # Learning rates
        if lr_deform is not None:
            cfg.train.lr_deform = lr_deform
        if lr_means is not None:
            cfg.train.lr_means = lr_means
        # Density control — iter override'indan sonra uygulanır, spesifik değerleri yazar
        if density_start_iter is not None:
            cfg.train.density_start_iter = density_start_iter
        if density_end_iter is not None:
            cfg.train.density_end_iter = density_end_iter
        if density_interval is not None:
            cfg.train.density_interval = density_interval
        if densify_grad_threshold is not None:
            cfg.train.densify_grad_threshold = densify_grad_threshold
        if prune_min_opacity is not None:
            cfg.train.prune_min_opacity = prune_min_opacity
        if prune_max_scale is not None:
            cfg.train.prune_max_scale = prune_max_scale
        # Model mimarisi
        if sh_degree is not None:
            cfg.model.sh_degree = sh_degree
        if hexplane_resolution is not None:
            cfg.model.hexplane_resolution = hexplane_resolution
        if hexplane_feat_dim is not None:
            cfg.model.hexplane_feat_dim = hexplane_feat_dim
        if mlp_width is not None:
            cfg.model.mlp_width = mlp_width
        if mlp_depth is not None:
            cfg.model.mlp_depth = mlp_depth
        if num_time_freqs is not None:
            cfg.model.num_time_freqs = num_time_freqs
        # v3.6 / Yol C
        if deform_pos_mode is not None:
            if deform_pos_mode not in ("mlp", "fourier", "hybrid"):
                raise HTTPException(400, f"deform_pos_mode geçersiz: {deform_pos_mode}")
            cfg.model.deform_pos_mode = deform_pos_mode
        if fourier_K is not None:
            cfg.model.fourier_K = max(0, fourier_K)
        if lr_fourier is not None:
            cfg.train.lr_fourier = lr_fourier
        if lambda_fourier_reg is not None:
            cfg.train.lambda_fourier_reg = lambda_fourier_reg
        # Foundation models
        if metric3d_model is not None:
            cfg.foundation.metric3d_model = metric3d_model
        if cotracker_num_points is not None:
            cfg.foundation.cotracker_num_points = cotracker_num_points
        if cotracker_grid_size is not None:
            cfg.foundation.cotracker_grid_size = cotracker_grid_size
        if sam2_threshold is not None:
            cfg.foundation.sam2_threshold = sam2_threshold
        # v3.9 — Preprocessing overrides
        if resize_long_edge is not None:
            cfg.preprocess.resize_long_edge = resize_long_edge
        if colmap_matching is not None:
            if colmap_matching not in ("sequential", "exhaustive"):
                raise HTTPException(400, f"colmap_matching: sequential | exhaustive ({colmap_matching})")
            cfg.preprocess.colmap_matching = colmap_matching
        if init_subsample_mode is not None:
            if init_subsample_mode not in ("random", "confidence"):
                raise HTTPException(400, f"init_subsample_mode: random | confidence ({init_subsample_mode})")
            cfg.preprocess.init_subsample_mode = init_subsample_mode
        if colmap_mv_timestamps is not None:
            if colmap_mv_timestamps < 1 or colmap_mv_timestamps > 50:
                raise HTTPException(400, f"colmap_mv_timestamps: 1-50 ({colmap_mv_timestamps})")
            cfg.preprocess.colmap_mv_timestamps = int(colmap_mv_timestamps)
        if colmap_mv_dense_mvs is not None:
            cfg.preprocess.colmap_mv_dense_mvs = bool(colmap_mv_dense_mvs)

        # New SOTA-tier overrides (added 2026-05-02). opacity_reset_interval
        # zaten yukarida override edilmis durumda; burada eklemiyoruz.
        if lambda_lpips is not None:
            cfg.train.lambda_lpips = float(lambda_lpips)
        if lpips_net is not None:
            if lpips_net not in ("alex", "vgg"):
                raise HTTPException(400, f"lpips_net invalid: {lpips_net}")
            cfg.train.lpips_net = lpips_net
        if lambda_flow is not None:
            cfg.train.lambda_flow = float(lambda_flow)
        if lambda_multiview_consistency is not None:
            cfg.train.lambda_multiview_consistency = float(lambda_multiview_consistency)
        if lr_cam_K is not None:
            cfg.train.lr_cam_K = float(lr_cam_K)
        if lr_cam_w2c is not None:
            cfg.train.lr_cam_w2c = float(lr_cam_w2c)
        if cam_refine_start_iter is not None:
            cfg.train.cam_refine_start_iter = int(cam_refine_start_iter)
        if multires_schedule is not None:
            # Parse "0:480,10000:720,25000:1080" → [(0, 480), (10000, 720), (25000, 1080)]
            try:
                schedule = []
                for pair in multires_schedule.split(","):
                    pair = pair.strip()
                    if not pair:
                        continue
                    k, v = pair.split(":")
                    schedule.append((int(k.strip()), int(v.strip())))
                cfg.train.multires_schedule = schedule
            except Exception as e:
                raise HTTPException(400, f"multires_schedule parse failed: {e}")

        return run_pipeline(
            str(video_path),
            safe_scene,
            cfg,
            skip_foundation=effective_skip_foundation,
            progress_callback=on_progress,
            force_preprocess=force_preprocess,
        )

    manager.submit(job.id, _runner)

    return ProcessResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/status/{job.id}",
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
@app.get("/status/{job_id}", response_model=Job, tags=["jobs"])
def get_status(job_id: str) -> Job:
    """Belirli bir job'un durumunu döndür."""
    manager: JobManager = app.state.manager
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job bulunamadı: {job_id}")
    return job


# ---------------------------------------------------------------------------
# Liste
# ---------------------------------------------------------------------------
@app.get("/jobs", response_model=JobListResponse, tags=["jobs"])
def list_jobs() -> JobListResponse:
    """Tüm job'ları (en yeni önce) listele."""
    manager: JobManager = app.state.manager
    jobs = sorted(manager.list_all(), key=lambda j: j.created_at, reverse=True)
    return JobListResponse(jobs=jobs, total=len(jobs))


# ---------------------------------------------------------------------------
# Download — .ply'ları zip olarak stream et
# ---------------------------------------------------------------------------
@app.get("/download/{job_id}", tags=["jobs"])
def download_result(job_id: str) -> StreamingResponse:
    """Bitmiş bir job'un .ply çıktılarını ZIP olarak indir."""
    manager: JobManager = app.state.manager
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job bulunamadı: {job_id}")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"Job henüz hazır değil (status: {job.status.value})")

    ply_dir = Path(job.ply_dir) if job.ply_dir else scene_paths(job.scene)["output"] / "ply"
    if not ply_dir.exists():
        raise HTTPException(404, f"Çıktı klasörü bulunamadı: {ply_dir}")

    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        raise HTTPException(404, f"{ply_dir} içinde .ply yok")

    # Memory stream'de zip oluştur ve stream'le gönder
    def _zip_iter():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in ply_files:
                zf.write(p, arcname=p.name)
        buf.seek(0)
        while chunk := buf.read(64 * 1024):
            yield chunk

    return StreamingResponse(
        _zip_iter(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job.scene}_{job_id[:8]}.zip"'},
    )


# ---------------------------------------------------------------------------
# Splat - viewer icin bireysel .ply erisimi (Faz 8a)
# Hem in-memory job_id'den hem de diskteki scene name'den ply bulabilir.
# ---------------------------------------------------------------------------
class SplatInfo(BaseModel):
    """Bir job/sahne'nin splat ciktilarinin metadata'si (viewer icin)."""
    job_id: str           # gelen identifier (job_id ya da scene name)
    scene: str            # cozulmus sahne adi
    num_frames: int
    frame_urls: list[str]
    total_size_bytes: int
    status: JobStatus
    source: str           # "registry" ya da "disk"


class SceneListItem(BaseModel):
    """/scenes listesinin bir elemani."""
    name: str
    num_frames: int
    total_size_bytes: int
    modified_ts: float


class SceneListResponse(BaseModel):
    scenes: list[SceneListItem]
    total: int


def _resolve_ply_dir(identifier: str) -> tuple[Path, str, JobStatus, str]:
    """
    Bir identifier'i (job_id veya scene name) ply klasorune cevir.
    Returns (ply_dir, scene_name, status, source).
    source: "registry" | "disk"
    """
    # 1) Once in-memory job registry
    manager: JobManager = app.state.manager
    job = manager.get(identifier)
    if job is not None:
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(
                409, f"Job henuz hazir degil (status: {job.status.value})"
            )
        ply_dir = Path(job.ply_dir) if job.ply_dir else scene_paths(job.scene)["output"] / "ply"
        return ply_dir, job.scene, job.status, "registry"

    # 2) Scene name olarak dene (data/<name>/output/ply)
    safe = _safe_scene_name(identifier)
    candidate = scene_paths(safe)["output"] / "ply"
    if candidate.exists() and any(candidate.glob("*.ply")):
        return candidate, safe, JobStatus.COMPLETED, "disk"

    raise HTTPException(
        404,
        f"Job veya sahne bulunamadi: {identifier} "
        f"(ne registry'de ne data/{safe}/output/ply altinda)",
    )


@app.get("/scenes", response_model=SceneListResponse, tags=["disk"])
def list_disk_scenes() -> SceneListResponse:
    """
    data/ altinda output/ply iceren tum sahneleri listele.
    Server restart olsa bile diskteki ciktilar burada gorunur.
    """
    data_root = Path("data")
    items: list[SceneListItem] = []
    if data_root.exists():
        for scene_dir in sorted(data_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            ply_dir = scene_dir / "output" / "ply"
            if not ply_dir.exists():
                continue
            ply_files = sorted(ply_dir.glob("*.ply"))
            if not ply_files:
                continue
            total = sum(p.stat().st_size for p in ply_files)
            items.append(SceneListItem(
                name=scene_dir.name,
                num_frames=len(ply_files),
                total_size_bytes=total,
                modified_ts=ply_dir.stat().st_mtime,
            ))
    # En yeni once
    items.sort(key=lambda s: s.modified_ts, reverse=True)
    return SceneListResponse(scenes=items, total=len(items))


@app.get("/splat/{job_id}/info", response_model=SplatInfo, tags=["splat"])
def splat_info(job_id: str) -> SplatInfo:
    """
    Viewer metadata: kac frame, hangi URL'lerden cekilecek, toplam boyut.
    `job_id` parametresi hem UUID hem sahne adi olabilir.
    """
    ply_dir, scene, status, source = _resolve_ply_dir(job_id)
    ply_files = sorted(ply_dir.glob("*.ply"))
    if not ply_files:
        raise HTTPException(404, f"{ply_dir} icinde .ply yok")

    total = sum(p.stat().st_size for p in ply_files)
    frame_urls = [f"/splat/{job_id}/frame/{i}" for i in range(len(ply_files))]
    return SplatInfo(
        job_id=job_id,
        scene=scene,
        num_frames=len(ply_files),
        frame_urls=frame_urls,
        total_size_bytes=total,
        status=status,
        source=source,
    )


@app.get("/splat/{job_id}/frame/{idx}", tags=["splat"])
def splat_frame(job_id: str, idx: int) -> FileResponse:
    """
    Belirli bir timestamp icin .ply dosyasini dondur.
    `job_id` hem UUID hem sahne adi olabilir.
    """
    ply_dir, _scene, _status, _source = _resolve_ply_dir(job_id)
    ply_files = sorted(ply_dir.glob("*.ply"))
    if idx < 0 or idx >= len(ply_files):
        raise HTTPException(
            404,
            f"Frame index gecersiz: {idx} (mevcut: 0..{len(ply_files)-1})",
        )

    return FileResponse(
        ply_files[idx],
        media_type="application/octet-stream",
        filename=ply_files[idx].name,
        headers={"Cache-Control": "public, max-age=3600"},
    )


# ---------------------------------------------------------------------------
# Analytics — training logs & summary
# ---------------------------------------------------------------------------
def _resolve_logs_dir(identifier: str) -> Path:
    """identifier (scene adı veya job id) → data/<scene>/output/logs/."""
    manager: JobManager = app.state.manager
    job = manager.get(identifier)
    if job is not None:
        return scene_paths(job.scene)["output"] / "logs"
    # Scene adı olarak dene
    safe = _safe_scene_name(identifier)
    candidate = scene_paths(safe)["output"] / "logs"
    if candidate.exists():
        return candidate
    raise HTTPException(
        404,
        f"Log klasoru bulunamadi: identifier='{identifier}'. "
        f"Train sirasinda yazilan dosyalar data/<scene>/output/logs/ altinda olmali.",
    )


@app.get("/jobs/{identifier}/metrics", tags=["analytics"])
def job_metrics(identifier: str) -> JSONResponse:
    """
    Training sirasinda yazilan metrics.jsonl dosyasini parse edip JSON array dondurur.
    Her entry: {iter, loss, recon, depth, track, ..., psnr, n_points, dpos_mean, dpos_max, warmup, t, it_per_sec}
    """
    logs_dir = _resolve_logs_dir(identifier)
    metrics_file = logs_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return JSONResponse({"metrics": [], "count": 0, "note": "metrics.jsonl yok (henuz training basslamadi olabilir)"})
    records = []
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return JSONResponse({"metrics": records, "count": len(records)})


@app.get("/jobs/{identifier}/summary", tags=["analytics"])
def job_summary(identifier: str) -> JSONResponse:
    """Run summary.json — final metrics + config + phase durations."""
    logs_dir = _resolve_logs_dir(identifier)
    summary_file = logs_dir / "summary.json"
    if not summary_file.exists():
        raise HTTPException(404, "summary.json yok (run bitmedi veya eski run loglamadan once calistirildi)")
    try:
        data = json.loads(summary_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"summary.json parse edilemedi: {e}")
    return JSONResponse(data)


@app.get("/jobs/{identifier}/events", tags=["analytics"])
def job_events(identifier: str, tail: int | None = None) -> JSONResponse:
    """Insan-okunabilir events.log — density ops, opacity resets, phase transitions, warnings."""
    logs_dir = _resolve_logs_dir(identifier)
    events_file = logs_dir / "events.log"
    if not events_file.exists():
        return JSONResponse({"events": [], "count": 0})
    lines = events_file.read_text(encoding="utf-8").splitlines()
    if tail is not None and tail > 0:
        lines = lines[-tail:]
    return JSONResponse({"events": lines, "count": len(lines)})


# ---------------------------------------------------------------------------
# Yardimcilar
# ---------------------------------------------------------------------------
def _safe_scene_name(raw: str) -> str:
    """Path traversal + weird chars sanitize. Sadece alfanumerik, _, -, nokta kabul."""
    safe = "".join(c if (c.isalnum() or c in "_-.") else "_" for c in raw)
    safe = safe.strip("._") or "unnamed_scene"
    return safe[:64]
