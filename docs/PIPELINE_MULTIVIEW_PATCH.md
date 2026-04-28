# Pipeline.py Multi-view Patch (Manuel Apply)

Linux mount truncate problemi yüzünden bash patch güvensiz. Sen Windows
tarafında editör (Notepad++/VSCode) ile aşağıdaki ufak değişiklikleri yap.

## Patch 1 — import ekle (en üst, mevcut import'ların sonuna)

**Bul:**
```python
from .run_logger                import RunLogger
```

**Sonrasına ekle:**
```python
from .config import is_multiview_scene  # v5.0
```

## Patch 2 — Faz 2b'den önce multi-view branching

**Bul** (paths["base"].mkdir'den önce kontrol et, ama daha güvenli yer Faz 2a'dan önce):

```python
    # -------- Faz 2a: Frame çıkarma --------
    print("\n[Faz 2a] Frame çıkarma")
```

**Üstüne ekle:**
```python
    # -------- v5.0: Multi-view scene detection --------
    is_mv = is_multiview_scene(scene_name)
    mv_ctx = None
    if is_mv:
        from .preprocess.multiview_pipeline import prepare_multiview_scene
        print(f"\n[v5.0] Multi-view scene detected: {scene_name}")
        mv_ctx = prepare_multiview_scene(paths, cfg, on_progress=cb)
        # Multi-view: Faz 2a (single video frame extract) + Faz 2b (COLMAP) atlanır
        # Bunun yerine multi-cam frame extraction + N3V calibration kullanılır
        status["frames"] = f"multi-view ok ({mv_ctx['n_cameras']} cams)"
        status["colmap"] = f"N3V calibration ({mv_ctx['n_cameras']} cams)"
        # Single-view-equivalent for trainer:
        cams = {f"{mv_ctx['primary_cam']}/frame_{i:04d}.png": {
            "K": np.array(mv_ctx['calibration'][mv_ctx['primary_cam']]['K']),
            "R": np.array(mv_ctx['calibration'][mv_ctx['primary_cam']]['w2c'])[:3, :3],
            "t": np.array(mv_ctx['calibration'][mv_ctx['primary_cam']]['w2c'])[:3, 3],
            "w2c": np.array(mv_ctx['calibration'][mv_ctx['primary_cam']]['w2c']),
            "width": mv_ctx['calibration'][mv_ctx['primary_cam']]['width'],
            "height": mv_ctx['calibration'][mv_ctx['primary_cam']]['height'],
        } for i in range(len(mv_ctx['primary_frame_paths']))}
        xyz = mv_ctx['init_xyz']
        rgb = mv_ctx['init_rgb']
        # Tüm Faz 2a-2b ve Faz 3 (foundation) atlanmış sayılır.
        # Faz 4'e direkt geçilir.
```

## Patch 3 — Single-view path'leri "if not is_mv:" altına alma

Faz 2a, 2b, 3 (foundation) blokları artık conditional olmalı:

**Bul:**
```python
    # -------- Faz 2a: Frame çıkarma --------
    print("\n[Faz 2a] Frame çıkarma")
    cb("frames", 0.0, "Video karelere ayrılıyor", {})
    extract_frames(...)
    status["frames"] = "ok"
    cb("frames", 1.0, "Kareler hazır", {})
```

**Şuna değiştir:**
```python
    if not is_mv:
        # -------- Faz 2a: Frame çıkarma --------
        print("\n[Faz 2a] Frame çıkarma")
        cb("frames", 0.0, "Video karelere ayrılıyor", {})
        extract_frames(...)
        status["frames"] = "ok"
        cb("frames", 1.0, "Kareler hazır", {})
```

Aynı şeyi `Faz 2b COLMAP` ve `Faz 3 Foundation` blokları için de yap (her birinin başına `if not is_mv:` ekle, içeriği bir seviye gir).

## Patch 4 — Trainer call multi-view aware

**Bul** (frame_paths param'ı geçen kısmı):

```python
    print("\n[Faz 5] Training loop")
    ...
    history = trainer.train(
        frame_paths=...,
        cam_K=cam_K,
        cam_w2c_per_frame=cam_w2c_list,
        ...
    )
```

**Multi-view ise frame_paths'i mv_ctx'ten al:**
```python
    if is_mv:
        train_frame_paths = mv_ctx['primary_frame_paths']
        train_cam_K = mv_ctx['primary_K']
        train_cam_w2c = mv_ctx['primary_w2c_per_frame']
    else:
        train_frame_paths = ...  # eski path
        train_cam_K = cam_K
        train_cam_w2c = cam_w2c_list
```

## Test

Patch sonrası:
```cmd
python -c "import ast; ast.parse(open('backend/pipeline.py', encoding='utf-8').read()); print('OK')"
```

Sonra flame_steak ile test:
```cmd
python scripts\load_n3v.py --src C:\Users\TAHA\Downloads\flame_steak --dst data\flame_steak
curl -X POST http://127.0.0.1:8000/process -F "video=@data\flame_steak\videos\cam05.mp4" -F "scene=flame_steak" -F "iters=15000"
```

Pipeline auto-detect ile multi-view path'i alacak, primary cam single-view
supervision ile train edecek.

## Karmaşıksa Söyle

Yukarıdaki manuel patches karmaşık ise, alternatif: trainer'a multi-view
hiç koymadan, sadece **scripts/run_multiview.py --select-cam** ile single-view
proxy approach kullan (önceki MVP yöntemi). Pipeline değişmez, manuel cam
seçimi ile çalışır.
