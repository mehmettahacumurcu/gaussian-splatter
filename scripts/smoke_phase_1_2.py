"""Phase 1.4-1.9 + Phase 2.1-2.5 smoke test.

Bash mevcut degil + 50+ edit yapildi. Bu script kullanici tarafinda
calistir + her shey importable, syntax saglam, hash determinism dogru
mu dogrular. Pipeline'i tam calistirmaz (GPU/COLMAP gerekmez).

Usage:
    cd 4dgs-studio
    python scripts/smoke_phase_1_2.py

Beklenen output: hepsi 'OK' satirlari + sonunda ozet 'PASS' veya 'FAIL'.
"""
from __future__ import annotations
import ast
import importlib
import sys
import traceback
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

results: list[tuple[str, bool, str]] = []


def step(name: str, fn):
    """Bir step koş, sonucu kaydet."""
    print(f"\n{'=' * 70}\n  {name}\n{'=' * 70}")
    try:
        fn()
        results.append((name, True, ""))
        print(f"  → OK")
    except Exception as e:
        tb = traceback.format_exc()
        results.append((name, False, str(e)))
        print(f"  → FAIL: {type(e).__name__}: {e}")
        print(tb)


# ---------------------------------------------------------------------------
# 1) AST parse — syntax sanity (no imports, just parse)
# ---------------------------------------------------------------------------
def test_ast_parse():
    files = [
        "backend/config.py",
        "backend/pipeline.py",
        "backend/preprocess/cache_utils.py",
        "backend/preprocess/depth_multiview.py",
        "backend/preprocess/dynamic_mask_multiview.py",
        "backend/preprocess/optical_flow.py",
        "backend/preprocess/multiview_pipeline.py",
        "backend/model/gaussian_model.py",
        "backend/model/deformation.py",
        "backend/model/trainer.py",
        "backend/model/losses_perceptual.py",
        "scripts/heavy_preprocess.py",
    ]
    failures = []
    for rel in files:
        p = PROJECT_ROOT / rel
        if not p.exists():
            failures.append(f"  MISSING: {rel}")
            continue
        try:
            ast.parse(p.read_text(encoding="utf-8"))
            print(f"  OK   {rel}")
        except SyntaxError as e:
            failures.append(f"  FAIL {rel}: line {e.lineno}: {e.msg}")
            print(failures[-1])
    if failures:
        raise AssertionError(f"{len(failures)} dosya syntax error:\n" + "\n".join(failures))


# ---------------------------------------------------------------------------
# 2) Import test — heavy modules
# ---------------------------------------------------------------------------
def test_imports():
    mods = [
        ("backend.preprocess.cache_utils", False),  # no torch needed
        ("backend.config", False),
        ("backend.preprocess.depth_multiview", True),  # torch
        ("backend.preprocess.dynamic_mask_multiview", True),  # cv2
        ("backend.preprocess.optical_flow", True),  # torch
        ("backend.model.losses_perceptual", True),  # torch
        ("backend.model.gaussian_model", True),  # torch
        ("backend.model.deformation", True),  # torch
        # Trainer + pipeline en agir, en sonda
        ("backend.model.trainer", True),
        ("backend.preprocess.multiview_pipeline", True),
        ("backend.pipeline", True),
    ]
    failures = []
    for m, needs_torch in mods:
        try:
            importlib.import_module(m)
            print(f"  OK import {m}")
        except Exception as e:
            msg = f"  FAIL import {m}: {type(e).__name__}: {e}"
            print(msg)
            # Sadece torch yoksa ve modul torch gerektiriyorsa toleranslı:
            if needs_torch and "torch" in str(e).lower():
                print(f"    (torch yok, atlanir)")
            else:
                failures.append(msg)
    if failures:
        raise AssertionError("\n".join(failures))


# ---------------------------------------------------------------------------
# 3) Cache utils — settings hash determinism + mutation
# ---------------------------------------------------------------------------
def test_cache_utils():
    from backend.preprocess.cache_utils import (
        compute_settings_hash, get_step_settings,
        _STEP_VALIDATORS, _STEP_SETTINGS,
    )
    from backend.config import default_config

    cfg = default_config()
    print(f"  Validators: {sorted(_STEP_VALIDATORS.keys())}")
    print(f"  Settings keys: {sorted(_STEP_SETTINGS.keys())}")

    # Tum step'ler için hash hesaplanabilmeli
    for step_name in _STEP_VALIDATORS:
        h = compute_settings_hash(cfg, step_name)
        s = get_step_settings(cfg, step_name)
        assert isinstance(h, str) and len(h) == 12, f"hash format hatali: {h}"
        print(f"    {step_name:14s} hash={h} settings={s}")

    # Determinism
    h1 = compute_settings_hash(cfg, "colmap_mv")
    h2 = compute_settings_hash(cfg, "colmap_mv")
    assert h1 == h2, f"deterministik degil: {h1} != {h2}"
    print(f"  ✓ Determinism: {h1} == {h2}")

    # Mutation invalidation
    cfg.preprocess.colmap_mv_timestamps = 999
    h3 = compute_settings_hash(cfg, "colmap_mv")
    assert h1 != h3, f"mutation hash'i degistirmedi: {h1} == {h3}"
    print(f"  ✓ Mutation invalidates: {h1} -> {h3}")

    # Frames hash fps degisikligine duyarli mi
    cfg2 = default_config()
    h_a = compute_settings_hash(cfg2, "frames")
    cfg2.preprocess.fps = cfg2.preprocess.fps + 5
    h_b = compute_settings_hash(cfg2, "frames")
    assert h_a != h_b, "fps mutation frames hash'ini degistirmedi"
    print(f"  ✓ frames hash fps mutation: {h_a} -> {h_b}")


# ---------------------------------------------------------------------------
# 4) Config — yeni Phase 1+2 alanlari erisilebilir mi
# ---------------------------------------------------------------------------
def test_config_fields():
    from backend.config import default_config
    cfg = default_config()
    # Phase 1.6/1.7/1.8/1.9
    new_train_fields = [
        "lambda_lpips", "lpips_net", "lpips_warmup_iters",
        "lambda_multiview_consistency",
        "lambda_flow", "flow_warmup_iters",
        "densify_mv_threshold_scale",
        "multires_schedule",
        "lr_cam_K", "lr_cam_w2c", "cam_refine_start_iter",
    ]
    for f in new_train_fields:
        v = getattr(cfg.train, f)
        print(f"  cfg.train.{f} = {v!r}")
    # Phase 2.5
    for f in ["multires_resolutions", "multires_feat_dim"]:
        v = getattr(cfg.model, f)
        print(f"  cfg.model.{f} = {v!r}")
    # Scene paths flow
    from backend.config import scene_paths
    sp = scene_paths("dummy_scene_test")
    for k in ["flow", "flow_mv"]:
        assert k in sp, f"scene_paths icinde '{k}' yok"
        print(f"  scene_paths['{k}'] = {sp[k]}")


# ---------------------------------------------------------------------------
# 5) GaussianModel — is_static / is_background buffers
# ---------------------------------------------------------------------------
def test_gs_buffers():
    import torch
    from backend.model.gaussian_model import GaussianModel
    pts = torch.randn(100, 3) * 2.0
    rgb = torch.rand(100, 3)
    gs = GaussianModel(init_points=pts, init_colors=rgb, sh_degree=1, fourier_K=4)
    assert hasattr(gs, "is_static"), "is_static buffer yok"
    assert hasattr(gs, "is_background"), "is_background buffer yok"
    assert gs.is_static.shape == (100,), f"shape: {gs.is_static.shape}"
    assert gs.is_static.all(), "is_static default True olmali"
    assert not gs.is_background.any(), "is_background default False olmali"
    print(f"  ✓ buffers: is_static={gs.is_static.shape}/{gs.num_static} static, "
          f"is_background={gs.is_background.shape}/{gs.num_background} bg")

    # Promote dynamic
    dyn = torch.zeros(100, dtype=torch.bool)
    dyn[:30] = True
    promoted = gs.promote_dynamic(dyn)
    assert promoted == 30, f"promoted: {promoted}"
    assert gs.num_dynamic == 30
    assert gs.num_static == 70
    print(f"  ✓ promote_dynamic(30 mask) → {gs.num_dynamic} dyn / {gs.num_static} static")

    # Background flag
    n_bg = gs.flag_background_by_distance(
        scene_center=torch.zeros(3), scene_extent=1.0, ratio=2.0,
    )
    print(f"  ✓ flag_background_by_distance: {n_bg} bg")

    # Prune senkronu
    keep = torch.ones(100, dtype=torch.bool); keep[:10] = False
    gs._apply_mask(keep)
    assert gs.num_points == 90
    assert gs.is_static.shape == (90,)
    assert gs.is_background.shape == (90,)
    print(f"  ✓ prune senkronu: {gs.num_points} points, buffers shape match")


# ---------------------------------------------------------------------------
# 6) DeformationField — multi-res HexPlane opt-in
# ---------------------------------------------------------------------------
def test_deformation_multires():
    import torch
    from backend.model.deformation import DeformationField, MultiResHexPlane

    # Single-res default
    df1 = DeformationField(resolution=32, feat_dim=12, mlp_width=64, mlp_depth=2)
    print(f"  ✓ single-res DeformationField: hexplane={type(df1.hexplane).__name__}")
    # Multi-res opt-in
    df2 = DeformationField(
        resolution=32, feat_dim=12, mlp_width=64, mlp_depth=2,
        multires_resolutions=[12, 24, 48], multires_feat_dim=8,
    )
    print(f"  ✓ multi-res: {type(df2.hexplane).__name__} "
          f"resolutions={df2.hexplane.resolutions} "
          f"total_feat={df2.hexplane.total_feat_dim}")
    # Forward pass
    means = torch.randn(50, 3) * 0.5
    out = df2.forward(means, t=0.5, scene_extent=1.0)
    # Out: (delta_pos, delta_quat, delta_scale) tuple veya tensor
    print(f"  ✓ forward output type: {type(out)}")


# ---------------------------------------------------------------------------
# 7) LPIPS module — graceful no-deps fallback
# ---------------------------------------------------------------------------
def test_lpips_fallback():
    import torch
    from backend.model.losses_perceptual import LPIPSLoss
    fn = LPIPSLoss(net="alex")
    # AlexNet 5 max-pool: minimum input boyutu ~64x64.
    # Training resolution genelde (640, 360) — 64x64 bol miktarda yeterli.
    # Production'da pred_chw GPU'da olur, LPIPS otomatik CUDA'ya yuklenir.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = torch.rand(3, 64, 64, device=device)
    target = torch.rand(3, 64, 64, device=device)
    out = fn(pred, target)
    print(f"  ✓ LPIPSLoss output: {out.item():.4f} on {device} "
          f"(lpips paketi yoksa 0.0 dondurur)")
    # Gradient akiyor mu (training-time dogrulamasi)
    pred2 = torch.rand(3, 64, 64, device=device, requires_grad=True)
    target2 = torch.rand(3, 64, 64, device=device)
    out2 = fn(pred2, target2)
    out2.backward()
    has_grad = pred2.grad is not None and pred2.grad.abs().sum() > 0
    print(f"  ✓ Gradient flow: {has_grad} "
          f"(pred.grad mean abs={pred2.grad.abs().mean().item():.6f})")


# ---------------------------------------------------------------------------
# 8) heavy_preprocess --list-profiles (subprocess)
# ---------------------------------------------------------------------------
def test_heavy_preprocess_help():
    import subprocess
    p = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "heavy_preprocess.py"),
         "--list-profiles"],
        capture_output=True, text=True, timeout=20,
    )
    print(p.stdout[-1000:] if p.stdout else "(no stdout)")
    if p.returncode != 0:
        print(f"  STDERR: {p.stderr[-500:]}")
        raise RuntimeError(f"return code {p.returncode}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    step("1) AST parse all 12 files", test_ast_parse)
    step("2) Imports", test_imports)
    step("3) Cache hash determinism + mutation", test_cache_utils)
    step("4) Config new fields exist", test_config_fields)
    step("5) GaussianModel is_static/is_background buffers", test_gs_buffers)
    step("6) DeformationField multi-res HexPlane", test_deformation_multires)
    step("7) LPIPSLoss graceful fallback", test_lpips_fallback)
    step("8) heavy_preprocess --list-profiles", test_heavy_preprocess_help)

    print("\n" + "=" * 70)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print(f"  SUMMARY: {n_pass}/{len(results)} pass, {n_fail} fail")
    print("=" * 70)
    for name, ok, err in results:
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}" + (f"  — {err[:100]}" if err else ""))

    sys.exit(0 if n_fail == 0 else 1)
