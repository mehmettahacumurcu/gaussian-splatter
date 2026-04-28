"""LLFF → OpenCV convention detection.

Tum permute + flip kombinasyonlarini dener, hangi en yuksek mean
dot(forward, to_centroid) veriyor onu raporlar. Hangi standart kombinasyonun
N3V flame_steak icin dogru oldugunu otomatik bulur.
"""
from __future__ import annotations
import itertools
from pathlib import Path
import numpy as np


def load_raw_poses(poses_bounds_path: Path):
    """Raw LLFF poses_bounds.npy → list of (R_3x3, t_3,) raw matrices."""
    pb = np.load(str(poses_bounds_path))  # (N, 17)
    raw = []
    for i in range(pb.shape[0]):
        mat3x5 = pb[i, :15].reshape(3, 5)
        R = mat3x5[:, :3].astype(np.float64)
        t = mat3x5[:, 3].astype(np.float64)
        raw.append((R, t))
    return raw


def apply_convention(R_raw, t_raw, col_perm, col_signs, row_perm=None, row_signs=None):
    """Apply a candidate convention transform.

    col_perm: tuple (a, b, c) where new col i = R_raw[:, perm[i]]
    col_signs: tuple (s0, s1, s2) where new col i *= s
    row_perm: optional same for rows (world frame relabel)
    row_signs: optional same
    """
    R = R_raw.copy()
    t = t_raw.copy()
    if row_perm is not None:
        R = R[list(row_perm), :]
        t = t[list(row_perm)]
        if row_signs is not None:
            R = R * np.array(row_signs)[:, None]
            t = t * np.array(row_signs)
    R = R[:, list(col_perm)] * np.array(col_signs)[None, :]
    return R, t


def evaluate(raw_poses, R_to_c2w, t_to_pos):
    """Compute mean dot(forward, to_centroid) for a candidate convention."""
    c2w_list = []
    for R_raw, t_raw in raw_poses:
        R, t = R_to_c2w(R_raw, t_raw)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = t_to_pos(R_raw, t_raw)
        c2w_list.append(c2w)

    positions = np.array([c2w[:3, 3] for c2w in c2w_list])
    centroid = positions.mean(0)

    dots = []
    for c2w in c2w_list:
        pos = c2w[:3, 3]
        fwd = c2w[:3, 2]   # OpenCV forward = col 2 of c2w R
        to_c = centroid - pos
        n = np.linalg.norm(to_c)
        if n < 1e-9:
            continue
        to_c /= n
        dots.append(float(np.dot(fwd, to_c)))
    return float(np.mean(dots)), float(np.min(dots)), centroid


def main():
    poses_path = Path("data/flame_steak/poses_bounds.npy")
    raw_poses = load_raw_poses(poses_path)
    print(f"Loaded {len(raw_poses)} cameras from {poses_path}")
    print()

    # Print cam00 raw matrix for debugging
    R0, t0 = raw_poses[0]
    print("cam00 raw R matrix:")
    print(R0)
    print(f"cam00 raw t: {t0}")
    print()

    # Strategy: try 6 column permutations × 8 sign combinations × 2 row variants
    col_perms = list(itertools.permutations([0, 1, 2]))
    sign_combos = list(itertools.product([1, -1], repeat=3))

    results = []

    # Variant A: column permutation only (no row permutation)
    for cp in col_perms:
        for cs in sign_combos:
            def make_fn(cp=cp, cs=cs):
                def R_to_c2w(R_raw, t_raw):
                    R = R_raw[:, list(cp)] * np.array(cs)[None, :]
                    return R, t_raw
                def t_to_pos(R_raw, t_raw):
                    return t_raw
                return R_to_c2w, t_to_pos
            R_fn, t_fn = make_fn()
            try:
                mean_d, min_d, _ = evaluate(raw_poses, R_fn, t_fn)
            except Exception:
                continue
            results.append({
                "variant": "A: col-only",
                "col_perm": cp,
                "col_signs": cs,
                "row_perm": None,
                "row_signs": None,
                "mean_dot": mean_d,
                "min_dot": min_d,
            })

    # Variant B: standard LLFF row permutation [r1, -r0, r2] then column flip
    # (4DGaussians-style)
    for cp in col_perms:
        for cs in sign_combos:
            row_perm = (1, 0, 2)
            row_signs = (1, -1, 1)
            def make_fn(cp=cp, cs=cs, row_perm=row_perm, row_signs=row_signs):
                def R_to_c2w(R_raw, t_raw):
                    R = R_raw[list(row_perm), :] * np.array(row_signs)[:, None]
                    R = R[:, list(cp)] * np.array(cs)[None, :]
                    return R, t_raw
                def t_to_pos(R_raw, t_raw):
                    t = t_raw[list(row_perm)] * np.array(row_signs)
                    return t
                return R_to_c2w, t_to_pos
            R_fn, t_fn = make_fn()
            try:
                mean_d, min_d, _ = evaluate(raw_poses, R_fn, t_fn)
            except Exception:
                continue
            results.append({
                "variant": "B: LLFF row-perm + col",
                "col_perm": cp,
                "col_signs": cs,
                "row_perm": row_perm,
                "row_signs": row_signs,
                "mean_dot": mean_d,
                "min_dot": min_d,
            })

    # Sort by mean_dot descending
    results.sort(key=lambda r: -r["mean_dot"])

    print(f"Top 15 conventions by mean dot:")
    print("-" * 100)
    print(f"{'variant':<28} {'col_perm':<10} {'col_signs':<14} {'row_perm':<10} {'row_signs':<14} {'mean':>7} {'min':>7}")
    print("-" * 100)
    for r in results[:15]:
        print(f"{r['variant']:<28} {str(r['col_perm']):<10} {str(r['col_signs']):<14} "
              f"{str(r['row_perm']):<10} {str(r['row_signs']):<14} "
              f"{r['mean_dot']:+7.3f} {r['min_dot']:+7.3f}")

    print()
    best = results[0]
    print("=" * 70)
    print(f"BEST: mean dot = {best['mean_dot']:+.3f}, min dot = {best['min_dot']:+.3f}")
    print(f"  variant: {best['variant']}")
    print(f"  col_perm: {best['col_perm']}, col_signs: {best['col_signs']}")
    print(f"  row_perm: {best['row_perm']}, row_signs: {best['row_signs']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
