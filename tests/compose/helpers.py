"""Shared builders for scene-composer tests."""
from __future__ import annotations

import json
import struct

import numpy as np

from backend.compose.plyio import GaussianCloud


def random_cloud(n: int, degree: int = 3, seed: int = 0) -> GaussianCloud:
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2 - 1
    quats = rng.normal(size=(n, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return GaussianCloud(
        means=rng.normal(size=(n, 3)).astype(np.float32),
        log_scales=rng.normal(-3.0, 0.5, size=(n, 3)).astype(np.float32),
        quats=quats.astype(np.float32),
        opacities=rng.normal(size=n).astype(np.float32),
        sh_dc=rng.normal(0.0, 0.5, size=(n, 3)).astype(np.float32),
        sh_rest=rng.normal(0.0, 0.2, size=(n, k, 3)).astype(np.float32),
    )


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    from backend.compose.bake import quat_to_matrix

    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return quat_to_matrix(tuple(q))


def minimal_glb(gltf: dict | None = None, bin_payload: bytes = b"\x01\x02\x03\x04") -> bytes:
    gltf = gltf or {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "n0", "children": [1]}, {"name": "n1"}],
    }
    js = json.dumps(gltf).encode("utf-8")
    js += b" " * (-len(js) % 4)
    bin_payload += b"\x00" * (-len(bin_payload) % 4)
    total = 12 + 8 + len(js) + 8 + len(bin_payload)
    return (
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(js), 0x4E4F534A) + js
        + struct.pack("<II", len(bin_payload), 0x004E4942) + bin_payload
    )
