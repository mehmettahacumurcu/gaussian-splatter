"""Minimal GLB (binary glTF 2.0) helpers: validate, and wrap a scene in a transform node."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942


class GlbFormatError(ValueError):
    """The file is not a usable binary glTF 2.0 file."""


def _read_chunks(data: bytes) -> list[tuple[int, bytes]]:
    if len(data) < 20:
        raise GlbFormatError("File too small to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GlbFormatError("Not a GLB file (bad magic)")
    if version != 2:
        raise GlbFormatError(f"Unsupported glTF version {version}")
    if length > len(data):
        raise GlbFormatError("GLB file is truncated")
    chunks: list[tuple[int, bytes]] = []
    off = 12
    while off + 8 <= length:
        clen, ctype = struct.unpack_from("<II", data, off)
        body = data[off + 8: off + 8 + clen]
        if len(body) != clen:
            raise GlbFormatError("GLB chunk is truncated")
        chunks.append((ctype, body))
        off += 8 + clen
    if not chunks or chunks[0][0] != CHUNK_JSON:
        raise GlbFormatError("GLB must start with a JSON chunk")
    return chunks


def _parse_json(chunk: bytes) -> dict:
    try:
        return json.loads(chunk.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlbFormatError(f"Invalid GLB JSON chunk: {exc}") from exc


def validate_glb(path: str | Path) -> dict:
    return _parse_json(_read_chunks(Path(path).read_bytes())[0][1])


def placement_matrix(position, quaternion_xyzw, scale: float) -> np.ndarray:
    from .bake import quat_to_matrix

    m = np.eye(4)
    m[:3, :3] = float(scale) * quat_to_matrix(tuple(quaternion_xyzw))
    m[:3, 3] = position
    return m


def wrap_with_transform(src: str | Path, dst: str | Path, matrix: np.ndarray) -> Path:
    """Copy ``src`` to ``dst`` with every scene root re-parented under one node carrying ``matrix``."""
    chunks = _read_chunks(Path(src).read_bytes())
    gltf = _parse_json(chunks[0][1])
    nodes = gltf.setdefault("nodes", [])
    scenes = gltf.get("scenes")
    scene_idx = gltf.get("scene", 0)
    if scenes and 0 <= scene_idx < len(scenes):
        roots = list(scenes[scene_idx].get("nodes", []))
    else:
        children = {c for node in nodes for c in node.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
        scenes = gltf["scenes"] = [{"nodes": []}]
        scene_idx = 0
    wrapper: dict = {
        "name": "compose_root",
        "matrix": [float(v) for v in np.asarray(matrix, dtype=np.float64).T.reshape(-1)],  # column-major
    }
    if roots:
        wrapper["children"] = roots
    nodes.append(wrapper)
    scenes[scene_idx]["nodes"] = [len(nodes) - 1]
    gltf["scene"] = scene_idx

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    body = [(CHUNK_JSON, json_bytes)] + chunks[1:]
    total = 12 + sum(8 + len(b) for _, b in body)
    out = bytearray(struct.pack("<III", GLB_MAGIC, 2, total))
    for ctype, b in body:
        out += struct.pack("<II", len(b), ctype) + b
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(out))
    return dst
