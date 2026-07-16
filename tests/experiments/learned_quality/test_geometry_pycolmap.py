from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from experiments.learned_quality.da3 import CameraRecord, PinholeCamera
from experiments.learned_quality.geometry import _write_known_pose_model


def _require_real_pycolmap_gpu():
    if os.environ.get("LEARNED_PYCOLMAP_GPU_TESTS") != "1":
        pytest.skip("set LEARNED_PYCOLMAP_GPU_TESTS=1 for the real fixture")
    pycolmap = pytest.importorskip("pycolmap")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if getattr(pycolmap, "__version__", "") != "3.12.6":
        pytest.skip("the learned experiment pins PyCOLMAP 3.12.6")
    return pycolmap


@pytest.mark.gpu
@pytest.mark.integration
def test_known_pose_pycolmap_uses_exact_database_ids_and_triangulates_shared_track(
    tmp_path: Path,
) -> None:
    pycolmap = _require_real_pycolmap_gpu()
    width, height = 128, 96
    camera = PinholeCamera("PINHOLE", width, height, 80.0, 80.0, 64.0, 48.0)
    image_root = tmp_path / "images"
    mask_root = tmp_path / "masks"
    image_root.mkdir()
    mask_root.mkdir()
    rng = np.random.default_rng(20260716)
    texture = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    names = tuple(f"frame_{index:06d}.png" for index in range(3))
    for index, name in enumerate(names):
        Image.fromarray(np.roll(texture, -3 * index, axis=1), mode="RGB").save(
            image_root / name,
            format="PNG",
        )
        Image.fromarray(np.full((height, width), 255, dtype=np.uint8), mode="L").save(
            mask_root / f"{name}.png",
            format="PNG",
        )

    database_path = tmp_path / "colmap.db"
    reader = pycolmap.ImageReaderOptions(
        mask_path=str(mask_root),
        camera_model="PINHOLE",
        camera_params="80,80,64,48",
    )
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_root),
        image_names=list(names),
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model="PINHOLE",
        reader_options=reader,
        device=pycolmap.Device.cuda,
    )
    pycolmap.match_exhaustive(
        database_path=str(database_path),
        device=pycolmap.Device.cuda,
    )
    database = pycolmap.Database(str(database_path))
    try:
        database_images = tuple(database.read_all_images())
        image_ids = {image.name: int(image.image_id) for image in database_images}
        camera_ids = {int(image.camera_id) for image in database_images}
    finally:
        database.close()
    assert set(image_ids) == set(names)
    assert len(camera_ids) == 1

    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    cameras = []
    for index, name in enumerate(names):
        matrix = [list(row) for row in identity]
        matrix[0][3] = -0.1875 * index
        cameras.append(
            CameraRecord(
                image_name=name,
                frame_id=f"frame-{index:06d}",
                w2c=tuple(tuple(value for value in row) for row in matrix),
            )
        )
    known_root = tmp_path / "known"
    _write_known_pose_model(
        known_root,
        camera_id=next(iter(camera_ids)),
        shared_camera=camera,
        image_ids=image_ids,
        anchor_cameras=tuple(cameras),
    )
    known = pycolmap.Reconstruction(str(known_root))
    assert {int(image_id) for image_id in known.images} == set(image_ids.values())
    for image_id, image in known.images.items():
        assert image.name in image_ids
        assert image_ids[image.name] == int(image_id)

    triangulated = pycolmap.triangulate_points(
        known,
        str(database_path),
        str(image_root),
        str(tmp_path / "triangulated"),
        clear_points=True,
        refine_intrinsics=False,
    )
    assert len(triangulated.points3D) > 0
    assert any(
        len(point.track.elements) >= 3 for point in triangulated.points3D.values()
    )
    final_camera = next(iter(triangulated.cameras.values()))
    np.testing.assert_allclose(final_camera.params, (80.0, 80.0, 64.0, 48.0))
