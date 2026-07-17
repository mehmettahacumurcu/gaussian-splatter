from pathlib import Path

from experiments.learned_quality.contracts import FrameArtifact
from experiments.learned_quality.runtime import _static_tracks


def test_static_tracks_drop_colmap_point_observed_twice_in_one_frame(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 frame_000000.png\n"
        "10 20 7 11 21 7\n"
        "2 1 0 0 0 0 0 0 1 frame_000001.png\n"
        "30 40 7\n",
        encoding="utf-8",
    )
    (model / "points3D.txt").write_text(
        "7 0 0 1 255 255 255 0.1 1 0 1 1 2 0\n",
        encoding="utf-8",
    )
    frames = (
        FrameArtifact("frame_000000.png", "frame-a", tmp_path / "a.png", "a" * 64),
        FrameArtifact("frame_000001.png", "frame-b", tmp_path / "b.png", "b" * 64),
    )

    assert _static_tracks(model, frames) == ()
