from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest

from experiments.learned_quality.model_adapters import SeaRaftTorchAdapter


_BATCH_COUNTERS = (
    "cnet.layer2.0.downsample.1.num_batches_tracked",
    "cnet.layer3.0.downsample.1.num_batches_tracked",
    "fnet.layer2.0.downsample.1.num_batches_tracked",
    "fnet.layer3.0.downsample.1.num_batches_tracked",
)
_SHARED_BATCH_STATE = tuple(
    f"{prefix}.{name}"
    for prefix in (
        "cnet.layer2.0.downsample.1",
        "cnet.layer3.0.downsample.1",
        "fnet.layer2.0.downsample.1",
        "fnet.layer3.0.downsample.1",
    )
    for name in ("weight", "bias", "running_mean", "running_var")
)


def _sea_raft_layout(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "sea-raft"
    (source / "core").mkdir(parents=True)
    config = source / "config" / "eval"
    config.mkdir(parents=True)
    (config / "spring-M.json").write_text(
        json.dumps({"scale": -1, "iters": 4, "var_min": 0, "var_max": 10}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"verified checkpoint")
    return source, checkpoint


def _install_fake_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    missing_keys: tuple[str, ...],
    load_model_error: BaseException,
) -> type[object]:
    class FakeRaft:
        constructed = 0
        loaded_state = None
        strict = None
        load_model_call = None

        def __init__(self, args: object) -> None:
            self.args = args
            self.device = "cpu"
            self.training = True
            type(self).constructed += 1

        @classmethod
        def from_pretrained(cls, *_args: object, **_kwargs: object) -> object:
            raise AssertionError(f"{set(_BATCH_COUNTERS)} != set()")

        def load_state_dict(self, state: object, *, strict: bool) -> object:
            type(self).loaded_state = state
            type(self).strict = strict
            return SimpleNamespace(
                missing_keys=list(missing_keys),
                unexpected_keys=[],
            )

        def to(self, device: str) -> "FakeRaft":
            self.device = device
            return self

        def eval(self) -> "FakeRaft":
            self.training = False
            return self

    raft_module = ModuleType("raft")
    raft_module.RAFT = FakeRaft
    safetensors_package = ModuleType("safetensors")
    safetensors_package.__path__ = []
    safetensors_torch = ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda path, device: {
        "checkpoint": (Path(path), device)
    }

    def load_model(model, path, *, strict, device):
        FakeRaft.load_model_call = (model, Path(path), strict, device)
        model.loaded_state = "alias-aware"
        raise load_model_error

    safetensors_torch.load_model = load_model
    monkeypatch.setitem(sys.modules, "raft", raft_module)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors_package)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    return FakeRaft


def test_sea_raft_loader_accepts_only_batchnorm_tracking_buffer_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, checkpoint = _sea_raft_layout(tmp_path)
    fake_raft = _install_fake_dependencies(
        monkeypatch,
        missing_keys=_SHARED_BATCH_STATE,
        load_model_error=AssertionError(f"{set(_BATCH_COUNTERS)} != set()"),
    )

    adapter = SeaRaftTorchAdapter(source, checkpoint, device="cuda")

    assert fake_raft.constructed == 1
    assert fake_raft.load_model_call == (
        adapter.model,
        checkpoint / "model.safetensors",
        True,
        "cpu",
    )
    assert adapter.model.loaded_state == "alias-aware"
    assert adapter.model.device == "cuda"
    assert adapter.model.training is False


def test_sea_raft_loader_rejects_a_missing_learned_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, checkpoint = _sea_raft_layout(tmp_path)
    _install_fake_dependencies(
        monkeypatch,
        missing_keys=("fnet.layer2.0.conv1.weight",),
        load_model_error=RuntimeError("SEA-RAFT checkpoint is missing learned weight"),
    )

    with pytest.raises(RuntimeError, match="learned weight"):
        SeaRaftTorchAdapter(source, checkpoint, device="cuda")


def test_sea_raft_loader_rejects_other_alias_consistency_assertions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, checkpoint = _sea_raft_layout(tmp_path)
    _install_fake_dependencies(
        monkeypatch,
        missing_keys=(),
        load_model_error=AssertionError(
            "{'fnet.layer2.0.downsample.1.weight'} != set()"
        ),
    )

    with pytest.raises(RuntimeError, match="alias consistency"):
        SeaRaftTorchAdapter(source, checkpoint, device="cuda")
