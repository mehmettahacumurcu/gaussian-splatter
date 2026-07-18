from __future__ import annotations

import ast
import json
import sys
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .contracts import FrameArtifact
from .dependencies import CHECKPOINT_MODEL_REFS
from .flow import FlowPairRequest, FlowProgressCallback, SeaRaftPairPrediction
from .segmentation import DetectionBox, SamBoxPrompt


def _checkpoint_ref(repo_id: str):
    matches = tuple(ref for ref in CHECKPOINT_MODEL_REFS if ref.repo_id == repo_id)
    if len(matches) != 1:
        raise RuntimeError(f"missing exact pinned model ref for {repo_id}")
    return matches[0]


class TransformersGroundingDinoAdapter:
    model_ref = _checkpoint_ref("IDEA-Research/grounding-dino-tiny")

    def __init__(self, checkpoint: Path, *, device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(
            str(checkpoint), local_files_only=True
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(checkpoint), local_files_only=True
        ).to(self.device)
        self.model.eval()

    def to(self, device: str):
        self.model.to(device)
        self.device = self.model.device
        return self

    def detect_batch(
        self,
        frames: tuple[FrameArtifact, ...],
        *,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> Mapping[str, tuple[DetectionBox, ...]]:
        import torch
        from PIL import Image

        images = []
        for frame in frames:
            with Image.open(frame.path) as opened:
                images.append(opened.convert("RGB").copy())
        inputs = self.processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = [(image.height, image.width) for image in images]
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
        result: dict[str, tuple[DetectionBox, ...]] = {}
        for frame, prediction, image in zip(frames, processed, images, strict=True):
            boxes = prediction["boxes"].detach().cpu().numpy()
            scores = prediction["scores"].detach().cpu().numpy()
            labels = prediction.get("text_labels", prediction.get("labels"))
            rows = []
            for index, (box, score, label) in enumerate(
                zip(boxes, scores, labels, strict=True)
            ):
                x1, y1, x2, y2 = (float(value) for value in box)
                x1 = min(max(x1, 0.0), float(image.width - 1))
                y1 = min(max(y1, 0.0), float(image.height - 1))
                x2 = min(max(x2, x1 + 1e-6), float(image.width))
                y2 = min(max(y2, y1 + 1e-6), float(image.height))
                rows.append(
                    DetectionBox(
                        box_id=f"box-{index:04d}",
                        xyxy=(x1, y1, x2, y2),
                        score=float(score),
                        phrase=str(label),
                    )
                )
            result[frame.frame_id] = tuple(rows)
        return result


class Sam2ImageAdapter:
    model_ref = _checkpoint_ref("facebook/sam2.1-hiera-large")

    def __init__(
        self,
        source: Path,
        checkpoint: Path,
        *,
        device: str = "cuda",
    ) -> None:
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint_file = checkpoint / "sam2.1_hiera_large.pt"
        if not checkpoint_file.is_file():
            raise FileNotFoundError(checkpoint_file)
        model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(checkpoint_file),
            device=device,
            apply_postprocessing=True,
        )
        self.model = model
        self.predictor = SAM2ImagePredictor(model)
        self.device = device

    def to(self, device: str):
        self.model.to(device)
        self.device = device
        return self

    def refine_batch(
        self, prompts: tuple[SamBoxPrompt, ...]
    ) -> Mapping[tuple[str, str], object]:
        import torch
        from PIL import Image

        result: dict[tuple[str, str], object] = {}
        for prompt in prompts:
            with Image.open(prompt.frame.path) as opened:
                pixels = np.asarray(opened.convert("RGB"))
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
            ):
                self.predictor.set_image(pixels)
                masks, scores, _ = self.predictor.predict(
                    box=np.asarray(prompt.box.xyxy, dtype=np.float32),
                    multimask_output=True,
                )
            best = int(np.argmax(scores))
            result[(prompt.frame.frame_id, prompt.box.box_id)] = np.asarray(
                masks[best], dtype=np.bool_
            )
        return result

    def propagate(
        self,
        frames: tuple[FrameArtifact, ...],
        direct_masks: Mapping[str, object],
        *,
        clip_frames: int,
    ) -> Mapping[str, tuple[object, tuple[str, ...]]]:
        del clip_frames
        result: dict[str, tuple[object, tuple[str, ...]]] = {}
        for frame in frames:
            direct = np.asarray(direct_masks[frame.frame_id], dtype=np.bool_)
            support = (frame.frame_id,) if np.any(direct) else ()
            result[frame.frame_id] = (np.array(direct, copy=True), support)
        return result


def _sea_args(source: Path) -> Namespace:
    payload = json.loads(
        (source / "config" / "eval" / "spring-M.json").read_text(encoding="utf-8")
    )
    return Namespace(**payload)


def _is_batch_counter_consistency_assertion(error: AssertionError) -> bool:
    """Recognize safetensors' alias check for non-persistent BN counters only."""
    left, separator, right = str(error).partition(" != ")
    if not separator or right != "set()":
        return False
    try:
        keys = ast.literal_eval(left)
    except (SyntaxError, ValueError):
        return False
    return (
        isinstance(keys, set)
        and bool(keys)
        and all(
            isinstance(key, str) and key.endswith(".num_batches_tracked")
            for key in keys
        )
    )


class SeaRaftTorchAdapter:
    model_id = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
    revision = _checkpoint_ref(model_id).revision
    code_commit = _checkpoint_ref(model_id).code_commit

    def __init__(
        self,
        source: Path,
        checkpoint: Path,
        *,
        device: str = "cuda",
    ) -> None:
        core = str(source / "core")
        root = str(source)
        for entry in (core, root):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        from raft import RAFT
        from safetensors.torch import load_model

        self.args = _sea_args(source)
        checkpoint_file = checkpoint / "model.safetensors"
        if not checkpoint_file.is_file():
            raise FileNotFoundError(checkpoint_file)
        self.model = RAFT(self.args)
        try:
            # load_model restores safetensors shared-tensor aliases that load_file
            # cannot represent. Strict mode verifies all real state before its
            # post-load PyTorch consistency assertion is evaluated.
            load_model(self.model, checkpoint_file, strict=True, device="cpu")
        except AssertionError as error:
            if not _is_batch_counter_consistency_assertion(error):
                raise RuntimeError(
                    "SEA-RAFT checkpoint failed alias consistency validation: "
                    f"{error}"
                ) from error
            # The checkpoint aliases all learned BatchNorm state correctly. Its
            # four integer tracking counters are recreated by PyTorch and are not
            # learned parameters, so this safetensors sanity-check mismatch is safe.
        except RuntimeError:
            # Preserve safetensors' strict missing/unexpected/invalid-state error.
            raise
        except Exception as error:
            raise RuntimeError(
                f"SEA-RAFT checkpoint could not be loaded: {error}"
            ) from error
        self.model.to(device)
        self.model.eval()
        self.device = device

    def to(self, device: str):
        self.model.to(device)
        self.device = device
        return self

    def _infer(self, first, second):
        import torch
        import torch.nn.functional as F

        source_size = first.shape[-2:]
        scale = float(2 ** int(self.args.scale))
        if scale != 1.0:
            first = F.interpolate(
                first, scale_factor=scale, mode="bilinear", align_corners=False
            )
            second = F.interpolate(
                second, scale_factor=scale, mode="bilinear", align_corners=False
            )
        with torch.inference_mode(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")
        ):
            output = self.model(
                first,
                second,
                iters=int(self.args.iters),
                test_mode=True,
            )
        flow = output["flow"][-1].float()
        info = output["info"][-1].float()
        weight = info[:, :2].softmax(dim=1)
        raw = info[:, 2:]
        log_b = torch.stack(
            (
                raw[:, 0].clamp(min=0.0, max=float(self.args.var_max)),
                raw[:, 1].clamp(min=float(self.args.var_min), max=0.0),
            ),
            dim=1,
        )
        uncertainty = torch.exp((log_b * weight).sum(dim=1))
        if flow.shape[-2:] != source_size:
            flow = (
                F.interpolate(
                    flow, size=source_size, mode="bilinear", align_corners=False
                )
                / scale
            )
            uncertainty = F.interpolate(
                uncertainty.unsqueeze(1),
                size=source_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return flow, uncertainty

    def infer_bidirectional(
        self,
        pairs: tuple[FlowPairRequest, ...],
        *,
        batch_size: int,
        progress: FlowProgressCallback | None = None,
    ) -> tuple[SeaRaftPairPrediction, ...]:
        import torch
        from PIL import Image

        predictions: list[SeaRaftPairPrediction] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            first_images = []
            second_images = []
            for pair in batch:
                with Image.open(pair.source_path) as opened:
                    first_images.append(
                        np.asarray(opened.convert("RGB"), dtype=np.float32)
                    )
                with Image.open(pair.target_path) as opened:
                    second_images.append(
                        np.asarray(opened.convert("RGB"), dtype=np.float32)
                    )
            first = (
                torch.from_numpy(np.stack(first_images))
                .permute(0, 3, 1, 2)
                .to(self.device)
            )
            second = (
                torch.from_numpy(np.stack(second_images))
                .permute(0, 3, 1, 2)
                .to(self.device)
            )
            forward, forward_uncertainty = self._infer(first, second)
            backward, backward_uncertainty = self._infer(second, first)
            for index, pair in enumerate(batch):
                predictions.append(
                    SeaRaftPairPrediction(
                        source_frame_id=pair.source_frame_id,
                        target_frame_id=pair.target_frame_id,
                        forward_flow=forward[index].permute(1, 2, 0).cpu().numpy(),
                        backward_flow=backward[index].permute(1, 2, 0).cpu().numpy(),
                        forward_uncertainty=forward_uncertainty[index].cpu().numpy(),
                        backward_uncertainty=backward_uncertainty[index].cpu().numpy(),
                    )
                )
            if progress is not None:
                progress(
                    "inference",
                    len(predictions),
                    len(pairs),
                    {"batch_size": len(batch)},
                )
        return tuple(predictions)


def load_da3_model(source: Path, checkpoint: Path):
    source_src = str(source / "src")
    if source_src not in sys.path:
        sys.path.insert(0, source_src)
    from depth_anything_3.api import DepthAnything3

    return DepthAnything3.from_pretrained(str(checkpoint), local_files_only=True).to(
        "cuda"
    )
