from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Hashable

import torch

from backend.model.gaussian_model import GaussianModel


GIB = 1024**3
MAX_EMERGENCY_GAUSSIANS = 6_000_000


@dataclass(frozen=True)
class AdaptiveDensityPolicy:
    hard_cap: int = MAX_EMERGENCY_GAUSSIANS
    grad_threshold: float = 1e-4
    split_scale_threshold: float = 0.01
    min_opacity: float = 0.005
    max_scale: float = 0.1
    reserve_min_bytes: int = 10 * GIB
    reserve_fraction: float = 0.15
    analytical_bytes_per_gaussian: int = 2048
    calibration_growth_limit: int = 10_000
    low_candidate_patience: int = 3
    minimum_quality_delta_db: float = 0.02
    quality_plateau_patience: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.hard_cap <= MAX_EMERGENCY_GAUSSIANS:
            raise ValueError("hard_cap must be within the 6,000,000 emergency ceiling")
        if not math.isfinite(self.grad_threshold) or self.grad_threshold <= 0:
            raise ValueError("grad_threshold must be finite and positive")
        if (
            not math.isfinite(self.split_scale_threshold)
            or self.split_scale_threshold <= 0
        ):
            raise ValueError("split_scale_threshold must be finite and positive")
        if not 0.0 <= self.min_opacity <= 1.0:
            raise ValueError("min_opacity must be within [0, 1]")
        if not math.isfinite(self.max_scale) or self.max_scale <= 0:
            raise ValueError("max_scale must be finite and positive")
        if self.reserve_min_bytes < 0:
            raise ValueError("reserve_min_bytes must be non-negative")
        if not 0.0 <= self.reserve_fraction < 1.0:
            raise ValueError("reserve_fraction must be within [0, 1)")
        if self.analytical_bytes_per_gaussian <= 0:
            raise ValueError("analytical_bytes_per_gaussian must be positive")
        if self.calibration_growth_limit <= 0:
            raise ValueError("calibration_growth_limit must be positive")
        if self.low_candidate_patience <= 0 or self.quality_plateau_patience <= 0:
            raise ValueError("patience values must be positive")
        if self.minimum_quality_delta_db < 0:
            raise ValueError("minimum_quality_delta_db must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MemoryProbe = Callable[[], tuple[int | float, int | float]]


class AdaptiveDensityController:
    """Prune-first, cross-view density control with a measured VRAM budget."""

    def __init__(
        self,
        policy: AdaptiveDensityPolicy | None = None,
        *,
        memory_probe: MemoryProbe | None = None,
    ) -> None:
        self.policy = policy or AdaptiveDensityPolicy()
        self._memory_probe = memory_probe or self._cuda_memory_probe
        self._grad_accum: torch.Tensor | None = None
        self._observation_count: torch.Tensor | None = None
        self._first_view: torch.Tensor | None = None
        self._second_view: torch.Tensor | None = None
        self._view_ids: dict[Hashable, int] = {}
        self._calibrated_bytes: int | None = None
        self._low_candidate_windows = 0
        self._growth_stopped_reason: str | None = None
        self._quality_resolution: tuple[int, int] | None = None
        self._last_quality: float | None = None
        self._quality_low_delta_count = 0
        self.history: list[dict[str, object]] = []

    @staticmethod
    def _cuda_memory_probe() -> tuple[int, int]:
        if not torch.cuda.is_available():
            return (0, 0)
        free, total = torch.cuda.mem_get_info()
        return int(free), int(total)

    @property
    def growth_stopped(self) -> bool:
        return self._growth_stopped_reason is not None

    @property
    def growth_stopped_reason(self) -> str | None:
        return self._growth_stopped_reason

    @property
    def quality_low_delta_count(self) -> int:
        return self._quality_low_delta_count

    def _reset_accumulators(self, count: int, device: torch.device) -> None:
        self._grad_accum = torch.zeros(count, dtype=torch.float64, device=device)
        self._observation_count = torch.zeros(count, dtype=torch.int64, device=device)
        self._first_view = torch.full((count,), -1, dtype=torch.int64, device=device)
        self._second_view = torch.zeros(count, dtype=torch.bool, device=device)

    def _ensure_accumulators(self, model: GaussianModel) -> None:
        if self._grad_accum is None or len(self._grad_accum) != model.num_points:
            self._reset_accumulators(model.num_points, model.means.device)

    def _encoded_view(self, view_id: Hashable) -> int:
        try:
            existing = self._view_ids.get(view_id)
        except TypeError as exc:
            raise ValueError("view_id must be hashable") from exc
        if existing is not None:
            return existing
        encoded = len(self._view_ids)
        self._view_ids[view_id] = encoded
        return encoded

    def accumulate_view(self, model: GaussianModel, *, view_id: Hashable) -> None:
        if model.means.grad is None:
            return
        self._ensure_accumulators(model)
        encoded = self._encoded_view(view_id)
        gradient = model.means.grad.detach().norm(dim=-1).to(torch.float64)
        visible = torch.isfinite(gradient) & (gradient > 1e-12)
        self._grad_accum[visible] += gradient[visible]
        self._observation_count[visible] += 1
        unseen = visible & (self._first_view < 0)
        self._first_view[unseen] = encoded
        distinct = visible & (self._first_view >= 0) & (self._first_view != encoded)
        self._second_view[distinct] = True

    def _memory_snapshot(self) -> tuple[int, int] | None:
        try:
            free_raw, total_raw = self._memory_probe()
            free = float(free_raw)
            total = float(total_raw)
        except Exception:
            return None
        if (
            not math.isfinite(free)
            or not math.isfinite(total)
            or free < 0
            or total <= 0
            or free > total
        ):
            return None
        return int(free), int(total)

    def _filter_state(self, keep: torch.Tensor) -> None:
        self._grad_accum = self._grad_accum[keep]
        self._observation_count = self._observation_count[keep]
        self._first_view = self._first_view[keep]
        self._second_view = self._second_view[keep]

    @staticmethod
    def _apply_keep(
        model: GaussianModel,
        keep: torch.Tensor,
        optimizer: torch.optim.Optimizer | None,
    ) -> None:
        if optimizer is None:
            model._apply_mask(keep)
        else:
            model._apply_mask_keep_optimizer(keep, optimizer)

    @staticmethod
    def _append(
        model: GaussianModel,
        optimizer: torch.optim.Optimizer | None,
        indices: torch.Tensor,
    ) -> None:
        fourier = (
            model.fourier_pos_coeffs[indices].detach().clone()
            if model.fourier_pos_coeffs is not None
            else None
        )
        values = (
            model.means[indices].detach().clone(),
            model.scales[indices].detach().clone(),
            model.quats[indices].detach().clone(),
            model.opacities[indices].detach().clone(),
            model.sh_dc[indices].detach().clone(),
            model.sh_rest[indices].detach().clone(),
        )
        if optimizer is None:
            model.append_gaussians(*values, new_fourier_coeffs=fourier)
        else:
            model._append_keep_optimizer(optimizer, *values, new_fourier_coeffs=fourier)

    @staticmethod
    def _split(
        model: GaussianModel,
        optimizer: torch.optim.Optimizer | None,
        indices: torch.Tensor,
    ) -> None:
        count = len(indices)
        if count == 0:
            return
        scales = model.get_scales[indices].detach()
        axes = scales.argmax(dim=-1)
        offsets = torch.zeros((count, 3), device=model.means.device)
        offsets[torch.arange(count, device=model.means.device), axes] = (
            scales[torch.arange(count, device=model.means.device), axes] * 0.5
        )
        new_means = torch.stack(
            (
                model.means[indices].detach() - offsets,
                model.means[indices].detach() + offsets,
            ),
            dim=1,
        ).reshape(-1, 3)
        new_scales = model.scales[indices].detach().repeat_interleave(
            2, dim=0
        ) - math.log(1.6)
        new_quats = model.quats[indices].detach().repeat_interleave(2, dim=0)
        new_opacities = model.opacities[indices].detach().repeat_interleave(2, dim=0)
        new_sh_dc = model.sh_dc[indices].detach().repeat_interleave(2, dim=0)
        new_sh_rest = model.sh_rest[indices].detach().repeat_interleave(2, dim=0)
        new_fourier = (
            model.fourier_pos_coeffs[indices].detach().repeat_interleave(2, dim=0)
            if model.fourier_pos_coeffs is not None
            else None
        )
        keep = torch.ones(model.num_points, dtype=torch.bool, device=model.means.device)
        keep[indices] = False
        AdaptiveDensityController._apply_keep(model, keep, optimizer)
        values = (
            new_means,
            new_scales,
            new_quats,
            new_opacities,
            new_sh_dc,
            new_sh_rest,
        )
        if optimizer is None:
            model.append_gaussians(*values, new_fourier_coeffs=new_fourier)
        else:
            model._append_keep_optimizer(
                optimizer, *values, new_fourier_coeffs=new_fourier
            )

    @torch.no_grad()
    def step_at(
        self,
        model: GaussianModel,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        dynamic_densify_scale: float = 1.0,
        iteration: int,
    ) -> dict[str, object]:
        if (
            isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration <= 0
        ):
            raise ValueError("iteration must be a positive integer")
        self._ensure_accumulators(model)
        before = model.num_points

        keep = (model.get_opacities.squeeze(-1) > self.policy.min_opacity) & (
            model.get_scales.max(dim=-1).values < self.policy.max_scale
        )
        pruned = int((~keep).sum().item())
        if pruned:
            self._filter_state(keep)
            self._apply_keep(model, keep, optimizer)
        post_prune = model.num_points

        observations = self._observation_count.to(torch.float64).clamp_min(1.0)
        average_gradient = self._grad_accum / observations
        threshold = torch.full_like(average_gradient, self.policy.grad_threshold)
        if (
            dynamic_densify_scale != 1.0
            and hasattr(model, "is_dynamic")
            and model.is_dynamic.numel() == post_prune
        ):
            threshold = torch.where(
                model.is_dynamic,
                threshold * float(dynamic_densify_scale),
                threshold,
            )
        eligible = (average_gradient > threshold) & self._second_view
        eligible_indices = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
        eligible_count = int(len(eligible_indices))

        if eligible_count == 0 and not self.growth_stopped:
            self._low_candidate_windows += 1
            if self._low_candidate_windows >= self.policy.low_candidate_patience:
                self._growth_stopped_reason = "low_candidate_patience"
        elif eligible_count > 0:
            self._low_candidate_windows = 0

        memory_before = self._memory_snapshot()
        if memory_before is None and not self.growth_stopped:
            self._growth_stopped_reason = "invalid_memory_calibration"

        bytes_per_gaussian = (
            self._calibrated_bytes
            if self._calibrated_bytes is not None
            else self.policy.analytical_bytes_per_gaussian
        )
        vram_slots = 0
        if memory_before is not None:
            free, total = memory_before
            reserve = max(
                self.policy.reserve_min_bytes,
                math.ceil(total * self.policy.reserve_fraction),
            )
            vram_slots = max(0, free - reserve) // bytes_per_gaussian

        selected_count = 0
        clone_count = 0
        split_count = 0
        if not self.growth_stopped and eligible_count:
            cap_slots = max(0, self.policy.hard_cap - post_prune)
            calibration_slots = (
                self.policy.calibration_growth_limit
                if self._calibrated_bytes is None
                else eligible_count
            )
            selected_count = int(
                min(eligible_count, cap_slots, vram_slots, calibration_slots)
            )
            ranked = sorted(
                eligible_indices.tolist(),
                key=lambda index: (-float(average_gradient[index].item()), index),
            )
            selected = torch.tensor(
                ranked[:selected_count], dtype=torch.long, device=model.means.device
            )
            if selected_count:
                small = (
                    model.get_scales[selected].max(dim=-1).values
                    <= self.policy.split_scale_threshold
                )
                clone_indices = selected[small]
                split_indices = selected[~small]
                clone_count = int(len(clone_indices))
                split_count = int(len(split_indices))
                if clone_count:
                    self._append(model, optimizer, clone_indices)
                if split_count:
                    self._split(model, optimizer, split_indices)

        calibrated = self._calibrated_bytes
        if selected_count and self._calibrated_bytes is None:
            memory_after = self._memory_snapshot()
            if memory_after is None:
                self._growth_stopped_reason = "invalid_memory_calibration"
            else:
                consumed = memory_before[0] - memory_after[0]
                observed = consumed / selected_count
                if not math.isfinite(observed) or observed < 0:
                    self._growth_stopped_reason = "invalid_memory_calibration"
                else:
                    calibrated = math.ceil(
                        max(self.policy.analytical_bytes_per_gaussian, observed) * 1.5
                    )
                    self._calibrated_bytes = calibrated

        event: dict[str, object] = {
            "type": "density",
            "iteration": iteration,
            "before": before,
            "pruned": pruned,
            "post_prune": post_prune,
            "eligible": eligible_count,
            "vram_growth_slots": int(vram_slots),
            "selected": selected_count,
            "cloned": clone_count,
            "split": split_count,
            "after": model.num_points,
            "calibrated_bytes_per_gaussian": calibrated,
            "growth_stopped_reason": self._growth_stopped_reason,
        }
        self.history.append(event)
        self._reset_accumulators(model.num_points, model.means.device)
        return event

    def record_quality(
        self,
        iteration: int,
        aggregate_valid_psnr_db: float,
        resolution: tuple[int, int],
    ) -> None:
        if not math.isfinite(aggregate_valid_psnr_db):
            raise ValueError("aggregate_valid_psnr_db must be finite")
        normalized_resolution = (int(resolution[0]), int(resolution[1]))
        if normalized_resolution[0] <= 0 or normalized_resolution[1] <= 0:
            raise ValueError("resolution must be positive")
        if self._quality_resolution != normalized_resolution:
            self._quality_resolution = normalized_resolution
            self._last_quality = aggregate_valid_psnr_db
            self._quality_low_delta_count = 0
            if self._growth_stopped_reason == "quality_plateau":
                self._growth_stopped_reason = None
            delta = None
        else:
            delta = aggregate_valid_psnr_db - self._last_quality
            self._last_quality = aggregate_valid_psnr_db
            if delta < self.policy.minimum_quality_delta_db:
                self._quality_low_delta_count += 1
            else:
                self._quality_low_delta_count = 0
            if (
                self._quality_low_delta_count >= self.policy.quality_plateau_patience
                and self._growth_stopped_reason is None
            ):
                self._growth_stopped_reason = "quality_plateau"
        self.history.append(
            {
                "type": "quality",
                "iteration": int(iteration),
                "aggregate_valid_psnr_db": float(aggregate_valid_psnr_db),
                "resolution": list(normalized_resolution),
                "delta_db": None if delta is None else float(delta),
                "low_delta_count": self._quality_low_delta_count,
                "growth_stopped_reason": self._growth_stopped_reason,
            }
        )
