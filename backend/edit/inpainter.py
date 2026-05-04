"""Inpainting backends with a common interface.

LaMaInpainter — fast preview, ~200 MB. Good for backgrounds and small objects.
SDInpainter   — quality, ~4 GB peak VRAM. Good for hallucinated content.

Both support .load() / .unload() so the caller can sequence model swaps
within an 8 GB VRAM budget.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch


class InpainterBase(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def inpaint(
        self,
        image_rgb: np.ndarray,  # (H, W, 3) uint8
        mask: np.ndarray,       # (H, W) uint8 — 1 = inpaint here
    ) -> np.ndarray:           # (H, W, 3) uint8 — RGB filled
        ...


class LaMaInpainter(InpainterBase):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model: Optional[object] = None

    def load(self) -> None:
        if self._model is not None:
            return
        # Lazy import — module file must remain importable without simple_lama_inpainting
        from simple_lama_inpainting import SimpleLama
        self._model = SimpleLama(device=self.device)
        print(f"[inpainter.lama] LaMa yuklendi (device={self.device})")

    def unload(self) -> None:
        if self._model is None:
            return
        del self._model
        self._model = None
        torch.cuda.empty_cache()
        print("[inpainter.lama] LaMa unloaded")

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._model is None:
            self.load()
        from PIL import Image
        img_pil = Image.fromarray(image_rgb)
        # SimpleLama expects mask as PIL with 255=inpaint
        mask_pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        result_pil = self._model(img_pil, mask_pil)  # returns PIL RGB
        return np.array(result_pil, dtype=np.uint8)


class SDInpainter(InpainterBase):
    def __init__(
        self,
        device: str = "cuda",
        model_id: str = "runwayml/stable-diffusion-inpainting",
        steps: int = 30,
    ):
        self.device = device
        self.model_id = model_id
        self.steps = steps
        self._pipe: Optional[object] = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        # Lazy import
        from diffusers import StableDiffusionInpaintPipeline
        self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        # Memory-efficient attention if available
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        # Slice attention to keep peak VRAM lower
        self._pipe.enable_attention_slicing()
        print(f"[inpainter.sd] SD inpaint yuklendi (model={self.model_id}, device={self.device})")

    def unload(self) -> None:
        if self._pipe is None:
            return
        del self._pipe
        self._pipe = None
        torch.cuda.empty_cache()
        print("[inpainter.sd] SD inpaint unloaded")

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._pipe is None:
            self.load()
        from PIL import Image
        H, W = image_rgb.shape[:2]
        # SD inpaint expects 8x-aligned dimensions; pad/crop to nearest 8
        Hp = (H // 8) * 8
        Wp = (W // 8) * 8
        img_pil = Image.fromarray(image_rgb).resize((Wp, Hp), Image.LANCZOS)
        # SD expects 255=inpaint in mask
        mask_pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize((Wp, Hp), Image.NEAREST)
        prompt = ""  # empty prompt → use surrounding context only (no hallucination guidance)
        result = self._pipe(
            prompt=prompt,
            image=img_pil,
            mask_image=mask_pil,
            num_inference_steps=self.steps,
            guidance_scale=7.5,
        ).images[0]
        # Resize back to original
        result = result.resize((W, H), Image.LANCZOS)
        return np.array(result, dtype=np.uint8)
