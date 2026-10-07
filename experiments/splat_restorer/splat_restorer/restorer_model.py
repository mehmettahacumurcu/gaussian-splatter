"""Single-step diffusion restorer (Difix3D architecture).

Adapted from NVIDIA Difix3D (https://github.com/nv-tlabs/Difix3D, src/model.py
and src/mv_unet.py) under the NVIDIA License in LICENSE-DIFIX3D.txt, which
allows non-commercial (research / evaluation) use only.

Changes from the original:
* works with current diffusers / peft (no pinned 0.25 UNet copy): the
  reference-view attention is an attention *processor*, not a forked UNet;
* the text prompt is fixed and encoded once;
* x0 is computed directly from the scheduler's alphas (epsilon or v);
* checkpoints contain the full UNet so a saved model is self-contained.

How it works: the degraded image is VAE-encoded and treated as a latent at
timestep ``t`` (default 199). The SD-Turbo UNet predicts the "noise", x0 is
recovered in one step and decoded. VAE encoder activations are fed to the
decoder through 1x1 skip convs so fine detail survives the round trip. With a
reference view, self-attention runs jointly over both views' tokens.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

BASE_MODEL = "stabilityai/sd-turbo"
PROMPT = "remove degradation"
VAE_LORA_TARGETS = (
    "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
    "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
    "to_k", "to_q", "to_v", "to_out.0",
)


def _vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    skips = []
    for down_block in self.down_blocks:
        skips.append(sample)
        sample = down_block(sample)
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = skips
    return sample


def _vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
    skips = self.incoming_skip_acts[::-1]
    for idx, up_block in enumerate(self.up_blocks):
        if not self.ignore_skip:
            sample = sample + skip_convs[idx](skips[idx] * self.gamma)
        sample = up_block(sample, latent_embeds)
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    return self.conv_out(sample)


class MultiViewSelfAttnProcessor:
    """Self-attention over the tokens of all views of a sample.

    The UNet sees a batch of (B * V) images; for attention the V views of each
    sample are concatenated along the token axis, which lets the degraded view
    borrow detail from the reference view.
    """

    def __init__(self):
        self.num_views = 1

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b0, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b0, c, h * w).transpose(1, 2)
        bv, n, d = hidden_states.shape
        v = self.num_views
        x = hidden_states.reshape(bv // v, v * n, d) if v > 1 else hidden_states
        b = x.shape[0]
        q, k, val = attn.to_q(x), attn.to_k(x), attn.to_v(x)
        heads = attn.heads
        hd = q.shape[-1] // heads
        q, k, val = (t.view(b, -1, heads, hd).transpose(1, 2) for t in (q, k, val))
        out = F.scaled_dot_product_attention(q, k, val)
        out = out.transpose(1, 2).reshape(b, -1, heads * hd).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if v > 1:
            out = out.reshape(bv, n, d)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(b0, c, h, w)
        if getattr(attn, "residual_connection", False):
            out = out + hidden_states
        return out / getattr(attn, "rescale_output_factor", 1.0)


class Restorer(torch.nn.Module):
    def __init__(
        self,
        timestep: int = 199,
        lora_rank_vae: int = 4,
        device: str = "cuda",
        pretrained: bool = True,
        unet_overrides: dict | None = None,
        base_model: str = BASE_MODEL,
        pretrained_text_encoder: bool | None = None,
    ):
        """``pretrained=False`` builds random weights from the configs only, and
        ``unet_overrides`` can shrink that random UNet (tests on small GPUs).
        ``base_model`` is the Hugging Face repo the parts are loaded from.
        ``pretrained_text_encoder`` defaults to ``pretrained``."""
        super().__init__()
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from diffusers.models.attention_processor import AttnProcessor2_0
        from peft import LoraConfig, inject_adapter_in_model
        from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModel

        def load(cls, sub, overrides=None):
            if pretrained:
                return cls.from_pretrained(base_model, subfolder=sub)
            config = dict(cls.load_config(base_model, subfolder=sub))
            config.update(overrides or {})
            return cls.from_config(config)

        tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
        if pretrained if pretrained_text_encoder is None else pretrained_text_encoder:
            text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
        else:
            text_encoder = CLIPTextModel(CLIPTextConfig.from_pretrained(base_model, subfolder="text_encoder"))
        text_encoder = text_encoder.to(device)
        with torch.no_grad():
            ids = tokenizer(PROMPT, max_length=tokenizer.model_max_length, padding="max_length",
                            truncation=True, return_tensors="pt").input_ids.to(device)
            self.register_buffer("prompt_embeds", text_encoder(ids)[0].float(), persistent=False)
        del text_encoder

        sched = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
        self.prediction_type = sched.config.prediction_type
        self.register_buffer("alphas_cumprod", sched.alphas_cumprod.float(), persistent=False)
        self.timestep = int(timestep)

        vae = load(AutoencoderKL, "vae")
        vae.encoder.forward = _vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = _vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        dec = vae.decoder
        dec.skip_conv_1 = torch.nn.Conv2d(512, 512, 1, bias=False)
        dec.skip_conv_2 = torch.nn.Conv2d(256, 512, 1, bias=False)
        dec.skip_conv_3 = torch.nn.Conv2d(128, 512, 1, bias=False)
        dec.skip_conv_4 = torch.nn.Conv2d(128, 256, 1, bias=False)
        for conv in (dec.skip_conv_1, dec.skip_conv_2, dec.skip_conv_3, dec.skip_conv_4):
            torch.nn.init.constant_(conv.weight, 1e-5)
        dec.ignore_skip = False
        dec.gamma = 1.0
        targets = [name for name, _ in dec.named_modules() if name and any(name.endswith(t) for t in VAE_LORA_TARGETS)]
        inject_adapter_in_model(
            LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian", target_modules=targets), dec, adapter_name="vae_skip"
        )
        vae.requires_grad_(False)
        self.vae = vae

        unet = load(UNet2DConditionModel, "unet", unet_overrides)
        self.mv_processor = MultiViewSelfAttnProcessor()
        unet.set_attn_processor(
            {name: (self.mv_processor if name.endswith("attn1.processor") else AttnProcessor2_0()) for name in unet.attn_processors}
        )
        self.unet = unet
        self.to(device)

    @classmethod
    def from_difix(cls, repo: str = "nvidia/difix", device: str = "cuda") -> "Restorer":
        """NVIDIA's released Difix weights ("nvidia/difix" or "nvidia/difix_ref").

        Their VAE (skip convs + "vae_skip" LoRA) and UNet have exactly this
        class's parameter names and shapes, so they load strictly.
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        # Build VAE/UNet from the configs only: Difix's VAE file stores LoRA-wrapped
        # layers (".base_layer."), which a plain AutoencoderKL.from_pretrained rejects.
        model = cls(device=device, base_model=repo, pretrained=False, pretrained_text_encoder=True)
        for part, module in (("vae", model.vae), ("unet", model.unet)):
            path = hf_hub_download(repo, f"{part}/diffusion_pytorch_model.safetensors")
            module.load_state_dict(load_file(path), strict=True)
        model.set_eval()
        return model

    # ----- trainable parameter groups -------------------------------------------------
    def vae_trainable_parameters(self):
        dec = self.vae.decoder
        params = [p for n, p in dec.named_parameters() if "lora_" in n]
        for conv in (dec.skip_conv_1, dec.skip_conv_2, dec.skip_conv_3, dec.skip_conv_4):
            params += [p for n, p in conv.named_parameters() if "lora_" not in n]
        return params

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.unet.requires_grad_(True)
        self.vae.requires_grad_(False)
        for p in self.vae_trainable_parameters():
            p.requires_grad_(True)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.requires_grad_(False)

    # ----- forward ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, sample_latent: bool = True) -> torch.Tensor:
        """x: (B, V, 3, H, W) in [-1, 1]; view 0 is the image to restore,
        view 1 (optional) the reference. Returns (B, V, 3, H, W) in [-1, 1]."""
        b, v = x.shape[:2]
        flat = x.flatten(0, 1)
        dist = self.vae.encode(flat).latent_dist
        z = (dist.sample() if sample_latent else dist.mode()) * self.vae.config.scaling_factor
        self.mv_processor.num_views = v
        t = torch.full((z.shape[0],), self.timestep, device=z.device, dtype=torch.long)
        cond = self.prompt_embeds.expand(z.shape[0], -1, -1).to(z.dtype)
        pred = self.unet(z, t, encoder_hidden_states=cond).sample
        a = self.alphas_cumprod[self.timestep].to(z.dtype)
        if self.prediction_type == "epsilon":
            z0 = (z - (1 - a).sqrt() * pred) / a.sqrt()
        elif self.prediction_type == "v_prediction":
            z0 = a.sqrt() * z - (1 - a).sqrt() * pred
        else:  # "sample"
            z0 = pred
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        out = self.vae.decode(z0 / self.vae.config.scaling_factor).sample.clamp(-1, 1)
        return out.view(b, v, *out.shape[1:])

    @torch.no_grad()
    def restore(self, deg: torch.Tensor, ref: torch.Tensor | None = None) -> torch.Tensor:
        """deg/ref: (B, 3, H, W) in [0, 1], H and W multiples of 8. Returns [0, 1]."""
        x = deg * 2 - 1
        if ref is not None:
            x = torch.stack([x, ref * 2 - 1], dim=1)
        else:
            x = x[:, None]
        # bf16 where the GPU supports it (A100/H100/L4); fp32 otherwise (T4), since
        # the SD VAE can overflow in fp16.
        # (is_bf16_supported() is also True on T4, where bf16 is only emulated and slow.)
        bf16 = x.is_cuda and torch.cuda.get_device_capability(x.device)[0] >= 8
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = self.forward(x, sample_latent=False)[:, 0]
        return (out.float() * 0.5 + 0.5).clamp(0, 1)

    # ----- persistence -----------------------------------------------------------------
    def trainable_state_dict(self) -> dict:
        vae_keys = {n for n, _ in self.vae.named_parameters() if "lora_" in n or "skip_conv" in n}
        return {
            "unet": self.unet.state_dict(),
            "vae": {k: v for k, v in self.vae.state_dict().items() if k in vae_keys},
            "timestep": self.timestep,
        }

    def load_trainable_state_dict(self, sd: dict) -> None:
        self.unet.load_state_dict(sd["unet"])
        missing = self.vae.load_state_dict(sd["vae"], strict=False)
        bad = [k for k in missing.unexpected_keys]
        if bad:
            raise KeyError(f"Unexpected VAE keys in checkpoint: {bad[:5]}")
        self.timestep = int(sd.get("timestep", self.timestep))
