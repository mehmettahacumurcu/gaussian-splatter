"""Modern image/background process, isolated from TRELLIS's Torch 2.4 environment."""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

if __package__:
    from .text_to_splat_helpers import StageCache, atomic_json, file_sha256
    from .text_to_splat_runtime import provenance_for
    from .text_to_splat_settings import BACKGROUND_MODELS, IMAGE_MODELS, estimate_requirements, model_provenance, validate_settings
else:
    from text_to_splat_helpers import StageCache, atomic_json, file_sha256
    from text_to_splat_runtime import provenance_for
    from text_to_splat_settings import BACKGROUND_MODELS, IMAGE_MODELS, estimate_requirements, model_provenance, validate_settings


def check_model_access(config):
    """HEAD requests only, including DINOv3, before any image/model downloads."""
    from huggingface_hub import hf_hub_url, get_hf_file_metadata
    models = model_provenance(config)
    selected = [(models['image'], 'model_index.json')]
    if config['reconstruction_model'] == 'trellis2':
        selected.append((models['encoder'], 'model.safetensors'))
    token = os.environ.get('HF_TOKEN')
    for model, filename in selected:
        if model.get('gated') and not token:
            raise RuntimeError(f'{model["id"]}: Hugging Face erişim/lisans koşullarını kabul edin (DINOv3 için Meta onayı gerekir) ve Colab Secrets içinde HF_TOKEN tanımlayın.')
        try:
            get_hf_file_metadata(hf_hub_url(model['id'], filename, revision=model['revision']), token=token)
        except Exception:
            # No request objects, authorization headers or secrets in logs.
            raise RuntimeError(f'{model["id"]} erişimi doğrulanamadı. Bağlantıyı, HF_TOKEN yetkisini ve lisans/Meta onayını kontrol edin.') from None


def image_preflight(config, *, check_access=True):
    import torch
    if sys.version_info[:2] != (3, 11) or torch.__version__.split('+')[0] != '2.7.1' or torch.version.cuda != '12.8':
        raise RuntimeError('Görüntü ortamı Python 3.11 / Torch 2.7.1 cu128 olmalı; kurulum hücresini tekrar çalıştırın.')
    if not torch.cuda.is_available():
        raise RuntimeError('Görüntü adımı için CUDA GPU bulunamadı. GPU oturumu seçin.')
    minimum = estimate_requirements(config)['image_vram']
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = torch.cuda.mem_get_info()[0] / 1024**3
    if total < minimum or free < minimum - 2:
        raise RuntimeError(f'Görüntü için VRAM yetersiz: toplam {total:.1f}, boş {free:.1f} GiB; seçiminiz en az {minimum} GiB ister.')
    # Imports and a real CUDA kernel fail before downloading any model weights.
    import diffusers
    expected = {'sdxl': 'StableDiffusionXLPipeline', 'flux1_dev': 'FluxPipeline',
                'flux1_schnell': 'FluxPipeline', 'flux2_klein_4b': 'Flux2KleinPipeline',
                'qwen_image': 'QwenImagePipeline'}
    getattr(diffusers, expected[config['image_model']])
    torch.ones((16, 16), device='cuda') @ torch.ones((16, 16), device='cuda')
    torch.cuda.synchronize()
    if check_access:
        check_model_access(config)
    return {'name': torch.cuda.get_device_name(0), 'total_gib': total, 'free_gib_at_start': free}


def inference_arguments(config, prompt, negative):
    """Keep model-specific guidance semantics explicit and CPU-testable."""
    choice = config['image_model']
    kwargs = {'prompt': prompt, 'num_inference_steps': config['image_steps'],
              'height': config['image_resolution'], 'width': config['image_resolution']}
    if choice == 'sdxl':
        kwargs.update(negative_prompt=negative, guidance_scale=config['image_guidance'])
    elif choice == 'qwen_image':
        kwargs.update(negative_prompt=negative or ' ', true_cfg_scale=config['image_guidance'])
    elif choice == 'flux1_dev':
        kwargs.update(guidance_scale=config['image_guidance'], max_sequence_length=512)
        if negative:
            kwargs.update(negative_prompt=negative, true_cfg_scale=config['image_guidance'])
    elif choice == 'flux1_schnell':
        kwargs.update(guidance_scale=0.0, max_sequence_length=256)
    elif choice == 'flux2_klein_4b':
        kwargs.update(guidance_scale=1.0)
    else:
        raise ValueError('Bilinmeyen görüntü modeli.')
    return kwargs


def remove_background(image, config):
    import numpy as np
    from PIL import Image
    if config['background_model'] == 'u2net':
        import rembg
        session = rembg.new_session('u2net', providers=['CPUExecutionProvider'])
        cutout = rembg.remove(image, session=session).convert('RGBA')
        metadata = {'weights_sha256': file_sha256(Path(os.environ['U2NET_HOME']) / 'u2net.onnx')}
    else:
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
        model = BACKGROUND_MODELS['birefnet']
        # This allow-listed official implementation and its weights are pinned to
        # the SAME commit; no user-supplied code/model repository is accepted.
        net = AutoModelForImageSegmentation.from_pretrained(model['id'], revision=model['revision'],
            code_revision=model['revision'], trust_remote_code=True, use_safetensors=True,
            token=os.environ.get('HF_TOKEN')).to('cuda').eval().half()
        transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        tensor = transform(image.convert('RGB')).unsqueeze(0).to('cuda').half()
        with torch.inference_mode():
            mask = net(tensor)[-1].sigmoid()[0].squeeze().float().cpu().numpy()
        alpha = Image.fromarray((np.clip(mask, 0, 1) * 255).astype('uint8')).resize(image.size, Image.Resampling.LANCZOS)
        cutout = image.convert('RGBA')
        cutout.putalpha(alpha)
        del net, tensor
        torch.cuda.empty_cache()
        metadata = {'model_id': model['id'], 'revision': model['revision'], 'code_revision': model['revision']}
    alpha = np.asarray(cutout)[:, :, 3]
    coords = np.argwhere(alpha > 204)
    if len(coords) < 64 or np.any(np.ptp(coords, axis=0) < 8):
        raise RuntimeError('Arka plan temizlemede nesne bulunamadı. İstemi sadeleştirin ve yeni OUTPUT_DIR seçin.')
    if np.all(alpha == 255):
        raise RuntimeError('Maske tüm görüntüyü kaplıyor. Tek nesne/beyaz arka plan ile yeniden deneyin.')
    return cutout, metadata


def generate_image(output, config):
    import torch
    import diffusers
    from PIL import Image
    choice = config['image_model']
    model = IMAGE_MODELS[choice]
    names = {'sdxl': 'StableDiffusionXLPipeline', 'flux1_dev': 'FluxPipeline',
             'flux1_schnell': 'FluxPipeline', 'flux2_klein_4b': 'Flux2KleinPipeline',
             'qwen_image': 'QwenImagePipeline'}
    options = {'revision': model['revision'], 'torch_dtype': torch.float16 if choice == 'sdxl' else torch.bfloat16,
               'use_safetensors': True, 'token': os.environ.get('HF_TOKEN')}
    if choice == 'sdxl':
        options['variant'] = 'fp16'
    pipe = getattr(diffusers, names[choice]).from_pretrained(model['id'], **options)
    pipe.enable_model_cpu_offload()
    if hasattr(pipe.vae, 'enable_slicing'):
        pipe.vae.enable_slicing()
    if hasattr(pipe.vae, 'enable_tiling'):
        pipe.vae.enable_tiling()
    prompt = '. '.join(filter(None, (config['prompt'], config['style'])))
    prompt += '. Single isolated object, full object visible, three-quarter view, plain white background, studio lighting, no text.'
    # FLUX dev's true CFG is used only when the user actually asks for a
    # negative prompt; SDXL/Qwen retain the framing defaults.
    negative = config['negative_prompt'] if model['negative_prompt'] else ''
    if choice in ('sdxl', 'qwen_image'):
        negative = ', '.join(filter(None, (negative, 'cropped, out of frame, multiple objects, text, watermark, blurry, flat illustration')))
    kwargs = inference_arguments(config, prompt, negative)
    print(f'Görüntü: {model["id"]} · {config["image_steps"]} adım · {config["image_resolution"]} px. CPU offload daha yavaş, VRAM tüketimi daha düşük.', flush=True)
    with torch.inference_mode():
        image = pipe(**kwargs, generator=torch.Generator(device='cpu').manual_seed(config['seed'])).images[0]
    image.save(output / 'input.png')
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    cutout, background_metadata = remove_background(image, config)
    cutout.save(output / 'cutout.png')
    for name in ('input.png', 'cutout.png'):
        with Image.open(output / name) as check:
            check.verify()
    return {'effective_prompt': prompt, 'effective_negative_prompt': negative,
            'inference_arguments': kwargs, 'background': background_metadata}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['image', 'check'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = validate_settings(json.loads(Path(args.config).read_text(encoding='utf8')))
    gpu = image_preflight(config)
    if args.stage == 'check':
        print('✓ Modern görüntü ortamı, CUDA ve model erişim kontrolü başarılı; ağırlık indirilmedi.', flush=True)
        return
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = StageCache(config['output_dir'], config, provenance_for(config))
    cache.data.setdefault('environments', {})['image'] = {'python': platform.python_version(),
        'torch_cuda': torch.version.cuda, 'gpu': gpu, 'packages': {
            dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata['Name']}}
    atomic_json(cache.path, cache.data)
    cache.run('image', ['input.png', 'cutout.png'], lambda: generate_image(cache.output, config))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'Görüntü adımı durdu: {exc}. Aynı ayarlarla yeniden deneyebilirsiniz; ayarlar değişirse yeni OUTPUT_DIR seçin.', file=sys.stderr, flush=True)
        raise
