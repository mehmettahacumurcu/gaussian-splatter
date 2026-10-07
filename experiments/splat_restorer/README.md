# Splat Restorer (pilot)

Question: can a single-step diffusion model (Difix3D architecture on SD-Turbo)
clean up Gaussian-splat render artefacts well enough to improve the splat
itself? This folder holds the pilot that answers it on Colab.

## Run it

1. Open `colab/splat_restorer_pilot.ipynb` in Colab (it embeds all code; no
   repo checkout needed). GPU runtime: A100 / H100 or larger.
2. Add a Colab secret `HF_TOKEN` (Hugging Face token with DL3DV access) and
   enable notebook access for it.
3. Run all. Every stage resumes after a disconnect; just run all again.

After changing anything in `splat_restorer/`, rebuild the notebook:

```bash
python experiments/splat_restorer/build_notebook.py
```

## Quick test: pretrained Difix on a render video

`colab/difix_video_test.ipynb` runs NVIDIA's released `nvidia/difix` (or
`difix_ref`) on every frame of a splat render video, with no training. Their
weights load strictly into `Restorer` (`Restorer.from_difix`), so it uses the
same pinned libraries as the pilot. Output: side-by-side video, zoomed
comparison sheet and change / sharpness / flicker numbers (`video.py`). Rebuild
with `python experiments/splat_restorer/build_video_notebook.py`.

## Pipeline

| Stage | Module | Output on Drive (`MyDrive/splat_restorer/`) |
|---|---|---|
| Scene split (stratified by bounded/unbounded) | `dl3dv.py`, `stages.py` | `split.json` |
| Weak splats + (render, photo, reference) pairs | `gs_trainer.py`, `pairs.py` | `pairs/<scene>.tar`, `splats/<test scene>/` |
| Restorer training (10/25/50 scenes, ref / no-ref) | `restorer_model.py`, `train_restorer.py` | `runs/restorer_*/` |
| Small U-Net baseline | `baseline.py` | `runs/cnn_*/` |
| Image metrics on unseen test scenes | `evaluate.py` | `results/eval_pilot/` |
| Distill-back: fine-tune splat with restored views | `distill.py` | `results/distill_pilot/` |

Degradation recipes (`splits.py`): `sparse_hard` trains on every 12th frame,
`sparse_mild` on every 4th, 4000 steps each. Every 8th frame is reserved for
evaluation and never trained on by any recipe.

## Reading the results

* `eval_pilot/summary.md`: a method helps only if LPIPS drops **without** a
  clear PSNR drop. Compare `restorer_ref_n10 → n25 → n50` for the scaling
  trend, `ref` vs `noref` for the reference image, and `cnn_*` to see whether
  the diffusion prior is needed at all.
* `distill_pilot/summary.md`: the decision table. Conditions beat `control`
  only if the splat itself got better on never-trained frames.
* Known risk: with a reference view a model can paste the reference instead
  of restoring (seen with the CNN in a local smoke test). Evaluating on unseen
  scenes and the no-ref ablation expose this.

## Licences

`restorer_model.py` adapts NVIDIA Difix3D code under the NVIDIA License
(`splat_restorer/LICENSE-DIFIX3D.txt`): non-commercial research or evaluation
use only. SD-Turbo weights follow the Stability AI Community License. DL3DV
data follows its own terms of use.
