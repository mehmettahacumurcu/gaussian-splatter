# Spirula preprocessing notebooks

Open **Notebook → Pipeline → Spirula dataset hazırlama (yalnız preprocessing)**.

1. Upload videos or a photo folder to Drive. Add each source using its MyDrive-relative path; choose an extraction FPS for each video.
2. Choose the output folder and dataset name. Each Colab execution creates a fresh run subfolder.
3. Select camera, matching, mask and geometry settings. The advanced section exposes overlap, calibration refinement, feature budget, image size and sensor options.
4. Download the notebook, select an NVIDIA GPU in Colab, and run its cells in order. This notebook never starts Gaussian training.
5. Use the printed `dataset.zip` path in a separate training notebook. Review coverage and components first.

The screen takes its field definitions/defaults/validation from the backend. **Hazır ayar olarak kaydet** downloads the complete settings and source paths as JSON. **Hazır ayar yükle** validates that JSON before replacing the current form. These are this app's presets, not Spirula desktop's preset format.

## Outputs

```
MyDrive/<output folder>/<dataset name>_<timestamp>_<id>/
  dataset.zip
  manifest.json
  reports/
    preprocess_settings.json
    extraction.json
    sfm_report.json
    stages.json
    archive.json
  logs/
```

The ZIP root contains `images/`, all `sparse/N` components, optional `masks/`, `normals/`, `depths/`, and JSON reports/settings. The optional intermediate-file switch adds `features/` and `matches.bin`; local intermediates are retained regardless. The archive's SHA-256 and byte size are recorded in `reports/archive.json`.

`sfm_report.json` distinguishes:

- `coverage`: union of registered images across components, divided by input images.
- `selected_coverage`: images in the largest component divided by input images.
- `dominant_share`: largest component's images divided by the registered union.
- `selected_model`: model used for geometry. Other components remain in the archive; their coordinate systems are independent.

SfM exit code 3 is accepted only when complete binary camera/image/point tables validate and all referenced images exist. Other failures retain local files and logs. Geometry output is decoded and checked for every registered image in the selected model. A known exit-time -11 crash is tolerated only with a completion marker and all valid maps, with an explicit warning. Re-running geometry overwrites maps and clears previous success so corrupt files cannot be silently skipped.

Logs and results are written to `/content/spirula_preprocessing/<run>` first. If Drive disconnects, reconnect it and rerun **Drive aktarımını yeniden dene** before closing Colab. GPU and input work are not repeated by that cell. Heavy preprocessing stages have no automatic wall-time cutoff; Colab's own session limits still apply.

## Desktop parity and deliberate boundaries

- Uses the official checksum-pinned **Spirula 2026.9.24 / 183b2c6** Linux binary, the same existing installer and NVIDIA Vulkan recovery as the training notebook. No source build.
- Device selection is the Colab NVIDIA GPU; local file browse and the live reconstruction preview are replaced by Drive paths and notebook logs/reports.
- Mirrors the main frame/camera/matching/mask/geometry options. It does not copy the full native GUI: brush/click mask editing, per-lens rig editing, importing existing datasets and arbitrary raw SfM flags are not exposed here.
- Supported photo inputs: JPG/JPEG, PNG, BMP. EXR is explicitly rejected; unsupported files are not a promise of HDR support. Videos are extracted to PNG without a resolution downscale.
- Sharpness uses the requested source-frame window. FPS is approximated by an integer source-frame skip and recorded in `extraction.json`; adaptive mode varies spacing around this rate.
- Video masks use SAM 3 independent-image mode, matching the desktop's default of no temporal memory. Photo masks isolate each photo in a one-frame session because the pinned CLI's folder tracker always enables memory. This is correct for unrelated photos but reloads the model per photo and is slower than the native GUI's in-process photo loop.
- Object masking selects checksum-verified SAM 3 Q4 or F16. Normal/depth generation selects any model ID in the pinned binary: MoGe-2 S/B/L or Metric3D small/large/giant2. Metric3D ignores the MoGe token setting.
- Geometry maps cover the selected largest component. The full Spirula trainer supports the exposed camera families. Our existing hybrid importer supports only one camera; compatible choices from this form are simple-pinhole, pinhole and OpenCV. Its additional SIMPLE_RADIAL (ID 2) support is not the Spirula radial (ID 3) choice in this form.

## API and maintenance

- `GET /notebooks/static/preprocess`: versioned field catalog.
- `POST /notebooks/static/preprocess/validate`: validate/canonicalize a preset.
- `POST /notebooks/static/preprocess`: download a clean `.ipynb`; never executes it.

Implementation: `backend/notebooks/spirula_preprocess.py`, the two `templates/spirula_preprocess_*.py` files, and `frontend/src/notebook/SpirulaPreprocessPanel.tsx`. Runtime setup is selected from the maintained `spirula.ipynb` with explicit drift checks. The generated recipe has no repository dependencies.

Verification: pytest exercises generated cells with an external-command fake, tiny valid COLMAP tables, partial reconstructions, masks, corrupt/truncated data, geometry crashes and Drive outages. Frontend tests cover payloads, dependent controls, bad paths and invalid preset import. A real Colab inference run remains the end-to-end GPU check; local tests do not establish reconstruction quality.

References used for option mapping: [pinned CLI and GUI source](https://github.com/harry7557558/spirula-studio/tree/183b2c6df72f42ecb9a0500cdad96749847da171/src), the installed binary's `sfm auto --help`, `sam extract --help`, `sam --help`, and `geometry --help`. Spirula Studio remains the preprocessing engine and is credited in the app and notebook. Model licenses remain with their publishers.
