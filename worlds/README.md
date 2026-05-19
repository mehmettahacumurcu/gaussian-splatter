# B Sub-Project Test Fixtures

These fixtures exercise the single-image full-scene splat reconstruction pipeline
(spec: `docs/superpowers/specs/2026-05-19-single-image-fullscene-splat-design.md`).

## Image sources

All fixture images are CC-licensed public-domain photographs from
[Wikimedia Commons](https://commons.wikimedia.org/).

| Slug | Source URL | License | Notes |
|---|---|---|---|
| `fixture-a-render` | Downscaled crop of `fixture-b-empty-photo` to 768×576 | (derived from CC-BY-SA below) | Acts as the "easy / synthetic-feel" fixture |
| `fixture-b-empty-photo` | https://commons.wikimedia.org/wiki/File:Empty_class_room.jpg | CC-BY-SA | Full-res empty classroom photograph |
| `fixture-c-photo-with-objects` | https://commons.wikimedia.org/wiki/File:Empty_classroom.jpg | CC-BY-SA | Classroom with rows of desks (foreground objects) |

If we need genuinely different categories of test images later (a CG render, an outdoor scene, etc.), replace the files at `worlds/<slug>/source/0-<slug>.<ext>` and update this table.

## Running the pipeline against a fixture

```powershell
cd C:\Users\TAHA\Desktop\gaussian-splatter\Gaussian Splatter\4dgs-studio
py -3 scripts/run_b_fixture.py --scene fixture-a-render --profile fast
```

`--profile` is one of `fast` (15 views, 1500 iters) | `default` (30 / 3000) | `quality` (50 / 5000).
