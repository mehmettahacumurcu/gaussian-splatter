# Spirula preprocessing notebook implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline; the user has approved implementation.

**Goal:** Configure Spirula dataset preparation in the app and export a portable Colab notebook without training.

**Architecture:** A validated backend settings catalog drives the form and notebook. A standalone Python recipe prepares inputs, invokes the pinned CLI, validates outputs and exports the dataset. The existing Spirula installer and Vulkan recovery are reused from the maintained training template.

**Tech stack:** FastAPI/Pydantic, nbformat, React/TypeScript, pytest/Vitest.

**Spec:** The approved conversation: familiar Spirula preprocessing groups, video/photos, camera/matching/masks/geometry settings, reusable presets, Drive output and quality report. Training is a separate downstream choice.

## Constraints and review focus

- Use Spirula v2026.9.24 / 183b2c6; verify flags with its CLI and upstream source.
- No local preprocessing, GPU job, driver install or training during generation.
- Preserve all reconstruction components; partial exit 3 is usable only after binary model validation.
- Scope geometry maps to the largest component and report that choice; keep other components intact.
- Serialize paths/prompts as literals and pass argument arrays, never shell commands.
- Drive outages must preserve local ZIP/logs with a retry cell.
- Check malformed COLMAP names, masks with different filenames, multi-input collisions, missing maps after a geometry crash, and offline export.

## Tasks

- [x] Add failing backend tests exercising generated recipe with tiny COLMAP fixtures and a fake external CLI boundary; test paths, commands, partial results, validation and archive contents.
- [x] Implement strict preprocessing schema/catalog, standalone recipe and notebook builder; reuse pinned runtime cells with explicit drift checks; expose catalog/validate/download routes.
- [x] Add failing frontend tests for selecting the new workflow, changing dependent settings and generating the selected payload; build the Spirula-style grouped form with preset save/load.
- [x] Run notebook/backend tests and frontend tests/build; inspect the UI and a generated notebook. Document output layout and differences from desktop (cloud GPU, no interactive mask editor).
- [x] Independently review the diff and fix findings.

Verification: all 81 backend notebook tests and 31 frontend notebook/API tests passed; the production frontend build succeeded. A notebook downloaded through the UI validates as nbformat and Python syntax. The real IMG_5966 COLMAP model validates with 923 registered images. GPU inference in Colab remains unexecuted.

Delivery branch: `feature/spirula-preprocessing-notebook` (commit and push per project policy).
