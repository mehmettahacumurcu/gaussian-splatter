import { useCallback, useEffect, useMemo, useReducer } from "react";
import { generateStaticNotebook, getStaticNotebookPresets } from "../api";
import { resolveDriveFolder } from "./drivePath";
import { DriveFolderSection } from "./DriveFolderSection";
import { FrameSelectionSection } from "./FrameSelectionSection";
import { NotebookAdvancedConfig } from "./NotebookAdvancedConfig";
import { NotebookDownload } from "./NotebookDownload";
import { NotebookReview } from "./NotebookReview";
import { QualitySection } from "./QualitySection";
import {
  buildStaticNotebookRunSpec,
  initialNotebookGeneratorState,
  notebookReducer,
  validateNotebookDraft,
} from "./reducer";
import type { NotebookAdvancedDraft, StaticNotebookRunSpec } from "./types";
import "./notebook.css";

export function NotebookGeneratorPanel() {
  const [state, dispatch] = useReducer(
    notebookReducer,
    undefined,
    initialNotebookGeneratorState,
  );
  const loadPresets = useCallback(() => {
    let active = true;
    dispatch({ type: "presets_loading" });
    getStaticNotebookPresets().then(
      (value) => active && dispatch({ type: "presets_succeeded", value }),
      (error) => active && dispatch({ type: "presets_failed", message: String(error) }),
    );
    return () => { active = false; };
  }, []);
  useEffect(() => loadPresets(), [loadPresets]);

  const resolution = resolveDriveFolder(state.draft.inputFolderRaw);
  const validation = useMemo(
    () => state.presets.status === "ready"
      ? validateNotebookDraft(state.draft, state.presets.value)
      : null,
    [state.draft, state.presets],
  );
  const spec = useMemo<StaticNotebookRunSpec | null>(() => {
    if (!validation?.valid || state.presets.status !== "ready") return null;
    return buildStaticNotebookRunSpec(state.draft, state.presets.value);
  }, [state.draft, state.presets, validation]);
  const profile = state.presets.status === "ready"
    ? state.presets.value.profiles.find((item) => item.id === state.draft.quality.profile) ?? null
    : null;

  const handleGenerate = useCallback(async () => {
    if (!spec) return;
    dispatch({ type: "generation_started" });
    try {
      const artifact = await generateStaticNotebook(spec);
      dispatch({ type: "generation_succeeded", artifact });
    } catch (error) {
      dispatch({ type: "generation_failed", message: String(error) });
    }
  }, [spec]);

  if (state.presets.status === "loading") {
    return <div className="notebook-generator nb-load-state"><p>Loading quality profiles…</p></div>;
  }
  if (state.presets.status === "error") {
    return (
      <div className="notebook-generator nb-load-state" role="alert">
        <p>Quality profiles could not be loaded: {state.presets.message}</p>
        <button type="button" onClick={loadPresets}>Retry profile loading</button>
      </div>
    );
  }

  return (
    <div className="notebook-generator">
      <header className="nb-hero">
        <p className="nb-eyebrow">Portable static Gaussian splat</p>
        <h1>Capture to splat, in one notebook.</h1>
        <p>Choose what is already in Drive. The generated notebook handles frames, COLMAP, quality gates, training, polish, and a verified result folder.</p>
      </header>
      <div className="nb-flow">
        <DriveFolderSection
          value={state.draft.inputFolderRaw}
          resolution={resolution}
          onChange={(value) => dispatch({ type: "input_changed", value })}
        />
        <FrameSelectionSection
          mode={state.draft.frameSelection.mode}
          fixedFps={state.draft.frameSelection.fixedFps}
          fixedFpsError={validation?.errors.fixedFps}
          onModeChange={(value) => dispatch({ type: "selection_mode_changed", value })}
          onFixedFpsChange={(value) => dispatch({ type: "fixed_fps_changed", value })}
        />
        <QualitySection
          profiles={state.presets.value.profiles}
          value={state.draft.quality}
          errors={validation?.errors ?? {}}
          onProfileChange={(value) => dispatch({ type: "profile_changed", value })}
          onIterationsChange={(value) => dispatch({ type: "iterations_changed", value })}
          onMaxGaussiansChange={(value) => dispatch({ type: "max_gaussians_changed", value })}
          onResetOverrides={() => dispatch({ type: "primary_overrides_reset" })}
        />
        <NotebookAdvancedConfig
          value={state.draft.quality.advanced}
          errors={validation?.errors ?? {}}
          onChange={<K extends keyof NotebookAdvancedDraft>(field: K, value: NotebookAdvancedDraft[K]) =>
            dispatch({ type: "advanced_changed", field, value })}
        />
        <NotebookReview
          spec={spec}
          profile={profile}
          path={validation?.path ?? resolution}
          warnings={validation?.warnings ?? []}
          generating={state.generation.status === "generating"}
          onGenerate={handleGenerate}
        />
      </div>
      {state.generation.status === "error" && (
        <div className="nb-generation-error" role="alert">
          <p>Notebook generation failed: {state.generation.message}</p>
          <button type="button" onClick={handleGenerate}>Retry generation</button>
        </div>
      )}
      {state.generation.status === "ready" && (
        <NotebookDownload
          artifact={state.generation.artifact}
          onRegenerate={handleGenerate}
          onEdit={() => dispatch({ type: "generation_reset" })}
        />
      )}
    </div>
  );
}
