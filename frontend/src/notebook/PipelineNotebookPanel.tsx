import { useEffect, useRef, useState } from "react";
import { generatePipelineNotebook, getPipelineNotebookPresets } from "../api";
import { NotebookDownload } from "./NotebookDownload";
import { resolveDriveFolder } from "./drivePath";
import type { GeneratedNotebook } from "./types";
import type { TextToSplatModelSettings } from "./textToSplatSettings";

export type TrainingPipeline = "hybrid" | "spirula";
export type ExternalPipeline = TrainingPipeline | "text_to_splat";
export type PipelinePreset = "baseline" | "quality" | "ultra";
export interface TrainingPipelineNotebookSpec {
  pipeline: TrainingPipeline;
  input_mode: "video" | "dataset_zip" | "dataset_folder";
  input_path: string;
  preset: PipelinePreset;
  iterations: number | null;
  max_gaussians: number | null;
  fps: number;
  model_name: string;
  allow_partial: boolean;
  generate_depth: boolean;
  geometry_model: "moge2-vits" | "moge2-vitb" | "moge2-vitl";
  sfm_quality: "high" | "extreme";
}
export interface TextToSplatNotebookSpec extends TextToSplatModelSettings {
  pipeline: "text_to_splat";
  input_mode: "text";
  input_path: "";
  prompt: string;
  negative_prompt: string;
  style: string;
  seed: number;
  preset: PipelinePreset;
  output_dir: string | null;
  target_splat_count: number | null;
}
export type PipelineNotebookSpec = TrainingPipelineNotebookSpec | TextToSplatNotebookSpec;
export interface PipelinePresets {
  template_version: number;
  presets: Record<TrainingPipeline, Record<PipelinePreset, {
    iterations: number; max_gaussians: number; min_vram: number; recipe: string;
  }>> & { text_to_splat: Record<PipelinePreset, {
    min_vram: number; recipe: string; image_steps: number; sparse_steps: number; slat_steps: number; resolution: number;
  }> };
}

export function PipelineNotebookPanel({ pipeline }: { pipeline: TrainingPipeline }) {
  const [catalog, setCatalog] = useState<PipelinePresets | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [artifact, setArtifact] = useState<GeneratedNotebook | null>(null);
  const alive = useRef(true);
  const [inputPath, setInputPath] = useState("");
  const [mode, setMode] = useState<TrainingPipelineNotebookSpec["input_mode"]>(pipeline === "spirula" ? "video" : "dataset_zip");
  const [preset, setPreset] = useState<PipelinePreset>("baseline");
  const [iterations, setIterations] = useState("");
  const [cap, setCap] = useState("");
  const [fps, setFps] = useState("4");
  const [model, setModel] = useState("0");
  const [partial, setPartial] = useState(false);
  const [depth, setDepth] = useState(true);
  const [geometry, setGeometry] = useState<TrainingPipelineNotebookSpec["geometry_model"]>("moge2-vitb");
  const [sfm, setSfm] = useState<TrainingPipelineNotebookSpec["sfm_quality"]>("high");
  useEffect(() => {
    alive.current = true;
    getPipelineNotebookPresets().then(
      (value) => { if (alive.current) setCatalog(value); },
      (err) => { if (alive.current) setError(String(err)); },
    );
    return () => { alive.current = false; };
  }, []);
  const path = resolveDriveFolder(inputPath);
  const defaults = catalog?.presets[pipeline][preset];
  async function generate() {
    if (!path.ok || !defaults) return;
    setLoading(true); setError(""); setArtifact(null);
    try {
      const result = await generatePipelineNotebook({
        pipeline, input_mode: mode, input_path: path.canonicalInput, preset,
        iterations: iterations ? Number(iterations) : null,
        max_gaussians: cap ? Number(cap) : null, fps: Number(fps), model_name: model,
        allow_partial: partial, generate_depth: depth, geometry_model: geometry, sfm_quality: sfm,
      });
      if (alive.current) setArtifact(result);
    } catch (err) {
      if (alive.current) setError(String(err));
    } finally {
      if (alive.current) setLoading(false);
    }
  }
  return <div className="notebook-generator">
    <header className="nb-hero">
      <h1>{pipeline === "hybrid" ? "Spirula dataset → our trainer" : "Spirula from capture to splat"}</h1>
      <p>{pipeline === "hybrid"
        ? "Use an existing dataset ZIP or folder containing images/ and sparse/. Preprocessing is not repeated. Normals and depth maps are not used by this trainer."
        : "Use a video or prepared dataset. Includes the pinned Spirula release, GPU setup, reconstruction checks, geometry and training."}</p>
    </header>
    <form className="nb-flow" onSubmit={(e) => { e.preventDefault(); void generate(); }} onChange={() => setArtifact(null)}>
      <fieldset disabled={loading} className="nb-stage-body nb-pipeline-fields">
        <legend>Input and training</legend>
        <label>Input type<select value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
          {pipeline === "spirula" && <option value="video">Video</option>}
          <option value="dataset_zip">Dataset ZIP</option><option value="dataset_folder">Dataset folder</option>
        </select></label>
        <label>MyDrive path<input value={inputPath} onChange={(e) => setInputPath(e.target.value)} placeholder="GaussianTests/inputs/capture.MOV" required /></label>
        {inputPath && !path.ok && <p role="alert">{path.error}</p>}
        <label>Training preset<select value={preset} onChange={(e) => { setPreset(e.target.value as PipelinePreset); setIterations(""); setCap(""); }}>
          <option value="baseline">Baseline</option><option value="quality">Quality experiment</option><option value="ultra">Ultra experiment</option>
        </select></label>
        {defaults && <p>{defaults.iterations.toLocaleString()} steps · {defaults.max_gaussians.toLocaleString()} Gaussians · {defaults.min_vram} GiB minimum VRAM. Larger budgets do not guarantee better quality.</p>}
        <label>Training steps<input type="number" min="1000" max="120000" step="1" value={iterations} placeholder={String(defaults?.iterations ?? "")} onChange={(e) => setIterations(e.target.value)} /></label>
        <label>Gaussian budget<input type="number" min="50000" max="20000000" step="1" value={cap} placeholder={String(defaults?.max_gaussians ?? "")} onChange={(e) => setCap(e.target.value)} /></label>
        {pipeline === "hybrid" ? <label>COLMAP model folder<input value={model} pattern="[0-9]+" required onChange={(e) => setModel(e.target.value)} /></label> : <>
          {mode === "video" && <label>Extraction target FPS<input type="number" min="1" max="30" required value={fps} onChange={(e) => setFps(e.target.value)} /></label>}
          <label>SfM quality<select value={sfm} onChange={(e) => setSfm(e.target.value as typeof sfm)}><option value="high">High</option><option value="extreme">Extreme</option></select></label>
          <label>Geometry model<select value={geometry} onChange={(e) => setGeometry(e.target.value as typeof geometry)}><option value="moge2-vits">MoGe2 small</option><option value="moge2-vitb">MoGe2 base</option><option value="moge2-vitl">MoGe2 large</option></select></label>
          <label><input type="checkbox" checked={depth} onChange={(e) => setDepth(e.target.checked)} /> Generate depth guidance</label>
          <label><input type="checkbox" checked={partial} onChange={(e) => setPartial(e.target.checked)} /> Allow training only the largest component when reconstruction is incomplete</label>
        </>}
        <p>Download creates a notebook only. Select a GPU and start it in Colab. Results and logs are saved to a separate Drive run folder.</p>
        <button type="submit" className="nb-generate-button" disabled={!path.ok || !defaults || loading}>{loading ? "Generating notebook…" : "Generate and download notebook"}</button>
      </fieldset>
    </form>
    {error && <p role="alert">{error}</p>}
    {artifact && <NotebookDownload artifact={artifact} onRegenerate={() => void generate()} onEdit={() => setArtifact(null)} />}
  </div>;
}
