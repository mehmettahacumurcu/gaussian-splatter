import { useEffect, useRef, useState } from "react";
import { generatePipelineNotebook, getPipelineNotebookPresets } from "../api";
import { resolveDriveFolder } from "./drivePath";
import { NotebookDownload } from "./NotebookDownload";
import type { PipelinePreset, PipelinePresets, TextToSplatNotebookSpec } from "./PipelineNotebookPanel";
import type { GeneratedNotebook } from "./types";

function integerInRange(value: string, minimum: number, maximum: number): boolean {
  return value.trim() !== "" && Number.isInteger(Number(value)) && Number(value) >= minimum && Number(value) <= maximum;
}

export function TextToSplatPanel() {
  const [catalog, setCatalog] = useState<PipelinePresets | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [retry, setRetry] = useState(0);
  const [prompt, setPrompt] = useState("");
  const [negative, setNegative] = useState("");
  const [style, setStyle] = useState("");
  const [seed, setSeed] = useState("42");
  const [preset, setPreset] = useState<PipelinePreset>("baseline");
  const [outputDir, setOutputDir] = useState("");
  const [targetCount, setTargetCount] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [artifact, setArtifact] = useState<GeneratedNotebook | null>(null);
  const alive = useRef(true);

  useEffect(() => {
    alive.current = true;
    return () => { alive.current = false; };
  }, []);
  useEffect(() => {
    let active = true;
    setCatalogError("");
    getPipelineNotebookPresets().then(value => {
      if (!active) return;
      if (!value.presets.text_to_splat) {
        setCatalogError("Metinden splat ayarları bulunamadı. Backend sürümünü kontrol et.");
      } else setCatalog(value);
    }, err => { if (active) setCatalogError(`Hazır ayarlar yüklenemedi: ${String(err)}`); });
    return () => { active = false; };
  }, [retry]);

  const defaults = catalog?.presets.text_to_splat?.[preset];
  const output = outputDir.trim() ? resolveDriveFolder(outputDir, { allowResultFolder: true }) : null;
  const promptValid = prompt.trim().length >= 1 && Array.from(prompt.trim()).length <= 2000;
  const seedValid = integerInRange(seed, 0, 2147483647);
  const countValid = !targetCount.trim() || integerInRange(targetCount, 1000, 2000000);
  const valid = promptValid && seedValid && countValid && (!output || output.ok)
    && Array.from(negative).length <= 2000 && Array.from(style).length <= 500 && Boolean(defaults);

  async function generate() {
    if (!valid || loading) return;
    const spec: TextToSplatNotebookSpec = {
      pipeline: "text_to_splat", input_mode: "text", input_path: "",
      prompt: prompt.trim(), negative_prompt: negative.trim(), style: style.trim(),
      seed: Number(seed), preset, output_dir: output?.ok ? output.canonicalInput : null,
      target_splat_count: targetCount.trim() ? Number(targetCount) : null,
    };
    setLoading(true); setError(""); setArtifact(null);
    try {
      const result = await generatePipelineNotebook(spec);
      if (alive.current) setArtifact(result);
    } catch (err) {
      if (alive.current) setError(`Notebook oluşturulamadı: ${String(err)}`);
    } finally {
      if (alive.current) setLoading(false);
    }
  }

  return <div className="notebook-generator">
    <header className="nb-hero">
      <p className="nb-eyebrow">Morph için hedef B</p>
      <h1>Metinden hedef splat.</h1>
      <p>Bir nesne tarif et. Colab notebook’u önce SDXL ile bir görsel, ardından TRELLIS ile standart 3DGS PLY üretir. Sonucu Scene Editor’a ekleyip morph hedefi olarak seçebilirsin.</p>
    </header>
    {catalogError && <div className="nb-generation-error" role="alert"><p>{catalogError}</p>
      <button type="button" onClick={() => setRetry(value => value + 1)}>Yeniden dene</button>
    </div>}
    {!catalog && !catalogError && <p role="status">Hazır ayarlar yükleniyor…</p>}
    <form className="nb-flow" onSubmit={event => { event.preventDefault(); void generate(); }} onChange={() => setArtifact(null)}>
      <fieldset disabled={loading} className="nb-stage-body nb-pipeline-fields">
        <legend>Hedef nesne ve Colab ayarları</legend>
        <label>Nesne tarifi (PROMPT)
          <textarea value={prompt} rows={4} required maxLength={2000} placeholder="a red sports car" aria-describedby="text-splat-prompt-help"
            onChange={event => setPrompt(event.target.value)} aria-invalid={Boolean(prompt && !promptValid)} />
        </label>
        <p id="text-splat-prompt-help" className="nb-help">Tek bir nesneyi tarif et; İngilizce tarifler önerilir. En fazla 2000 karakter.</p>
        {prompt && !promptValid && <p className="nb-error" role="alert">Tarif boş olamaz ve 2000 karakteri geçemez.</p>}
        <div className="nb-inline-fields">
          <label>Kalite ayarı<select value={preset} onChange={event => setPreset(event.target.value as PipelinePreset)}>
            <option value="baseline">Dengeli · L4 24 GB</option>
            <option value="quality">Kaliteli · L4 24 GB</option>
            <option value="ultra">Ultra · A100 40/80 GB</option>
          </select></label>
          <label>Tohum (SEED)<input type="number" min="0" max="2147483647" step="1" required value={seed}
            onChange={event => setSeed(event.target.value)} aria-invalid={!seedValid} /></label>
        </div>
        {!seedValid && <p className="nb-error" role="alert">Tohum 0–2147483647 arasında bir tam sayı olmalı.</p>}
        {defaults && <p className="nb-help">En az {defaults.min_vram} GiB GPU belleği gerekir. Daha fazla adım daha iyi sonucu garanti etmez.</p>}
        <details className="nb-advanced">
          <summary>İsteğe bağlı ayarlar</summary>
          <div className="nb-advanced-grid">
            <label className="nb-wide-field">İstenmeyen özellikler (NEGATIVE)<textarea value={negative} rows={2} maxLength={2000}
              placeholder="blurry, multiple objects, text, watermark" onChange={event => setNegative(event.target.value)} /></label>
            <label className="nb-wide-field">Görsel tarzı (STYLE)<input value={style} maxLength={500}
              placeholder="studio product photo, white background" onChange={event => setStyle(event.target.value)} /></label>
            <label className="nb-wide-field">Drive çıktı klasörü (OUTPUT_DIR)<input value={outputDir}
              placeholder="GaussianTests/text_to_splat/&lt;tarif&gt;" aria-invalid={Boolean(output && !output.ok)}
              onChange={event => setOutputDir(event.target.value)} aria-describedby="text-splat-output-help" /></label>
            <p id="text-splat-output-help" className="nb-help nb-wide-field">Boş bırakırsan MyDrive/GaussianTests/text_to_splat/ altında tariften türetilen klasör kullanılır. Özel yol MyDrive’a göre yazılır.</p>
            {output && !output.ok && <p className="nb-error nb-wide-field" role="alert">MyDrive altında göreli bir klasör yaz; yerel bilgisayar yolu, boş yol parçası veya “..” kullanma.</p>}
            <label>Hedef splat sayısı<input type="number" min="1000" max="2000000" step="1" value={targetCount}
              placeholder="Otomatik" aria-invalid={!countValid} onChange={event => setTargetCount(event.target.value)} /></label>
            <p className="nb-help">Boş bırakmak üretilen sayıyı korur. Bu bir üst sınırdır; fazlası kayıplı örneklenir, eksik splat eklenmez. İlk morph denemesi için 200000 önerilir.</p>
            {!countValid && <p className="nb-error nb-wide-field" role="alert">Hedef splat sayısı 1000–2000000 arasında bir tam sayı olmalı.</p>}
          </div>
        </details>
        <p className="nb-help">Notebook’u Colab’de aç, L4 veya A100 GPU seç ve “Tümünü çalıştır” komutunu kullan. PLY, dönüş önizlemesi, ara görsel ve manifest Drive’a kaydedilir.</p>
        <button type="submit" className="nb-generate-button" disabled={!valid || loading}>
          {loading ? "Notebook hazırlanıyor…" : "Metinden splat notebook’unu indir"}
        </button>
      </fieldset>
    </form>
    {error && <p role="alert" className="nb-error">{error}</p>}
    {artifact && <NotebookDownload artifact={artifact} onRegenerate={() => void generate()} onEdit={() => setArtifact(null)} />}
  </div>;
}
